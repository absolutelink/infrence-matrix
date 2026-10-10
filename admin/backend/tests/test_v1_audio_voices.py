"""Client ``GET /v1/audio/voices`` proxy (Phase 24; boot-on-demand P25;
cache-first with the voice-catalog cache).

The live agent catalog is the source of truth. The route resolves in order and
**falls through** on any earlier-path failure so a client never dead-ends when a
cache exists: a **loaded** backend is proxied slot-free (no scheduler slot); a
transport failure or non-2xx there is treated as a cache miss and falls to the
**cached** catalog (from ``backend.metadata``, served with ``"cached": true``
and no boot); only when neither yields a 200 does it **boot on demand** through
the scheduler (acquire -> proxy -> release). Asserts:

- 200 relays the agent's JSON catalog and leaves no slot held;
- a loaded backend is proxied without taking a scheduler slot;
- a cached catalog is served (``"cached": true``) with no boot at all;
- a loaded-path transport failure / non-2xx falls through to the cache;
- a live read refreshes the cache onto the definition (preserving other keys);
- boot-on-demand (stopped + no cache) acquires, proxies, and releases the slot;
- 404 unknown / disabled / non-tts alias;
- 503 when no connected agent hosts the definition;
- an agent 503 with no cache falls through to boot-on-demand and passes 503.
"""

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import ProviderDefinition
from app.services.connection_manager import manager
from app.services.wire import Frame
from tests.helpers import (
    get_or_create_machine,
    make_agent,
    make_definition,
    make_instance,
)

AGENT = "http://127.0.0.1:8081"


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def seed_tts(
    session: Session,
    *,
    alias: str,
    machine_uid: str,
    modality: str = "tts",
    connected: bool = True,
    running: bool = True,
) -> ProviderDefinition:
    machine = get_or_create_machine(session, uid=machine_uid, host="127.0.0.1")
    agent = make_agent(session, machine, connected=connected)
    definition = make_definition(
        session,
        alias=alias,
        provider_type="mock",
        modality=modality,
        backend_config={"model": {"file": "m.gguf"}},
    )
    make_instance(
        session,
        agent,
        definition,
        backend_status="running" if running else "stopped",
    )
    return definition


@respx.mock
def test_voices_relays_agent_catalog(client: TestClient, session: Session) -> None:
    # A loaded backend (running=True) is proxied slot-free: no scheduler slot is
    # taken, so active_count stays 0 (nothing to release).
    seed_tts(session, alias="v-a", machine_uid="v-a-m")
    respx.get(f"{AGENT}/v1/audio/voices").mock(
        return_value=httpx.Response(
            200, json={"object": "list", "data": [{"voice_id": "alloy"}]}
        )
    )
    resp = client.get("/v1/audio/voices", params={"model": "v-a"})
    assert resp.status_code == 200
    assert resp.json()["data"][0]["voice_id"] == "alloy"
    assert client.app.state.scheduler.active_count("v-a") == 0


def test_voices_cache_hit_without_boot(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopped backend + a cached catalog -> served from cache, no boot."""
    definition = seed_tts(
        session, alias="v-cache", machine_uid="v-cache-m", running=False
    )
    definition.model_metadata = {
        "models": [
            {"id": "v-cache", "voices": [{"voice": "Vivian", "model": "v-cache"}]}
        ]
    }
    session.add(definition)
    session.commit()

    called = {"acquire": False}
    scheduler = client.app.state.scheduler
    real_acquire = scheduler.acquire

    async def spy(*args: object, **kwargs: object) -> object:
        called["acquire"] = True
        return await real_acquire(*args, **kwargs)

    monkeypatch.setattr(scheduler, "acquire", spy)

    resp = client.get("/v1/audio/voices", params={"model": "v-cache"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cached"] is True
    assert body["voices"] == [{"voice": "Vivian", "model": "v-cache"}]
    assert called["acquire"] is False


@respx.mock
def test_voices_live_read_refreshes_cache(
    client: TestClient, session: Session
) -> None:
    """A loaded live read seeds/refreshes the definition's voice cache."""
    definition = seed_tts(session, alias="v-live", machine_uid="v-live-m", running=True)
    respx.get(f"{AGENT}/v1/audio/voices").mock(
        return_value=httpx.Response(
            200, json={"voices": [{"voice": "Vivian", "model": "v-live"}]}
        )
    )
    resp = client.get("/v1/audio/voices", params={"model": "v-live"})
    assert resp.status_code == 200
    session.refresh(definition)
    entry = next(
        m for m in definition.model_metadata["models"] if m["id"] == "v-live"
    )
    assert entry["voices"] == [{"voice": "Vivian", "model": "v-live"}]


def test_voices_unknown_alias_404(client: TestClient) -> None:
    resp = client.get("/v1/audio/voices", params={"model": "nope"})
    assert resp.status_code == 404


def test_voices_asr_alias_404(client: TestClient, session: Session) -> None:
    seed_tts(session, alias="v-asr", machine_uid="v-asr-m", modality="asr")
    resp = client.get("/v1/audio/voices", params={"model": "v-asr"})
    assert resp.status_code == 404


def test_voices_no_connected_agent_503(client: TestClient, session: Session) -> None:
    seed_tts(session, alias="v-off", machine_uid="v-off-m", connected=False)
    resp = client.get("/v1/audio/voices", params={"model": "v-off"})
    assert resp.status_code == 503


@respx.mock
def test_voices_agent_503_passthrough(client: TestClient, session: Session) -> None:
    # DB says running (loaded) but the agent races to not-ready -> its 503 is a
    # cache miss (no cache seeded) so the route falls through to boot-on-demand,
    # which re-proxies and passes the agent's 503 through untouched. The slot is
    # released (active_count 0).
    seed_tts(session, alias="v-notready", machine_uid="v-nr-m", running=True)
    respx.get(f"{AGENT}/v1/audio/voices").mock(
        return_value=httpx.Response(503, json={"detail": "backend is stopped"})
    )
    resp = client.get("/v1/audio/voices", params={"model": "v-notready"})
    assert resp.status_code == 503
    assert client.app.state.scheduler.active_count("v-notready") == 0


@respx.mock
def test_voices_loaded_transport_error_falls_back_to_cache(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A loaded backend that fails at the TRANSPORT level must not dead-end the
    cache: fall through to the cached catalog (no boot)."""
    definition = seed_tts(session, alias="v-te", machine_uid="v-te-m", running=True)
    definition.model_metadata = {
        "models": [{"id": "v-te", "voices": [{"voice": "Vivian", "model": "v-te"}]}]
    }
    session.add(definition)
    session.commit()
    respx.get(f"{AGENT}/v1/audio/voices").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    called = {"acquire": False}
    scheduler = client.app.state.scheduler
    real_acquire = scheduler.acquire

    async def spy(*args: object, **kwargs: object) -> object:
        called["acquire"] = True
        return await real_acquire(*args, **kwargs)

    monkeypatch.setattr(scheduler, "acquire", spy)

    resp = client.get("/v1/audio/voices", params={"model": "v-te"})
    assert resp.status_code == 200
    assert resp.json()["cached"] is True
    assert resp.json()["voices"] == [{"voice": "Vivian", "model": "v-te"}]
    assert called["acquire"] is False


@respx.mock
def test_voices_loaded_non200_falls_back_to_cache(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A loaded backend returning a non-2xx (503) is treated as a cache miss and
    falls through to the cached catalog rather than surfacing the error."""
    definition = seed_tts(session, alias="v-n2", machine_uid="v-n2-m", running=True)
    definition.model_metadata = {
        "models": [{"id": "v-n2", "voices": [{"voice": "Nova", "model": "v-n2"}]}]
    }
    session.add(definition)
    session.commit()
    respx.get(f"{AGENT}/v1/audio/voices").mock(
        return_value=httpx.Response(503, json={"detail": "not ready"})
    )

    called = {"acquire": False}
    scheduler = client.app.state.scheduler
    real_acquire = scheduler.acquire

    async def spy(*args: object, **kwargs: object) -> object:
        called["acquire"] = True
        return await real_acquire(*args, **kwargs)

    monkeypatch.setattr(scheduler, "acquire", spy)

    resp = client.get("/v1/audio/voices", params={"model": "v-n2"})
    assert resp.status_code == 200
    assert resp.json()["cached"] is True
    assert resp.json()["voices"] == [{"voice": "Nova", "model": "v-n2"}]
    assert called["acquire"] is False


@respx.mock
def test_voices_boot_on_demand_releases_slot(
    client: TestClient, session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopped + connected + no cache -> boot-on-demand: acquire -> proxy ->
    release (slot symmetry). The backend.start ack is stubbed (no live agent)."""

    async def fake_send_command(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        return Frame(type="ack", payload={"ok": True, "detail": {"capacity": 1}})

    monkeypatch.setattr(manager, "send_command", fake_send_command)
    seed_tts(session, alias="v-boot", machine_uid="v-boot-m", running=False)
    respx.get(f"{AGENT}/v1/audio/voices").mock(
        return_value=httpx.Response(
            200, json={"voices": [{"voice": "Vivian", "model": "v-boot"}]}
        )
    )
    resp = client.get("/v1/audio/voices", params={"model": "v-boot"})
    assert resp.status_code == 200
    assert resp.json()["voices"] == [{"voice": "Vivian", "model": "v-boot"}]
    # The boot-on-demand slot is released after the relay (no leak).
    assert client.app.state.scheduler.active_count("v-boot") == 0


@respx.mock
def test_voices_live_read_preserves_other_metadata_keys(
    client: TestClient, session: Session
) -> None:
    """A cache refresh replaces only the ``models`` key: unrelated top-level
    keys in ``model_metadata`` survive."""
    definition = seed_tts(session, alias="v-meta", machine_uid="v-meta-m", running=True)
    definition.model_metadata = {
        "models": [{"id": "v-meta", "voices": [{"voice": "stale"}]}],
        "capabilities": {"streaming": True},
    }
    session.add(definition)
    session.commit()
    respx.get(f"{AGENT}/v1/audio/voices").mock(
        return_value=httpx.Response(
            200, json={"voices": [{"voice": "Vivian", "model": "v-meta"}]}
        )
    )
    resp = client.get("/v1/audio/voices", params={"model": "v-meta"})
    assert resp.status_code == 200
    session.refresh(definition)
    assert definition.model_metadata["capabilities"] == {"streaming": True}
    entry = next(m for m in definition.model_metadata["models"] if m["id"] == "v-meta")
    assert entry["voices"] == [{"voice": "Vivian", "model": "v-meta"}]

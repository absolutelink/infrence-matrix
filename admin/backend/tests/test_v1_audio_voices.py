"""Client ``GET /v1/audio/voices`` proxy (Phase 24; boot-on-demand P25).

The live agent catalog is the source of truth. The route admits through the
scheduler (BOOTING a stopped backend on demand — the playground voice dropdown
must not dead-end on an idle-reaped model) and releases the slot after the
relay. Asserts:

- 200 relays the agent's JSON catalog and leaves no slot held;
- 404 unknown / disabled / non-tts alias;
- 503 when no connected agent hosts the definition;
- an agent 503 (backend raced to stopped after admission) is passed through.
"""

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlmodel import Session

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
) -> None:
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


@respx.mock
def test_voices_relays_agent_catalog(client: TestClient, session: Session) -> None:
    seed_tts(session, alias="v-a", machine_uid="v-a-m")
    respx.get(f"{AGENT}/v1/audio/voices").mock(
        return_value=httpx.Response(
            200, json={"object": "list", "data": [{"voice_id": "alloy"}]}
        )
    )
    resp = client.get("/v1/audio/voices", params={"model": "v-a"})
    assert resp.status_code == 200
    assert resp.json()["data"][0]["voice_id"] == "alloy"
    # The boot-on-demand slot is released after the relay (no leak).
    assert client.app.state.scheduler.active_count("v-a") == 0


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
    # Admitted (DB says running) but the agent races to not-ready -> its 503
    # passes through. (A DB-stopped backend is now BOOTED on demand instead,
    # so the not-ready case is simulated at the agent boundary.)
    seed_tts(session, alias="v-notready", machine_uid="v-nr-m", running=True)
    respx.get(f"{AGENT}/v1/audio/voices").mock(
        return_value=httpx.Response(503, json={"detail": "backend is stopped"})
    )
    resp = client.get("/v1/audio/voices", params={"model": "v-notready"})
    assert resp.status_code == 503
    assert client.app.state.scheduler.active_count("v-notready") == 0

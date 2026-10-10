"""Admin saved-voice enrollment (Phase 24): ``/admin/api/voices``.

The admin proxies the write to a running backend's agent HTTP surface and
persists the ``TTSVoice`` row only after the agent acks. Asserts list / enroll /
delete against a respx-stubbed agent, the tts-modality gate, name validation,
and the documented delete semantics (row dropped on 2xx/404, kept on 5xx).
"""

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import ProviderDefinition, TTSVoice
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


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------
def test_list_voices_empty(client: TestClient, session: Session) -> None:
    definition = seed_tts(session, alias="lv-a", machine_uid="lv-a-m")
    resp = client.get("/admin/api/voices", params={"definition_id": str(definition.id)})
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_voices_returns_rows(client: TestClient, session: Session) -> None:
    definition = seed_tts(session, alias="lv-b", machine_uid="lv-b-m")
    session.add(TTSVoice(definition_id=definition.id, name="alloy", language="en"))
    session.add(TTSVoice(definition_id=definition.id, name="echo"))
    session.commit()
    resp = client.get("/admin/api/voices", params={"definition_id": str(definition.id)})
    assert resp.status_code == 200
    names = [v["name"] for v in resp.json()]
    assert names == ["alloy", "echo"]


def test_list_voices_requires_tts(client: TestClient, session: Session) -> None:
    definition = seed_tts(
        session, alias="lv-llm", machine_uid="lv-llm-m", modality="llm"
    )
    resp = client.get("/admin/api/voices", params={"definition_id": str(definition.id)})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# enroll
# ---------------------------------------------------------------------------
@respx.mock
def test_enroll_proxies_and_persists(client: TestClient, session: Session) -> None:
    definition = seed_tts(session, alias="en-a", machine_uid="en-a-m")
    route = respx.put(f"{AGENT}/v1/audio/voices/myvoice").mock(
        return_value=httpx.Response(
            200, json={"object": "voice", "voice_id": "myvoice"}
        )
    )
    resp = client.post(
        "/admin/api/voices",
        data={
            "definition_id": str(definition.id),
            "name": "myvoice",
            "ref_text": "the quick brown fox",
            "language": "en",
        },
        files={"file": ("ref.wav", b"RIFF....fake", "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert resp.json()["voice"]["name"] == "myvoice"

    # The agent received the clip body + model/ref_text/language params.
    req = route.calls.last.request
    assert req.url.params["model"] == "en-a"
    assert req.url.params["ref_text"] == "the quick brown fox"
    assert req.url.params["language"] == "en"
    assert req.content == b"RIFF....fake"

    row = (
        session.query(TTSVoice)
        .filter(TTSVoice.definition_id == definition.id, TTSVoice.name == "myvoice")
        .first()
    )
    assert row is not None
    assert row.ref_text == "the quick brown fox"
    assert row.language == "en"


@respx.mock
def test_enroll_agent_503_not_ready(client: TestClient, session: Session) -> None:
    definition = seed_tts(session, alias="en-err", machine_uid="en-err-m")
    respx.put(f"{AGENT}/v1/audio/voices/bad").mock(
        return_value=httpx.Response(503, json={"detail": "backend not ready"})
    )
    resp = client.post(
        "/admin/api/voices",
        data={"definition_id": str(definition.id), "name": "bad"},
        files={"file": ("ref.wav", b"x", "audio/wav")},
    )
    # LOW-3: a not-ready agent 503 is passed through (no longer 200 ok:false).
    assert resp.status_code == 503
    assert session.query(TTSVoice).count() == 0


@respx.mock
def test_enroll_agent_5xx_maps_502(client: TestClient, session: Session) -> None:
    definition = seed_tts(session, alias="en-500", machine_uid="en-500-m")
    respx.put(f"{AGENT}/v1/audio/voices/bad").mock(
        return_value=httpx.Response(500, json={"detail": "engine boom"})
    )
    resp = client.post(
        "/admin/api/voices",
        data={"definition_id": str(definition.id), "name": "bad"},
        files={"file": ("ref.wav", b"x", "audio/wav")},
    )
    assert resp.status_code == 502
    assert session.query(TTSVoice).count() == 0


@respx.mock
def test_enroll_agent_4xx_relayed(client: TestClient, session: Session) -> None:
    definition = seed_tts(session, alias="en-400", machine_uid="en-400-m")
    respx.put(f"{AGENT}/v1/audio/voices/bad").mock(
        return_value=httpx.Response(400, json={"detail": "bad clip"})
    )
    resp = client.post(
        "/admin/api/voices",
        data={"definition_id": str(definition.id), "name": "bad"},
        files={"file": ("ref.wav", b"x", "audio/wav")},
    )
    assert resp.status_code == 400
    assert session.query(TTSVoice).count() == 0


def test_enroll_invalid_name_422(client: TestClient, session: Session) -> None:
    definition = seed_tts(session, alias="en-name", machine_uid="en-name-m")
    resp = client.post(
        "/admin/api/voices",
        data={"definition_id": str(definition.id), "name": "../etc/passwd"},
        files={"file": ("ref.wav", b"x", "audio/wav")},
    )
    assert resp.status_code == 422


def test_enroll_no_connected_agent_503(client: TestClient, session: Session) -> None:
    definition = seed_tts(
        session, alias="en-off", machine_uid="en-off-m", connected=False
    )
    resp = client.post(
        "/admin/api/voices",
        data={"definition_id": str(definition.id), "name": "v"},
        files={"file": ("ref.wav", b"x", "audio/wav")},
    )
    assert resp.status_code == 503


def test_enroll_requires_tts(client: TestClient, session: Session) -> None:
    definition = seed_tts(
        session, alias="en-llm", machine_uid="en-llm-m", modality="llm"
    )
    resp = client.post(
        "/admin/api/voices",
        data={"definition_id": str(definition.id), "name": "v"},
        files={"file": ("ref.wav", b"x", "audio/wav")},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------
@respx.mock
def test_delete_confirmed_drops_row(client: TestClient, session: Session) -> None:
    definition = seed_tts(session, alias="del-a", machine_uid="del-a-m")
    voice = TTSVoice(definition_id=definition.id, name="gone")
    session.add(voice)
    session.commit()
    session.refresh(voice)
    respx.delete(f"{AGENT}/v1/audio/voices/gone").mock(
        return_value=httpx.Response(200, json={"deleted": True})
    )
    resp = client.delete(f"/admin/api/voices/{voice.id}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert session.query(TTSVoice).count() == 0


@respx.mock
def test_delete_agent_404_still_drops_row(client: TestClient, session: Session) -> None:
    definition = seed_tts(session, alias="del-404", machine_uid="del-404-m")
    voice = TTSVoice(definition_id=definition.id, name="ghost")
    session.add(voice)
    session.commit()
    session.refresh(voice)
    respx.delete(f"{AGENT}/v1/audio/voices/ghost").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )
    resp = client.delete(f"/admin/api/voices/{voice.id}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert session.query(TTSVoice).count() == 0


@respx.mock
def test_delete_agent_503_keeps_row(client: TestClient, session: Session) -> None:
    definition = seed_tts(session, alias="del-503", machine_uid="del-503-m")
    voice = TTSVoice(definition_id=definition.id, name="stuck")
    session.add(voice)
    session.commit()
    session.refresh(voice)
    respx.delete(f"{AGENT}/v1/audio/voices/stuck").mock(
        return_value=httpx.Response(503, json={"detail": "not ready"})
    )
    resp = client.delete(f"/admin/api/voices/{voice.id}")
    # LOW-3: not-ready relays 503 and the row is kept for retry.
    assert resp.status_code == 503
    assert session.query(TTSVoice).count() == 1


@respx.mock
def test_delete_agent_5xx_maps_502_keeps_row(
    client: TestClient, session: Session
) -> None:
    definition = seed_tts(session, alias="del-500", machine_uid="del-500-m")
    voice = TTSVoice(definition_id=definition.id, name="boom")
    session.add(voice)
    session.commit()
    session.refresh(voice)
    respx.delete(f"{AGENT}/v1/audio/voices/boom").mock(
        return_value=httpx.Response(500, json={"detail": "engine boom"})
    )
    resp = client.delete(f"/admin/api/voices/{voice.id}")
    assert resp.status_code == 502
    assert session.query(TTSVoice).count() == 1


def test_delete_unknown_voice_404(client: TestClient) -> None:
    resp = client.delete(f"/admin/api/voices/{__import__('uuid').uuid4()}")
    assert resp.status_code == 404

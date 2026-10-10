"""Route integration for POST /v1/audio/transcriptions (Phase 24) with the
agent stubbed via respx.

Mirrors the speech tests: a connected, running ``asr`` instance is seeded so
the real scheduler admits against it, and the agent's
``/v1/audio/transcriptions`` is mocked. Asserts the admin-side ASR contract
(ARCHITECTURE.md §7 Audio specifics):

- the multipart upload + text fields are relayed to the agent;
- the upstream response (json / text / verbose_json) is passed through
  untouched;
- an ``AudioUsageSample`` persists (chars = transcript length; audio_seconds
  from a ``verbose_json`` ``duration`` when present, else null);
- 404 unknown / non-asr alias; 400 missing model / no file; 413 oversize;
  503 no provider; non-2xx relay; slot released on every path.
"""

import uuid

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.models import AudioUsageSample, ProviderInstance, ResponseRecord
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


def seed_asr_instance(
    session: Session,
    *,
    alias: str,
    machine_uid: str,
    modality: str = "asr",
    enabled: bool = True,
    websocket_connected: bool = True,
) -> ProviderInstance:
    machine = get_or_create_machine(session, uid=machine_uid, host="127.0.0.1")
    agent = make_agent(session, machine, connected=websocket_connected)
    definition = make_definition(
        session,
        alias=alias,
        provider_type="mock",
        modality=modality,
        enabled=enabled,
        backend_config={"model": {"file": "m.gguf"}},
    )
    return make_instance(session, agent, definition, backend_status="running")


# ---------------------------------------------------------------------------
# Happy path (json)
# ---------------------------------------------------------------------------
@respx.mock
def test_transcription_relays_multipart_and_persists(
    client: TestClient, session: Session
) -> None:
    instance = seed_asr_instance(session, alias="asr-a", machine_uid="asr-a-m")
    route = respx.post(f"{AGENT}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(
            200,
            json={"text": "hello there"},
            headers={"content-type": "application/json"},
        )
    )

    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr-a", "language": "en"},
        files={"file": ("a.wav", b"RIFF....fake", "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "hello there"}

    # The agent received the multipart relay (model + language fields + file).
    req = route.calls.last.request
    body = req.content
    assert b'name="model"' in body and b"asr-a" in body
    assert b'name="language"' in body and b"en" in body
    assert b'name="file"' in body

    sample = (
        session.query(AudioUsageSample)
        .filter(AudioUsageSample.provider_instance_id == instance.id)
        .first()
    )
    assert sample is not None
    assert sample.modality == "asr"
    assert sample.chars == len("hello there")
    assert sample.audio_seconds is None  # plain json has no duration
    assert session.query(ResponseRecord).count() == 0
    assert client.app.state.scheduler.active_count("asr-a") == 0


# ---------------------------------------------------------------------------
# verbose_json duration
# ---------------------------------------------------------------------------
@respx.mock
def test_transcription_verbose_json_duration(
    client: TestClient, session: Session
) -> None:
    seed_asr_instance(session, alias="asr-vj", machine_uid="asr-vj-m")
    respx.post(f"{AGENT}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(
            200,
            json={"text": "hi", "duration": 3.5, "segments": []},
            headers={"content-type": "application/json"},
        )
    )
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr-vj", "response_format": "verbose_json"},
        files={"file": ("a.wav", b"x", "audio/wav")},
    )
    assert resp.status_code == 200
    sample = session.query(AudioUsageSample).first()
    assert sample.chars == 2
    assert sample.audio_seconds == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# text passthrough
# ---------------------------------------------------------------------------
@respx.mock
def test_transcription_text_passthrough(client: TestClient, session: Session) -> None:
    seed_asr_instance(session, alias="asr-tx", machine_uid="asr-tx-m")
    respx.post(f"{AGENT}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(
            200, content=b"plain transcript", headers={"content-type": "text/plain"}
        )
    )
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr-tx", "response_format": "text"},
        files={"file": ("a.wav", b"x", "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.content == b"plain transcript"
    sample = session.query(AudioUsageSample).first()
    assert sample.chars == len("plain transcript")
    assert sample.audio_seconds is None


# ---------------------------------------------------------------------------
# file_path (no upload) is accepted
# ---------------------------------------------------------------------------
@respx.mock
def test_transcription_file_path_only(client: TestClient, session: Session) -> None:
    seed_asr_instance(session, alias="asr-fp", machine_uid="asr-fp-m")
    respx.post(f"{AGENT}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "ok"})
    )
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr-fp", "file_path": "/data/a.wav"},
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Non-2xx relay + release
# ---------------------------------------------------------------------------
@respx.mock
def test_transcription_upstream_429_relays_and_releases(
    client: TestClient, session: Session
) -> None:
    seed_asr_instance(session, alias="asr-429", machine_uid="asr-429-m")
    respx.post(f"{AGENT}/v1/audio/transcriptions").mock(
        return_value=httpx.Response(429, json={"detail": "busy"})
    )
    scheduler = client.app.state.scheduler
    released: list[str] = []
    real_release = scheduler.release

    async def spy_release(alias: str, request_id: str):
        released.append(alias)
        return await real_release(alias, request_id)

    scheduler.release = spy_release  # type: ignore[method-assign]

    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr-429"},
        files={"file": ("a.wav", b"x", "audio/wav")},
    )
    assert resp.status_code == 429
    assert released == ["asr-429"]
    assert scheduler.active_count("asr-429") == 0
    assert session.query(AudioUsageSample).count() == 0


# ---------------------------------------------------------------------------
# Upstream connect failure -> 502 + release
# ---------------------------------------------------------------------------
@respx.mock
def test_transcription_connect_error_502(client: TestClient, session: Session) -> None:
    seed_asr_instance(session, alias="asr-err", machine_uid="asr-err-m")
    respx.post(f"{AGENT}/v1/audio/transcriptions").mock(
        side_effect=httpx.ConnectError("boom", request=httpx.Request("POST", AGENT))
    )
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr-err"},
        files={"file": ("a.wav", b"x", "audio/wav")},
    )
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_failed"
    assert client.app.state.scheduler.active_count("asr-err") == 0


# ---------------------------------------------------------------------------
# Oversize upload -> 413 (before acquire)
# ---------------------------------------------------------------------------
@respx.mock
def test_transcription_oversize_413(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_asr_instance(session, alias="asr-big", machine_uid="asr-big-m")
    monkeypatch.setattr(settings, "AUDIO_MAX_UPLOAD_BYTES", 8)
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr-big"},
        files={"file": ("a.wav", b"x" * 64, "audio/wav")},
    )
    assert resp.status_code == 413
    # No slot was ever acquired.
    assert client.app.state.scheduler.active_count("asr-big") == 0


# ---------------------------------------------------------------------------
# Validation / gating
# ---------------------------------------------------------------------------
def test_transcription_unknown_alias_404(client: TestClient) -> None:
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "nope"},
        files={"file": ("a.wav", b"x", "audio/wav")},
    )
    assert resp.status_code == 404


def test_transcription_llm_alias_rejected_404(
    client: TestClient, session: Session
) -> None:
    seed_asr_instance(session, alias="asr-llm", machine_uid="asr-llm-m", modality="llm")
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr-llm"},
        files={"file": ("a.wav", b"x", "audio/wav")},
    )
    assert resp.status_code == 404
    assert "not a transcription" in resp.json()["detail"]


def test_transcription_missing_model_400(client: TestClient) -> None:
    resp = client.post(
        "/v1/audio/transcriptions", files={"file": ("a.wav", b"x", "audio/wav")}
    )
    assert resp.status_code == 400


def test_transcription_no_file_400(client: TestClient, session: Session) -> None:
    seed_asr_instance(session, alias="asr-nofile", machine_uid="asr-nofile-m")
    resp = client.post("/v1/audio/transcriptions", data={"model": "asr-nofile"})
    assert resp.status_code == 400


def test_transcription_no_provider_503(client: TestClient, session: Session) -> None:
    seed_asr_instance(
        session,
        alias="asr-np",
        machine_uid="asr-np-m",
        websocket_connected=False,
    )
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr-np"},
        files={"file": ("a.wav", b"x", "audio/wav")},
    )
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# transcript metrics unit
# ---------------------------------------------------------------------------
def test_transcript_metrics_unit() -> None:
    from app.api.v1.audio_transcriptions import _transcript_metrics

    assert _transcript_metrics(200, b'{"text": "abc"}') == (3, None)
    assert _transcript_metrics(200, b'{"text": "ab", "duration": 2.0}') == (2, 2.0)
    assert _transcript_metrics(200, b"plain") == (5, None)
    assert _transcript_metrics(429, b'{"detail":"x"}') == (0, None)


def test_uuid_sanity() -> None:
    assert isinstance(uuid.uuid4(), uuid.UUID)

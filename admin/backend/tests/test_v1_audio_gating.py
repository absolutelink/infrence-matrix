"""Phase 24 modality gating matrix + stub un-stubbing.

Locks the routing key (ARCHITECTURE.md §4/§7): each endpoint family accepts
ONLY its modality and answers a clean 404 for every other kind.

- ``/v1/responses`` + ``/v1/chat/completions`` + ``/v1/responses/compact``:
  ``llm`` only.
- ``/v1/embeddings``: ``embedding`` only.
- ``/v1/audio/speech`` + ``/v1/audio/voices``: ``tts`` only.
- ``/v1/audio/transcriptions``: ``asr`` only.

Also asserts the audio speech/transcriptions endpoints are no longer 501 stubs
(they reach the route and answer 400 on a missing model) while ``translations``
stays 501.
"""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.api.v1.stubs import STUBBED_ENDPOINTS
from tests.helpers import make_definition


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def _seed(session: Session, alias: str, modality: str) -> None:
    make_definition(
        session,
        alias=alias,
        provider_type="mock",
        modality=modality,
        backend_config={"model": {"file": "m.gguf"}},
    )


# ---------------------------------------------------------------------------
# tts / asr rejected by the LLM + embedding endpoints
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("modality", ["tts", "asr"])
def test_audio_alias_rejected_by_chat(
    client: TestClient, session: Session, modality: str
) -> None:
    _seed(session, f"gate-{modality}-chat", modality)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": f"gate-{modality}-chat",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert resp.status_code == 404
    assert "not a chat model" in resp.json()["detail"]


@pytest.mark.parametrize("modality", ["tts", "asr"])
def test_audio_alias_rejected_by_responses(
    client: TestClient, session: Session, modality: str
) -> None:
    _seed(session, f"gate-{modality}-resp", modality)
    resp = client.post(
        "/v1/responses", json={"model": f"gate-{modality}-resp", "input": "x"}
    )
    assert resp.status_code == 404
    assert "not a chat model" in resp.json()["detail"]


@pytest.mark.parametrize("modality", ["tts", "asr"])
def test_audio_alias_rejected_by_compact(
    client: TestClient, session: Session, modality: str
) -> None:
    _seed(session, f"gate-{modality}-compact", modality)
    resp = client.post(
        "/v1/responses/compact",
        json={"model": f"gate-{modality}-compact", "input": "x"},
    )
    assert resp.status_code == 404


@pytest.mark.parametrize("modality", ["tts", "asr"])
def test_audio_alias_rejected_by_embeddings(
    client: TestClient, session: Session, modality: str
) -> None:
    _seed(session, f"gate-{modality}-emb", modality)
    resp = client.post(
        "/v1/embeddings", json={"model": f"gate-{modality}-emb", "input": "x"}
    )
    assert resp.status_code == 404
    assert "not an embedding model" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# llm / embedding rejected by the audio endpoints
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("modality", ["llm", "embedding"])
def test_non_tts_rejected_by_speech(
    client: TestClient, session: Session, modality: str
) -> None:
    _seed(session, f"gate-speech-{modality}", modality)
    resp = client.post(
        "/v1/audio/speech", json={"model": f"gate-speech-{modality}", "input": "x"}
    )
    assert resp.status_code == 404
    assert "not a speech" in resp.json()["detail"]


@pytest.mark.parametrize("modality", ["llm", "embedding"])
def test_non_asr_rejected_by_transcriptions(
    client: TestClient, session: Session, modality: str
) -> None:
    _seed(session, f"gate-asr-{modality}", modality)
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": f"gate-asr-{modality}"},
        files={"file": ("a.wav", b"x", "audio/wav")},
    )
    assert resp.status_code == 404
    assert "not a transcription" in resp.json()["detail"]


@pytest.mark.parametrize("modality", ["llm", "embedding", "asr"])
def test_non_tts_rejected_by_voices(
    client: TestClient, session: Session, modality: str
) -> None:
    _seed(session, f"gate-voices-{modality}", modality)
    resp = client.get("/v1/audio/voices", params={"model": f"gate-voices-{modality}"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# cross audio gating (tts != asr)
# ---------------------------------------------------------------------------
def test_tts_rejected_by_transcriptions(client: TestClient, session: Session) -> None:
    _seed(session, "gate-tts-only", "tts")
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "gate-tts-only"},
        files={"file": ("a.wav", b"x", "audio/wav")},
    )
    assert resp.status_code == 404


def test_asr_rejected_by_speech(client: TestClient, session: Session) -> None:
    _seed(session, "gate-asr-only", "asr")
    resp = client.post(
        "/v1/audio/speech", json={"model": "gate-asr-only", "input": "x"}
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# stubs: speech + transcriptions un-stubbed; translations still 501
# ---------------------------------------------------------------------------
def test_speech_transcriptions_no_longer_stubbed() -> None:
    assert ("POST", "/v1/audio/speech") not in STUBBED_ENDPOINTS
    assert ("POST", "/v1/audio/transcriptions") not in STUBBED_ENDPOINTS
    assert ("POST", "/v1/audio/translations") in STUBBED_ENDPOINTS


def test_speech_reaches_route_not_stub(client: TestClient) -> None:
    # Missing model -> 400 from the real route (a 501 stub would answer 501).
    resp = client.post("/v1/audio/speech", json={})
    assert resp.status_code == 400


def test_transcriptions_reaches_route_not_stub(client: TestClient) -> None:
    resp = client.post("/v1/audio/transcriptions", data={})
    assert resp.status_code == 400


def test_translations_still_501(client: TestClient) -> None:
    resp = client.post("/v1/audio/translations", json={})
    assert resp.status_code == 501
    assert resp.json()["error"]["code"] == "endpoint_not_supported"

"""Phase 25 v1 endpoint gates via per-model served modality.

Every /v1 endpoint resolves ``request.model`` through ``resolve_served`` and
gates on the resolved **spec's** modality (not the definition's). A served name
with the right modality passes the gate; a wrong-modality or disabled served
name 404s — whether it came from a single- or multi-model definition.

The gate is exercised without a connected provider (a passing gate then surfaces
503 NoProviderAvailable), plus one full end-to-end (responses) proving a served
name reaches its owning instance and is forwarded as the request model.
"""

from typing import Any

import litellm
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.db import engine
from app.models import (
    ProviderDefinition,
    ProviderInstance,
    ProviderModel,
)
from tests.helpers import get_or_create_machine, make_agent, make_definition


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def _mm_def(
    session: Session,
    *,
    alias: str,
    specs: list[dict[str, Any]],
    connected: bool = False,
    backend_status: str = "stopped",
) -> ProviderDefinition:
    """A definition with ProviderModel rows (multi-model shape). ``connected``
    seeds one backend on a connected agent (for the end-to-end path)."""
    d = make_definition(
        session,
        alias=alias,
        provider_type="mock",
        modality=specs[0]["modality"],
        backend_config={"engine": "shared"},
    )
    for spec in specs:
        session.add(
            ProviderModel(
                definition_id=d.id,
                name=spec["name"],
                modality=spec["modality"],
                backend_config=spec.get("backend_config", {}),
                enabled=spec.get("enabled", True),
            )
        )
    session.commit()
    if connected:
        machine = get_or_create_machine(session, uid=f"{alias}-m", host="127.0.0.1")
        agent = make_agent(session, machine, connected=True)
        session.add(
            ProviderInstance(
                agent_id=agent.id,
                provider_definition_id=d.id,
                backend_status=backend_status,
            )
        )
        session.commit()
    return d


# A multi-model definition exposing one name per modality + a disabled llm.
SPECS = [
    {"name": "mm-llm", "modality": "llm"},
    {"name": "mm-emb", "modality": "embedding"},
    {"name": "mm-tts", "modality": "tts"},
    {"name": "mm-asr", "modality": "asr"},
    {"name": "mm-llm-off", "modality": "llm", "enabled": False},
]


# ---------------------------------------------------------------------------
# responses (llm)
# ---------------------------------------------------------------------------
def test_responses_accepts_llm_served_name(
    client: TestClient, session: Session
) -> None:
    _mm_def(session, alias="mm-a", specs=SPECS)
    # Gate passes (llm served name); no connected provider -> 503 (not 404).
    resp = client.post("/v1/responses", json={"model": "mm-llm", "input": "x"})
    assert resp.status_code == 503


def test_responses_rejects_wrong_modality(client: TestClient, session: Session) -> None:
    _mm_def(session, alias="mm-a", specs=SPECS)
    resp = client.post("/v1/responses", json={"model": "mm-tts", "input": "x"})
    assert resp.status_code == 404
    assert "not a chat model" in resp.json()["detail"]


def test_responses_rejects_disabled_served_name(
    client: TestClient, session: Session
) -> None:
    _mm_def(session, alias="mm-a", specs=SPECS)
    resp = client.post("/v1/responses", json={"model": "mm-llm-off", "input": "x"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# chat completions (llm)
# ---------------------------------------------------------------------------
def test_chat_accepts_llm_served_name(client: TestClient, session: Session) -> None:
    _mm_def(session, alias="mm-a", specs=SPECS)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "mm-llm", "messages": [{"role": "user", "content": "x"}]},
    )
    assert resp.status_code == 503


def test_chat_rejects_embedding_served_name(
    client: TestClient, session: Session
) -> None:
    _mm_def(session, alias="mm-a", specs=SPECS)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "mm-emb", "messages": [{"role": "user", "content": "x"}]},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# embeddings (embedding)
# ---------------------------------------------------------------------------
def test_embeddings_accepts_embedding_served_name(
    client: TestClient, session: Session
) -> None:
    _mm_def(session, alias="mm-a", specs=SPECS)
    resp = client.post("/v1/embeddings", json={"model": "mm-emb", "input": "x"})
    assert resp.status_code == 503


def test_embeddings_rejects_llm_served_name(
    client: TestClient, session: Session
) -> None:
    _mm_def(session, alias="mm-a", specs=SPECS)
    resp = client.post("/v1/embeddings", json={"model": "mm-llm", "input": "x"})
    assert resp.status_code == 404
    assert "not an embedding model" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# audio speech (tts)
# ---------------------------------------------------------------------------
def test_speech_accepts_tts_served_name(client: TestClient, session: Session) -> None:
    _mm_def(session, alias="mm-a", specs=SPECS)
    resp = client.post("/v1/audio/speech", json={"model": "mm-tts", "input": "x"})
    assert resp.status_code == 503


def test_speech_rejects_asr_served_name(client: TestClient, session: Session) -> None:
    _mm_def(session, alias="mm-a", specs=SPECS)
    resp = client.post("/v1/audio/speech", json={"model": "mm-asr", "input": "x"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# audio transcriptions (asr)
# ---------------------------------------------------------------------------
def test_transcriptions_accepts_asr_served_name(
    client: TestClient, session: Session
) -> None:
    _mm_def(session, alias="mm-a", specs=SPECS)
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "mm-asr"},
        files={"file": ("a.wav", b"x", "audio/wav")},
    )
    assert resp.status_code == 503


def test_transcriptions_rejects_tts_served_name(
    client: TestClient, session: Session
) -> None:
    _mm_def(session, alias="mm-a", specs=SPECS)
    resp = client.post(
        "/v1/audio/transcriptions",
        data={"model": "mm-tts"},
        files={"file": ("a.wav", b"x", "audio/wav")},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# voices (tts)
# ---------------------------------------------------------------------------
def test_voices_accepts_tts_served_name(client: TestClient, session: Session) -> None:
    _mm_def(session, alias="mm-a", specs=SPECS)
    resp = client.get("/v1/audio/voices", params={"model": "mm-tts"})
    # Gate passes; no connected agent -> 503 (not 404).
    assert resp.status_code == 503


def test_voices_rejects_asr_served_name(client: TestClient, session: Session) -> None:
    _mm_def(session, alias="mm-a", specs=SPECS)
    resp = client.get("/v1/audio/voices", params={"model": "mm-asr"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /v1/models expansion
# ---------------------------------------------------------------------------
def test_models_lists_enabled_served_names_with_modality(
    client: TestClient, session: Session
) -> None:
    _mm_def(session, alias="mm-a", specs=SPECS)
    make_definition(session, alias="solo", modality="llm")
    data = client.get("/v1/models").json()["data"]
    by_id = {m["id"]: m for m in data}
    # Every ENABLED served name + the single-model alias appear.
    assert {"mm-llm", "mm-emb", "mm-tts", "mm-asr", "solo"} <= set(by_id)
    # The disabled served name is absent.
    assert "mm-llm-off" not in by_id
    # Per-model modality marker is the served spec's modality.
    assert by_id["mm-tts"]["modality"] == "tts"
    assert by_id["mm-emb"]["modality"] == "embedding"
    assert by_id["solo"]["modality"] == "llm"
    # owned_by = the owning definition's provider type.
    assert by_id["mm-tts"]["owned_by"] == "mock"


# ---------------------------------------------------------------------------
# end-to-end: a served name reaches its owning instance + is forwarded
# ---------------------------------------------------------------------------
def test_responses_end_to_end_served_name_reaches_instance(
    client: TestClient, session: Session, monkeypatch
) -> None:
    _mm_def(
        session,
        alias="mm-e2e",
        specs=[{"name": "mm-e2e-llm", "modality": "llm"}],
        connected=True,
        backend_status="running",
    )
    captured: dict[str, Any] = {}

    async def fake_aresponses(*args, **kwargs):  # noqa: ARG001
        captured.update(kwargs)

        async def gen():
            yield {
                "type": "response.completed",
                "response": {
                    "id": "resp_upstream",
                    "object": "response",
                    "status": "completed",
                    "output": [
                        {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hi"}],
                        }
                    ],
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            }

        return gen()

    monkeypatch.setattr(litellm, "aresponses", fake_aresponses)
    resp = client.post(
        "/v1/responses",
        json={"model": "mm-e2e-llm", "input": "x", "stream": True},
    )
    assert resp.status_code == 200
    # The served name is what litellm is asked to route (per-name).
    assert captured["model"] == "mm-e2e-llm"
    # The base_url targets the owning instance's agent port.
    assert captured["api_base"].startswith("http://127.0.0.1:8081")
    frames = [ln for ln in resp.text.splitlines() if ln.startswith("data: ")]
    assert any('"response.completed"' in ln for ln in frames)


# ---------------------------------------------------------------------------
# responses_ws gate (via the resolve helper)
# ---------------------------------------------------------------------------
def test_responses_ws_gate(session: Session) -> None:
    from app.api.v1.responses_ws import _resolve_alias

    _mm_def(session, alias="mm-ws", specs=SPECS)
    with Session(engine) as s:
        ok = _resolve_alias(s, "mm-llm")
        assert not isinstance(ok, dict)  # (definition_id, provider_type)
        bad = _resolve_alias(s, "mm-tts")
        assert isinstance(bad, dict) and bad["error"]["code"] == "model_not_found"
        off = _resolve_alias(s, "mm-llm-off")
        assert isinstance(off, dict)

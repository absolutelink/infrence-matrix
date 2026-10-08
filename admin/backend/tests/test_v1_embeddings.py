"""Route integration for POST /v1/embeddings with litellm monkeypatched.

Mirrors ``test_v1_chat_completions.py``: a connected provider instance is
seeded directly in the UTF8 test DB so the real ``InferenceScheduler``
admits against it, and ``litellm.aembedding`` is replaced with a fake
returning a spec ``EmbeddingResponse``. Asserts the admin-side embeddings
contract (ARCHITECTURE.md §7 Phase 18):

- 200 with the spec body, admin-owned ``model`` (never the backend path);
- a ``TokenUsageSample`` persisted (prompt tokens) and NO ``ResponseRecord``;
- 404 unknown alias / 404 when the alias is an ``llm`` definition /
  400 missing model / 400 missing-or-empty input / 503 no provider;
- 502 JSON (OpenAI error envelope) when ``aembedding`` raises, slot released;
- the alias is registered with litellm ``mode="embedding"``;
- scheduler acquire/release called exactly once per request.
"""

import uuid
from typing import Any

import litellm
import pytest
from fastapi.testclient import TestClient
from litellm.types.utils import EmbeddingResponse, Usage
from sqlmodel import Session

from app.models import (
    ProviderInstance,
    ResponseRecord,
    TokenUsageSample,
)
from app.services import alias_registry
from tests.helpers import (
    get_or_create_machine,
    make_agent,
    make_definition,
    make_instance,
)

BACKEND_PATH = "/models/bge-m3.gguf"


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def seed_embedding_instance(
    session: Session,
    *,
    alias: str,
    machine_uid: str,
    modality: str = "embedding",
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
    return make_instance(
        session, agent, definition, backend_status="running"
    )


def fake_embedding_response(n: int = 1) -> EmbeddingResponse:
    return EmbeddingResponse(
        data=[
            {"object": "embedding", "index": i, "embedding": [0.1 * i, 0.2, 0.3]}
            for i in range(n)
        ],
        model=BACKEND_PATH,
        usage=Usage(prompt_tokens=5, total_tokens=5),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
def test_happy_path_persists_usage_only(
    client: TestClient, session: Session, monkeypatch
) -> None:
    instance = seed_embedding_instance(session, alias="emb-a", machine_uid="emb-m")
    captured: dict[str, Any] = {}

    async def fake_aembedding(*args, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        return fake_embedding_response()

    monkeypatch.setattr(litellm, "aembedding", fake_aembedding)

    resp = client.post("/v1/embeddings", json={"model": "emb-a", "input": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["embedding"]
    # Admin owns the model name — the backend GGUF path never leaks.
    assert body["model"] == "emb-a"
    assert body["model"] != BACKEND_PATH

    # litellm called against the admitted provider port + openai embedding path.
    assert captured["api_base"] == "http://127.0.0.1:8081/v1"
    assert captured["custom_llm_provider"] == "openai"
    assert captured["api_key"] == "unused"
    assert captured["input"] == "hello"

    # TokenUsageSample written with prompt tokens...
    sample = (
        session.query(TokenUsageSample)
        .filter(TokenUsageSample.provider_instance_id == instance.id)
        .first()
    )
    assert sample is not None
    assert sample.prompt_tokens == 5
    assert sample.completion_tokens == 0
    assert sample.provider_definition_id == instance.provider_definition_id
    # ...and NO ResponseRecord (embeddings are not conversation turns).
    assert session.query(ResponseRecord).count() == 0

    scheduler = client.app.state.scheduler
    assert scheduler.active_count("emb-a") == 0


def test_happy_path_list_input_and_dimensions(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_embedding_instance(session, alias="emb-list", machine_uid="emb-list-m")
    captured: dict[str, Any] = {}

    async def fake_aembedding(*args, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        return fake_embedding_response(n=2)

    monkeypatch.setattr(litellm, "aembedding", fake_aembedding)

    resp = client.post(
        "/v1/embeddings",
        json={
            "model": "emb-list",
            "input": ["a", "b"],
            "dimensions": 3,
            "user": "u-1",
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 2
    assert captured["input"] == ["a", "b"]
    assert captured["dimensions"] == 3
    assert captured["user"] == "u-1"


# ---------------------------------------------------------------------------
# litellm registration mode
# ---------------------------------------------------------------------------
def test_registered_with_embedding_mode(
    client: TestClient, session: Session, monkeypatch
) -> None:
    alias = f"emb-reg-{uuid.uuid4().hex[:8]}"
    seed_embedding_instance(session, alias=alias, machine_uid="emb-reg-m2")

    calls: list[tuple[str, str]] = []
    real = alias_registry.ensure_registered

    def spy(a: str, mode: str = "chat") -> None:
        calls.append((a, mode))
        real(a, mode=mode)

    monkeypatch.setattr("app.api.v1.embeddings.ensure_registered", spy)

    async def fake_aembedding(*args, **kwargs):  # noqa: ARG001
        return fake_embedding_response()

    monkeypatch.setattr(litellm, "aembedding", fake_aembedding)

    resp = client.post("/v1/embeddings", json={"model": alias, "input": "x"})
    assert resp.status_code == 200
    assert (alias, "embedding") in calls
    # The litellm model table reflects the embedding mode (no streaming flags).
    info = litellm.model_cost[alias]
    assert info["mode"] == "embedding"
    assert info["litellm_provider"] == "openai"
    assert "supports_native_streaming" not in info


def test_modality_change_reregisters_alias() -> None:
    """Discriminating test for the alias cache: a definition's modality can be
    PATCHed (llm<->embedding) while no backends are attached, so the process
    cache must track the registered mode and re-register on a switch — NOT
    early-return on alias alone. If the cache were a plain ``set[str]`` the
    second call would be a no-op and the litellm mode would stay stale."""
    alias = f"reg-switch-{uuid.uuid4().hex[:8]}"

    # Warm as chat (llm modality): native streaming declared.
    alias_registry.ensure_registered(alias)
    assert litellm.model_cost[alias]["mode"] == "chat"
    assert litellm.model_cost[alias]["supports_native_streaming"] is True

    # Switch to embedding: the stale chat entry must be overwritten. (litellm
    # merges into the existing model_cost entry, so supports_native_streaming
    # lingers from the chat registration — harmless, aembedding keys off mode.
    # The discriminating signal is the mode flip: a set-based cache would
    # early-return here and leave mode == "chat".)
    alias_registry.ensure_registered(alias, mode="embedding")
    assert litellm.model_cost[alias]["mode"] == "embedding"

    # Switch back to chat: native streaming restored (guards the fake-stream
    # APIError a stale embedding entry would otherwise cause on first use).
    alias_registry.ensure_registered(alias)
    assert litellm.model_cost[alias]["mode"] == "chat"
    assert litellm.model_cost[alias]["supports_native_streaming"] is True


# ---------------------------------------------------------------------------
# Validation / admission errors
# ---------------------------------------------------------------------------
def test_unknown_alias_404(client: TestClient) -> None:
    resp = client.post("/v1/embeddings", json={"model": "nope", "input": "x"})
    assert resp.status_code == 404


def test_llm_alias_rejected_404(client: TestClient, session: Session) -> None:
    seed_embedding_instance(
        session, alias="emb-llm", machine_uid="emb-llm-m", modality="llm"
    )
    resp = client.post("/v1/embeddings", json={"model": "emb-llm", "input": "x"})
    assert resp.status_code == 404
    assert "not an embedding model" in resp.json()["detail"]


def test_disabled_embedding_alias_404(client: TestClient, session: Session) -> None:
    seed_embedding_instance(
        session, alias="emb-off", machine_uid="emb-off-m", enabled=False
    )
    resp = client.post("/v1/embeddings", json={"model": "emb-off", "input": "x"})
    assert resp.status_code == 404
    assert "disabled" in resp.json()["detail"]


def test_missing_model_400(client: TestClient) -> None:
    resp = client.post("/v1/embeddings", json={"input": "x"})
    assert resp.status_code == 400


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "emb-v"},  # no input
        {"model": "emb-v", "input": ""},  # empty string
        {"model": "emb-v", "input": []},  # empty list
        {"model": "emb-v", "input": 5},  # wrong type
    ],
)
def test_bad_input_400(client: TestClient, payload: dict) -> None:
    # Input validation happens before definition resolution, so no seed needed.
    resp = client.post("/v1/embeddings", json=payload)
    assert resp.status_code == 400


def test_no_connected_provider_503(client: TestClient, session: Session) -> None:
    seed_embedding_instance(
        session,
        alias="emb-np",
        machine_uid="emb-np-m",
        websocket_connected=False,
    )
    resp = client.post("/v1/embeddings", json={"model": "emb-np", "input": "x"})
    assert resp.status_code == 503


def test_queue_cleared_503(client: TestClient, session: Session, monkeypatch) -> None:
    """QueueCleared on the embeddings path -> 503 JSON."""
    from app.services.scheduler import QueueCleared

    seed_embedding_instance(session, alias="emb-qc", machine_uid="emb-qc-m")
    scheduler = client.app.state.scheduler

    async def cleared_acquire(alias, request_id):  # noqa: ARG001
        raise QueueCleared("the 'emb-qc' queue was cleared by an operator")

    monkeypatch.setattr(scheduler, "acquire", cleared_acquire)
    resp = client.post("/v1/embeddings", json={"model": "emb-qc", "input": "x"})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Upstream failure -> 502 JSON + slot released
# ---------------------------------------------------------------------------
def test_upstream_failure_502_and_release(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_embedding_instance(session, alias="emb-fail", machine_uid="emb-fail-m")

    async def raising_aembedding(*args, **kwargs):  # noqa: ARG001
        raise litellm.exceptions.APIError(
            status_code=500,
            message="Error code: 500 - embedding backend down",
            llm_provider="openai",
            model="emb-fail",
        )

    monkeypatch.setattr(litellm, "aembedding", raising_aembedding)

    scheduler = client.app.state.scheduler
    acquire_calls: list[tuple[str, str]] = []
    release_calls: list[tuple[str, str]] = []
    real_acquire, real_release = scheduler.acquire, scheduler.release

    async def spy_acquire(alias: str, request_id: str):
        acquire_calls.append((alias, request_id))
        return await real_acquire(alias, request_id)

    async def spy_release(alias: str, request_id: str):
        release_calls.append((alias, request_id))
        return await real_release(alias, request_id)

    monkeypatch.setattr(scheduler, "acquire", spy_acquire)
    monkeypatch.setattr(scheduler, "release", spy_release)

    resp = client.post("/v1/embeddings", json={"model": "emb-fail", "input": "x"})
    assert resp.status_code == 502
    body = resp.json()
    assert body["error"]["type"] == "server_error"
    assert body["error"]["code"] == "upstream_failed"

    assert [a for a, _ in acquire_calls] == ["emb-fail"]
    assert [(a, r) for a, r in release_calls] == acquire_calls
    assert scheduler.active_count("emb-fail") == 0
    # No usage sample on a failed call.
    assert session.query(TokenUsageSample).count() == 0


# ---------------------------------------------------------------------------
# Scheduler interaction: acquire/release exactly once on success
# ---------------------------------------------------------------------------
def test_scheduler_acquire_release_once(
    client: TestClient, session: Session, monkeypatch
) -> None:
    seed_embedding_instance(session, alias="emb-cnt", machine_uid="emb-cnt-m")

    async def fake_aembedding(*args, **kwargs):  # noqa: ARG001
        return fake_embedding_response()

    monkeypatch.setattr(litellm, "aembedding", fake_aembedding)

    scheduler = client.app.state.scheduler
    acquire_calls: list[tuple[str, str]] = []
    release_calls: list[tuple[str, str]] = []
    real_acquire, real_release = scheduler.acquire, scheduler.release

    async def spy_acquire(alias: str, request_id: str):
        acquire_calls.append((alias, request_id))
        return await real_acquire(alias, request_id)

    async def spy_release(alias: str, request_id: str):
        release_calls.append((alias, request_id))
        return await real_release(alias, request_id)

    monkeypatch.setattr(scheduler, "acquire", spy_acquire)
    monkeypatch.setattr(scheduler, "release", spy_release)

    resp = client.post("/v1/embeddings", json={"model": "emb-cnt", "input": "x"})
    assert resp.status_code == 200
    assert [a for a, _ in acquire_calls] == ["emb-cnt"]
    assert [(a, r) for a, r in release_calls] == acquire_calls
    assert scheduler.active_count("emb-cnt") == 0


# ---------------------------------------------------------------------------
# Cross-gating: chat endpoints reject an embedding-modality alias (404)
# ---------------------------------------------------------------------------
def test_chat_completions_rejects_embedding_alias(
    client: TestClient, session: Session
) -> None:
    seed_embedding_instance(
        session, alias="gate-emb", machine_uid="gate-emb-m", modality="embedding"
    )
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gate-emb", "messages": [{"role": "user", "content": "x"}]},
    )
    assert resp.status_code == 404
    assert "not a chat model" in resp.json()["detail"]


def test_responses_rejects_embedding_alias(client: TestClient, session: Session) -> None:
    seed_embedding_instance(
        session, alias="gate-emb2", machine_uid="gate-emb2-m", modality="embedding"
    )
    resp = client.post(
        "/v1/responses", json={"model": "gate-emb2", "input": "x"}
    )
    assert resp.status_code == 404
    assert "not a chat model" in resp.json()["detail"]

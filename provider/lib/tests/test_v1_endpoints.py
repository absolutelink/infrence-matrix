"""/v1 translation layer endpoint tests (fake driver over ASGITransport)."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from provider_lib.app_factory import (
    BackendOverrides,
    _aggregate_chat_chunk,
    create_provider_app,
)
from provider_lib.backend import BackendDriver, BackendLifecycle
from provider_lib.config import ProviderSettings
from provider_lib.registry import BackendHandle, BackendRegistry


class EndpointDriver(BackendDriver):
    def __init__(self, *, chat: bool = True) -> None:
        self.chat = chat

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def health(self) -> bool:
        return True

    async def list_models(self) -> list[dict[str, Any]]:
        return [{"id": "ep-model", "object": "model"}]

    def stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        async def gen() -> AsyncIterator[dict[str, Any]]:
            yield {"type": "response.created", "response": {"id": "r1"}}
            yield {"type": "response.in_progress", "response": {"id": "r1"}}
            yield {"type": "response.output_text.delta", "delta": "hello "}
            yield {"type": "response.output_text.delta", "delta": "world"}
            yield {
                "type": "response.completed",
                "response": {
                    "id": "r1",
                    "status": "completed",
                    "usage": {"total_tokens": 7},
                },
            }

        return gen()

    def stream_chat_completions(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        if not self.chat:
            return super().stream_chat_completions(request)

        async def gen() -> AsyncIterator[dict[str, Any]]:
            yield {
                "id": "c1",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": "hi"}}],
            }
            yield {
                "id": "c1",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

        return gen()


EMBEDDING_RESPONSE: dict[str, Any] = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    "model": "m",
    "usage": {"prompt_tokens": 3, "total_tokens": 3},
}


class EmbeddingsDriver(EndpointDriver):
    """EndpointDriver that also implements the embeddings surface."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_calls = 0

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        self.embed_calls += 1
        return dict(EMBEDDING_RESPONSE)


class FailingEmbeddingsDriver(EndpointDriver):
    """Embeddings-capable driver whose embeddings() raises a generic error."""

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")


def _settings() -> ProviderSettings:
    return ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="m-v1",
        MACHINE_SECRET="tok",
        AGENT_ID="m-v1-agent",
        ADMIN_BASE_URL="http://admin:8000",
        CACHE_DIR="/tmp/prov_test_cache",
        MODELS_DIR="/tmp/prov_test_models",
    )


async def _make(
    driver: EndpointDriver, *, started: bool = True, capacity: int = 1
) -> tuple[BackendLifecycle, httpx.AsyncClient]:
    lifecycle = BackendLifecycle(driver, capacity=capacity)
    if started:
        await lifecycle.start()
    app = create_provider_app(
        _settings(),
        BackendOverrides(provider_type="test", version="v", lifecycle=lifecycle),
    )
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://provider"
    )
    return lifecycle, client


def _events(body: str) -> list[dict[str, Any]]:
    events = []
    for block in body.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[len("data: ") :]))
    return events


async def test_models_ok() -> None:
    _, client = await _make(EndpointDriver())
    async with client:
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert body["data"][0]["id"] == "ep-model"


async def test_models_503_when_stopped() -> None:
    _, client = await _make(EndpointDriver(), started=False)
    async with client:
        assert (await client.get("/v1/models")).status_code == 503


async def test_responses_streams_sse() -> None:
    lifecycle, client = await _make(EndpointDriver())
    async with client:
        resp = await client.post("/v1/responses", json={"input": "hi", "stream": True})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        types = [e["type"] for e in _events(resp.text)]
        assert types == [
            "response.created",
            "response.in_progress",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.completed",
        ]
        event_lines = [ln for ln in resp.text.splitlines() if ln.startswith("event: ")]
        assert event_lines == [f"event: {t}" for t in types]
        assert lifecycle.in_flight == 0


async def test_responses_non_stream_returns_json() -> None:
    """Spec default (stream absent/false) returns the terminal response as JSON."""
    lifecycle, client = await _make(EndpointDriver())
    async with client:
        resp = await client.post("/v1/responses", json={"input": "hi"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        assert body["status"] == "completed"
        assert body["usage"]["total_tokens"] > 0
        assert lifecycle.in_flight == 0


async def test_responses_429_at_capacity() -> None:
    lifecycle, client = await _make(EndpointDriver(), capacity=1)
    async with client:
        await lifecycle.acquire_slot()  # occupy the only slot
        resp = await client.post("/v1/responses", json={})
        assert resp.status_code == 429
        assert lifecycle.in_flight == 1  # the busy request leaked nothing
        await lifecycle.release_slot()


async def test_responses_503_when_not_started() -> None:
    lifecycle, client = await _make(EndpointDriver(), started=False)
    async with client:
        resp = await client.post("/v1/responses", json={})
        assert resp.status_code == 503
        assert lifecycle.in_flight == 0


async def test_chat_completions_streams() -> None:
    lifecycle, client = await _make(EndpointDriver())
    async with client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "data: [DONE]" in resp.text
        chunks = [
            e for e in _events(resp.text) if e.get("object") == "chat.completion.chunk"
        ]
        assert chunks
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert lifecycle.in_flight == 0


async def test_chat_completions_non_stream_aggregates() -> None:
    """Spec default (stream absent/false): the chunk stream is folded into
    a single chat.completion JSON object (admin's litellm non-stream path
    requires this)."""
    lifecycle, client = await _make(EndpointDriver())
    async with client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "hi"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert lifecycle.in_flight == 0


def test_aggregate_chat_chunk_merges_deltas_and_tools() -> None:
    acc: dict[str, Any] | None = None
    acc = _aggregate_chat_chunk(
        acc,
        {
            "id": "c1",
            "created": 5,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "f", "arguments": ""},
                            }
                        ],
                    },
                }
            ],
        },
    )
    acc = _aggregate_chat_chunk(
        acc,
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "hello ",
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"a":'}}
                        ],
                    },
                }
            ]
        },
    )
    acc = _aggregate_chat_chunk(
        acc,
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "world",
                        "tool_calls": [{"index": 0, "function": {"arguments": "1}"}}],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        },
    )
    assert acc["id"] == "c1"
    assert acc["created"] == 5
    assert acc["model"] == "m"
    msg = acc["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert msg["content"] == "hello world"
    assert msg["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "f", "arguments": '{"a":1}'},
        }
    ]
    assert acc["choices"][0]["finish_reason"] == "tool_calls"
    assert acc["usage"]["total_tokens"] == 5


async def test_chat_completions_501_when_unsupported() -> None:
    lifecycle, client = await _make(EndpointDriver(chat=False))
    async with client:
        resp = await client.post("/v1/chat/completions", json={})
        assert resp.status_code == 501
        assert lifecycle.in_flight == 0


async def test_embeddings_ok() -> None:
    driver = EmbeddingsDriver()
    lifecycle, client = await _make(driver)
    async with client:
        resp = await client.post("/v1/embeddings", json={"input": "hi", "model": "m"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json() == EMBEDDING_RESPONSE
        # slot acquired + released exactly once (driver ran once, none held)
        assert driver.embed_calls == 1
        assert lifecycle.in_flight == 0


async def test_embeddings_501_when_unsupported() -> None:
    # EndpointDriver has no embeddings() override -> default ABC raises -> 501
    lifecycle, client = await _make(EndpointDriver())
    async with client:
        resp = await client.post("/v1/embeddings", json={"input": "hi"})
        assert resp.status_code == 501
        assert lifecycle.in_flight == 0


async def test_embeddings_429_at_capacity() -> None:
    lifecycle, client = await _make(EmbeddingsDriver(), capacity=1)
    async with client:
        await lifecycle.acquire_slot()  # occupy the only slot
        resp = await client.post("/v1/embeddings", json={"input": "hi"})
        assert resp.status_code == 429
        assert lifecycle.in_flight == 1  # the busy request leaked nothing
        await lifecycle.release_slot()


async def test_embeddings_503_when_not_started() -> None:
    lifecycle, client = await _make(EmbeddingsDriver(), started=False)
    async with client:
        resp = await client.post("/v1/embeddings", json={"input": "hi"})
        assert resp.status_code == 503
        assert lifecycle.in_flight == 0


async def test_embeddings_500_on_driver_error() -> None:
    # A generic (non-NotImplementedError) driver failure surfaces as 500 and
    # still releases the slot via the lifecycle's `finally`.
    lifecycle = BackendLifecycle(FailingEmbeddingsDriver(), capacity=1)
    await lifecycle.start()
    app = create_provider_app(
        _settings(),
        BackendOverrides(provider_type="test", version="v", lifecycle=lifecycle),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://provider",
    ) as client:
        resp = await client.post("/v1/embeddings", json={"input": "hi"})
        assert resp.status_code == 500
        assert lifecycle.in_flight == 0


async def test_health_reports_backend_state() -> None:
    lifecycle, client = await _make(EndpointDriver())
    async with client:
        body = (await client.get("/health")).json()
        assert body["backend_status"] == "running"
        assert body["in_flight"] == 0
        assert body["capacity"] == 1


async def test_no_lifecycle_wired_returns_503() -> None:
    app = create_provider_app(
        _settings(), BackendOverrides(provider_type="test", version="v")
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://provider"
    ) as client:
        assert (await client.get("/v1/models")).status_code == 503
        assert (await client.post("/v1/responses", json={})).status_code == 503
        assert (await client.post("/v1/chat/completions", json={})).status_code == 503
        assert (await client.post("/v1/embeddings", json={})).status_code == 503


# ---------------------------------------------------------------------------
# Port model overhaul: single /v1 surface multiplexed by request `model`.
# ---------------------------------------------------------------------------


class RoutingDriver(BackendDriver):
    """Fake driver that stamps its ``tag`` into every response so a test can
    tell which hosted backend served a request."""

    def __init__(self, tag: str, *, models: list[str] | None = None) -> None:
        self.tag = tag
        self._models = models if models is not None else [tag]
        self.embed_calls = 0

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def health(self) -> bool:
        return True

    async def list_models(self) -> list[dict[str, Any]]:
        return [{"id": m, "object": "model"} for m in self._models]

    def stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        async def gen() -> AsyncIterator[dict[str, Any]]:
            yield {
                "type": "response.completed",
                "response": {"id": self.tag, "status": "completed", "tag": self.tag},
            }

        return gen()

    def stream_chat_completions(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        async def gen() -> AsyncIterator[dict[str, Any]]:
            yield {
                "id": self.tag,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": self.tag}}],
            }
            yield {
                "id": self.tag,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

        return gen()

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        self.embed_calls += 1
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
            "model": self.tag,
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }


def _registry_app(
    handles: list[BackendHandle],
) -> tuple[BackendRegistry, httpx.AsyncClient]:
    reg = BackendRegistry()
    for h in handles:
        reg.add(h)
    app = create_provider_app(
        _settings(),
        BackendOverrides(provider_type="test", version="v", registry=reg),
    )
    return reg, httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://provider"
    )


async def _running_handle(iid: str, alias: str, tag: str) -> BackendHandle:
    lifecycle = BackendLifecycle(RoutingDriver(tag), instance_id=iid)
    await lifecycle.start()
    return BackendHandle(iid, lifecycle, None, alias=alias)


async def test_registry_routes_responses_by_model() -> None:
    a = await _running_handle("i1", "alpha", "A")
    b = await _running_handle("i2", "beta", "B")
    _reg, client = _registry_app([a, b])
    async with client:
        ra = await client.post("/v1/responses", json={"model": "alpha"})
        rb = await client.post("/v1/responses", json={"model": "beta"})
        assert ra.json()["tag"] == "A"
        assert rb.json()["tag"] == "B"
        # Each backend served exactly one request; slots released.
        assert a.lifecycle.in_flight == 0 and b.lifecycle.in_flight == 0


async def test_registry_unknown_model_is_404() -> None:
    a = await _running_handle("i1", "alpha", "A")
    _reg, client = _registry_app([a])
    async with client:
        assert (
            await client.post("/v1/responses", json={"model": "nope"})
        ).status_code == 404
        # Missing model field is also a 404 (not a fallback to the sole handle).
        assert (await client.post("/v1/responses", json={})).status_code == 404


async def test_registry_routes_chat_and_embeddings_by_model() -> None:
    a = await _running_handle("i1", "alpha", "A")
    b = await _running_handle("i2", "beta", "B")
    _reg, client = _registry_app([a, b])
    async with client:
        chat = await client.post(
            "/v1/chat/completions",
            json={"model": "beta", "messages": [{"role": "user", "content": "x"}]},
        )
        assert chat.status_code == 200
        assert chat.json()["choices"][0]["message"]["content"] == "B"

        emb = await client.post("/v1/embeddings", json={"model": "alpha", "input": "x"})
        assert emb.status_code == 200
        assert emb.json()["model"] == "A"
        # The chat hit beta's driver only; the embedding hit alpha's driver only.
        assert a.lifecycle.driver.embed_calls == 1
        assert b.lifecycle.driver.embed_calls == 0


async def test_registry_models_aggregates_running_backends() -> None:
    a = await _running_handle("i1", "alpha", "A")
    b = await _running_handle("i2", "beta", "B")
    # A stopped backend contributes nothing to the aggregate.
    stopped = BackendLifecycle(RoutingDriver("C", models=["gamma"]), instance_id="i3")
    c = BackendHandle("i3", stopped, None, alias="gamma")
    _reg, client = _registry_app([a, b, c])
    async with client:
        body = (await client.get("/v1/models")).json()
        ids = {m["id"] for m in body["data"]}
        assert ids == {"A", "B"}  # "gamma" excluded (backend stopped)


async def test_registry_health_reports_backends() -> None:
    a = await _running_handle("i1", "alpha", "A")
    _reg, client = _registry_app([a])
    async with client:
        body = (await client.get("/health")).json()
        assert body["status"] == "ok"
        assert body["provider_type"] == "test"
        summary = {b["alias"]: b for b in body["backends"]}
        assert summary["alpha"]["backend_status"] == "running"


async def test_registry_non_object_body_is_400() -> None:
    """L1: a JSON array/string body reaches ``_resolve_lifecycle``; it must be a
    clean 400, not an AttributeError->500 from ``body.get``."""
    a = await _running_handle("i1", "alpha", "A")
    _reg, client = _registry_app([a])
    async with client:
        for bad in ([1, 2], "a string", 123):
            resp = await client.post("/v1/responses", json=bad)
            assert resp.status_code == 400, bad
            assert resp.json()["detail"] == "request body must be a JSON object"
        # Same guard on the chat surface.
        assert (
            await client.post("/v1/chat/completions", json=[1, 2])
        ).status_code == 400


async def test_registry_non_string_model_is_404() -> None:
    """L1: a non-string ``model`` (list/int) degrades to a clean 404 via the
    registry's non-string guard, not a TypeError from ``dict.get``."""
    a = await _running_handle("i1", "alpha", "A")
    _reg, client = _registry_app([a])
    async with client:
        assert (
            await client.post("/v1/responses", json={"model": ["x"]})
        ).status_code == 404
        assert (
            await client.post("/v1/responses", json={"model": 123})
        ).status_code == 404

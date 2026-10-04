"""/v1 translation layer endpoint tests (fake driver over ASGITransport)."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from provider_lib.app_factory import BackendOverrides, create_provider_app
from provider_lib.backend import BackendDriver, BackendLifecycle
from provider_lib.config import ProviderSettings


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


def _settings() -> ProviderSettings:
    return ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="m-v1",
        PROVIDER_REGISTRATION_TOKEN="tok",
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
            json={"messages": [{"role": "user", "content": "hi"}]},
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


async def test_chat_completions_501_when_unsupported() -> None:
    lifecycle, client = await _make(EndpointDriver(chat=False))
    async with client:
        resp = await client.post("/v1/chat/completions", json={})
        assert resp.status_code == 501
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

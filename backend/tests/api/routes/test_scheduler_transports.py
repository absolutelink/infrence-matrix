"""Scheduler lease behavior at inference transport boundaries."""

import asyncio
import json
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.api.routes.v1 import v1_completions
from app.api.routes.v1.responses import router as responses_router
from app.api.routes.v1.responses import ws as responses_ws
from app.api.routes.v1.responses.schemas import CreateResponseBody
from app.services.inference_scheduler import (
    InferenceLeaseHandle,
    InferenceRequestCancelled,
)


class FakeStreamResponse:
    def __init__(self, lines: list[str], events: list[str] | None = None) -> None:
        self.lines = lines
        self.events = events

    async def __aenter__(self) -> FakeStreamResponse:
        return self

    async def __aexit__(self, *args: object) -> None:
        if self.events is not None:
            self.events.append("upstream_closed")

    def raise_for_status(self) -> None:
        pass

    def aiter_lines(self):
        async def iterate():
            for line in self.lines:
                yield line

        return iterate()


class FakeClient:
    def __init__(self, response: FakeStreamResponse) -> None:
        self.response = response
        self.stream_calls: list[dict[str, Any]] = []

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    def stream(self, method: str, url: str, **kwargs: Any) -> FakeStreamResponse:
        self.stream_calls.append({"method": method, "url": url, **kwargs})
        return self.response


class FakeLease:
    def __init__(self, events: list[str], server: object | None = None) -> None:
        self.slot_generation = 23
        self.server = server
        self.lost = asyncio.Event()
        self.events = events
        self.upstream_started = False

    def mark_upstream_started(self) -> None:
        self.upstream_started = True

    async def guard(self, awaitable, *, cancelled=None):
        self.upstream_started = True
        return await awaitable

    async def release(self) -> None:
        self.events.append("released")


def _install_client(monkeypatch: pytest.MonkeyPatch, module: Any, client: FakeClient):
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **_kwargs: client)


@pytest.mark.asyncio
async def test_guard_cancels_upstream_when_client_disconnects() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    operation_cancelled = asyncio.Event()

    async def upstream() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            operation_cancelled.set()

    handle = InferenceLeaseHandle("request-1", SimpleNamespace(), uuid.uuid4())
    guarded = asyncio.create_task(handle.guard(upstream(), cancelled=cancelled))
    await started.wait()
    cancelled.set()

    with pytest.raises(InferenceRequestCancelled):
        await guarded

    assert operation_cancelled.is_set()


def test_responses_reasoning_effort_reaches_upstream_payload() -> None:
    request = CreateResponseBody(
        model="model",
        input="hello",
        reasoning={"effort": "none"},
        stream=True,
    )

    payload = responses_router._llama_payload(
        request, [], stream=True, reasoning_effort="none"
    )

    assert payload["reasoning_effort"] == "none"


def test_responses_payload_omits_reasoning_effort_when_none() -> None:
    request = CreateResponseBody(model="model", input="hello", stream=True)

    payload = responses_router._llama_payload(request, [], stream=True)

    assert "reasoning_effort" not in payload


@pytest.mark.asyncio
async def test_legacy_completion_releases_before_usage_and_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    upstream = FakeStreamResponse(
        [
            'data: {"choices":[{"text":"ok","finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":1,"completion_tokens":1}}',
            "data: [DONE]",
        ],
        events,
    )
    client = FakeClient(upstream)
    lease = FakeLease(events)
    _install_client(monkeypatch, v1_completions, client)
    monkeypatch.setattr(
        v1_completions.agent_manager,
        "get_agent",
        AsyncMock(return_value=SimpleNamespace(host="agent", port=8080)),
    )
    request = v1_completions.CompletionRequest(
        model="model", prompt="hello", stream=True
    )

    stream = v1_completions._stream_completion_via_agent(
        "agent-id", "server-id", request, "cmpl-1", lease
    )
    content = await anext(stream)
    usage = await anext(stream)

    assert '"text":"ok"' in content
    assert '"choices":[]' in usage
    assert events[:2] == ["upstream_closed", "released"]
    assert client.stream_calls[0]["headers"] == {"X-Inference-Slot-Generation": "23"}

    done = await anext(stream)
    assert done == "data: [DONE]\n\n"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.asyncio
async def test_responses_sse_releases_before_terminal_event_and_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    upstream = FakeStreamResponse(
        [
            "data:"
            + json.dumps(
                {
                    "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ),
            "data: [DONE]",
        ]
    )
    client = FakeClient(upstream)
    lease = FakeLease(events)
    _install_client(monkeypatch, responses_router, client)
    monkeypatch.setattr(
        responses_router,
        "ensure_server_ready_by_id",
        AsyncMock(return_value=SimpleNamespace(agent_id="agent-id")),
    )
    monkeypatch.setattr(
        responses_router.agent_manager,
        "get_agent",
        AsyncMock(return_value=SimpleNamespace(host="agent", port=8080)),
    )

    def persist(**_kwargs: Any) -> None:
        assert "released" in events
        events.append("persisted")

    monkeypatch.setattr(responses_router, "_persist_response", persist)
    request = CreateResponseBody(model="model", input="hello", stream=True)
    frames: list[str] = []
    async for frame in responses_router._stream_events(
        request,
        "server-id",
        "agent-id",
        [],
        "resp-1",
        1,
        model_id=uuid.uuid4(),
        instance_agent_id=uuid.uuid4(),
        persist=True,
        lease=lease,
    ):
        if frame.startswith("event: response.completed"):
            assert "released" in events
            events.append("terminal")
        frames.append(frame)

    assert events == ["released", "terminal", "persisted"]
    assert lease.upstream_started
    assert frames[-1] == "data: [DONE]\n\n"
    assert client.stream_calls[0]["headers"] == {"X-Inference-Slot-Generation": "23"}


@pytest.mark.asyncio
async def test_responses_websocket_propagates_disconnect_and_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disconnected = asyncio.Event()
    disconnected.set()
    server = SimpleNamespace(id="server-id", agent_id="agent-id", model_id="model-id")
    model = SimpleNamespace(id="model-id")
    lease = InferenceLeaseHandle("request-id", server, uuid.uuid4(), slot_generation=31)
    lease.release = AsyncMock()
    acquire_callbacks: list[Any] = []

    async def acquire(*_args: Any, **kwargs: Any) -> InferenceLeaseHandle:
        acquire_callbacks.append(kwargs["is_cancelled"])
        assert await kwargs["is_cancelled"]()
        return lease

    class HangingResponse(FakeStreamResponse):
        def aiter_lines(self):
            async def iterate():
                await asyncio.Event().wait()
                yield ""

            return iterate()

    client = FakeClient(HangingResponse([]))
    _install_client(monkeypatch, responses_ws, client)
    monkeypatch.setattr(
        responses_ws,
        "resolve_target",
        AsyncMock(return_value=(server, model, None)),
    )
    monkeypatch.setattr(responses_ws.inference_scheduler, "acquire", acquire)
    monkeypatch.setattr(
        responses_ws.agent_manager,
        "get_agent",
        AsyncMock(return_value=SimpleNamespace(host="agent", port=8080)),
    )
    websocket = SimpleNamespace(send_text=AsyncMock())
    request = CreateResponseBody(model="model", input="hello", store=False)

    with pytest.raises(InferenceRequestCancelled):
        await responses_ws._stream_to_ws(websocket, request, disconnected=disconnected)

    assert len(acquire_callbacks) == 1
    lease.release.assert_awaited_once()
    assert client.stream_calls[0]["headers"] == {"X-Inference-Slot-Generation": "31"}

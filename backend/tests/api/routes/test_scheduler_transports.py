"""Scheduler lease behavior at inference transport boundaries."""

import asyncio
import json
import logging
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from app.api.routes.v1 import v1_completions
from app.api.routes.v1.responses import router as responses_router
from app.api.routes.v1.responses import ws as responses_ws
from app.api.routes.v1.responses.schemas import CreateResponseBody
from app.services import inference_stream
from app.services.inference_scheduler import (
    InferenceLeaseHandle,
    InferenceRequestCancelled,
)


async def _drain_release_tasks() -> None:
    """Let detached background lease releases finish."""
    while inference_stream._release_tasks:
        await asyncio.gather(*list(inference_stream._release_tasks))


class FakeStreamResponse:
    def __init__(self, lines: list[str], events: list[str] | None = None) -> None:
        self.lines = lines
        self.events = events
        self.status_code = 200

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
        # The shared client must never be entered-and-closed per request; only
        # the streamed response context closes (returning its connection to
        # the pool). Any client-level close is a regression.
        self.closed = False

    async def __aenter__(self) -> FakeClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True

    def stream(self, method: str, url: str, **kwargs: Any) -> FakeStreamResponse:
        self.stream_calls.append({"method": method, "url": url, **kwargs})
        return self.response


class FakeLease:
    def __init__(self, events: list[str], server: object | None = None) -> None:
        self.request_id = "fake-request-id"
        self.slot_generation = 23
        self.server = server
        self.lost = asyncio.Event()
        self.events = events
        self.upstream_started = False

    def mark_upstream_started(self) -> None:
        self.upstream_started = True

    def dispatch_headers(self) -> dict[str, str]:
        return {
            "X-Inference-Slot-Generation": str(self.slot_generation),
            "X-Inference-Request-ID": self.request_id,
        }

    async def guard(self, awaitable, *, cancelled=None):
        self.upstream_started = True
        return await awaitable

    async def release(self, outcome: str = "completed") -> None:
        self.events.append("released")


def _install_client(monkeypatch: pytest.MonkeyPatch, module: Any, client: FakeClient):
    monkeypatch.setattr(module, "get_http_client", lambda: client)


def _responses_stream(lease: Any):
    return responses_router._stream_events(
        CreateResponseBody(model="model", input="hello", stream=True, store=False),
        "server-id",
        "agent-id",
        [],
        "resp-1",
        1,
        lease=lease,
    )


async def _initial_responses_frames(stream: Any) -> None:
    assert (await anext(stream)).startswith("event: response.created")
    assert (await anext(stream)).startswith("event: response.in_progress")


def _text_upstream() -> FakeStreamResponse:
    return FakeStreamResponse(
        [
            'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ],
    )


def _has_text_delta(frames: list[str]) -> bool:
    return any(
        json.loads(frame.split("data: ", 1)[1])["delta"] == "ok"
        for frame in frames
        if frame.startswith("event: response.output_text.delta")
    )


@pytest.mark.asyncio
async def test_responses_gufo_native_buffered_uses_responses_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """engine=gufo buffered /v1/responses must call the agent's native
    /v1/responses with a native payload (input + store=false), not chat."""
    events: list[str] = []
    lease = FakeLease(events, server=SimpleNamespace(engine="gufo"))
    captured: dict[str, Any] = {}

    async def fake_send(_agent_id, _method, path, payload, **_kwargs):
        captured["path"] = path
        captured["payload"] = payload
        return {
            "id": "resp_g",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "hi", "annotations": []}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 3,
                "total_tokens": 13,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "timings": {"prompt_ms": 1},
        }

    monkeypatch.setattr(responses_router.agent_manager, "send_to_agent", fake_send)
    result = await responses_router._complete(
        CreateResponseBody(model="qwen", input="hi", store=False),
        "server-1",
        "agent-1",
        [],
        "resp-1",
        1,
        lease,
    )
    assert captured["path"] == "/proxy/server-1/v1/responses"
    assert captured["payload"]["store"] is False
    assert "messages" not in captured["payload"]
    assert captured["payload"]["input"][0]["role"] == "user"
    assert result.status == "completed"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 3
    assert result.output[0].content[0].text == "hi"


@pytest.mark.asyncio
async def test_responses_llamacpp_buffered_still_uses_chat_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: non-gufo engines keep the chat/completions translation path."""
    events: list[str] = []
    lease = FakeLease(events, server=SimpleNamespace(engine="llamacpp"))
    captured: dict[str, Any] = {}

    async def fake_send(_agent_id, _method, path, payload, **_kwargs):
        captured["path"] = path
        captured["payload"] = payload
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }

    monkeypatch.setattr(responses_router.agent_manager, "send_to_agent", fake_send)
    result = await responses_router._complete(
        CreateResponseBody(model="m", input="hi", store=False),
        "server-1",
        "agent-1",
        [],
        "resp-1",
        1,
        lease,
    )
    assert captured["path"] == "/proxy/server-1/v1/chat/completions"
    assert "messages" in captured["payload"]
    assert result.output[0].content[0].text == "ok"


@pytest.mark.asyncio
async def test_responses_halogen_flash_native_buffered_uses_responses_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """engine=halogen-flash buffered /v1/responses must call the agent's native
    /v1/responses with the shared native payload (input + store=false + model),
    not chat."""
    events: list[str] = []
    lease = FakeLease(events, server=SimpleNamespace(engine="halogen-flash"))
    captured: dict[str, Any] = {}

    async def fake_send(_agent_id, _method, path, payload, **_kwargs):
        captured["path"] = path
        captured["payload"] = payload
        return {
            "id": "resp_hf",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "hi", "annotations": []}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 11,
                "output_tokens": 4,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 5},
                "output_tokens_details": {"reasoning_tokens": 1},
            },
            "timings": {"prompt_ms": 2},
        }

    monkeypatch.setattr(responses_router.agent_manager, "send_to_agent", fake_send)
    result = await responses_router._complete(
        CreateResponseBody(model="halogen-qwen3.8-flash-next", input="hi", store=False),
        "server-1",
        "agent-1",
        [],
        "resp-1",
        1,
        lease,
    )
    assert captured["path"] == "/proxy/server-1/v1/responses"
    assert captured["payload"]["store"] is False
    assert captured["payload"]["model"] == "halogen-qwen3.8-flash-next"
    assert "messages" not in captured["payload"]
    assert captured["payload"]["input"][0]["role"] == "user"
    assert result.status == "completed"
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 4
    assert result.output[0].content[0].text == "hi"


@pytest.mark.asyncio
async def test_responses_gufo_native_stream_uses_responses_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming engine=gufo must open the agent's native /v1/responses SSE and
    translate typed events (no [DONE] sentinel) into broker frames."""
    events: list[str] = []
    lease = FakeLease(events, server=SimpleNamespace(engine="gufo"))
    gufo_lines = [
        'data: {"type":"response.created","response":{"id":"resp_g"}}',
        'data: {"type":"response.in_progress","response":{"id":"resp_g"}}',
        'data: {"type":"response.output_text.delta","delta":"Hel"}',
        'data: {"type":"response.output_text.delta","delta":"lo"}',
        'data: {"type":"response.completed","response":{"id":"resp_g","status":"completed",'
        '"usage":{"input_tokens":8,"output_tokens":2,"total_tokens":10}}}',
    ]
    client = FakeClient(FakeStreamResponse(gufo_lines, events))
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
    stream = _responses_stream(lease)
    try:
        await _initial_responses_frames(stream)
        frames = [frame async for frame in stream]
    finally:
        await stream.aclose()
    assert client.stream_calls[0]["url"].endswith("/proxy/server-id/v1/responses")
    assert _has_text_delta_value(frames, "Hel")
    assert _has_text_delta_value(frames, "lo")
    assert any(frame.startswith("event: response.completed") for frame in frames)
    assert frames[-1] == "data: [DONE]\n\n"
    await _drain_release_tasks()


@pytest.mark.asyncio
async def test_responses_halogen_flash_native_stream_uses_responses_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming engine=halogen-flash must open the agent's native /v1/responses
    SSE and translate typed events (no [DONE] sentinel) into broker frames."""
    events: list[str] = []
    lease = FakeLease(events, server=SimpleNamespace(engine="halogen-flash"))
    flash_lines = [
        'data: {"type":"response.created","response":{"id":"resp_hf"}}',
        'data: {"type":"response.in_progress","response":{"id":"resp_hf"}}',
        'data: {"type":"response.output_text.delta","delta":"Hel"}',
        'data: {"type":"response.output_text.delta","delta":"lo"}',
        'data: {"type":"response.completed","response":{"id":"resp_hf","status":"completed",'
        '"usage":{"input_tokens":9,"output_tokens":2,"total_tokens":11}}}',
    ]
    client = FakeClient(FakeStreamResponse(flash_lines, events))
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
    stream = _responses_stream(lease)
    try:
        await _initial_responses_frames(stream)
        frames = [frame async for frame in stream]
    finally:
        await stream.aclose()
    assert client.stream_calls[0]["url"].endswith("/proxy/server-id/v1/responses")
    assert client.stream_calls[0]["json"]["model"] == "model"
    assert _has_text_delta_value(frames, "Hel")
    assert _has_text_delta_value(frames, "lo")
    assert any(frame.startswith("event: response.completed") for frame in frames)
    assert frames[-1] == "data: [DONE]\n\n"
    await _drain_release_tasks()


def _has_text_delta_value(frames: list[str], value: str) -> bool:
    return any(
        frame.startswith("event: response.output_text.delta")
        and json.loads(frame.split("data: ", 1)[1])["delta"] == value
        for frame in frames
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_phase", ["readiness", "headers"])
async def test_responses_keepalive_before_setup_completes(
    monkeypatch: pytest.MonkeyPatch, blocked_phase: str
) -> None:
    gate = asyncio.Event()
    entered = asyncio.Event()
    events: list[str] = []

    class DelayedResponse(FakeStreamResponse):
        async def __aenter__(self) -> FakeStreamResponse:
            entered.set()
            if blocked_phase == "headers":
                await gate.wait()
            return self

    async def ready(_server_id: str) -> SimpleNamespace:
        entered.set()
        if blocked_phase == "readiness":
            await gate.wait()
        return SimpleNamespace(agent_id="agent-id")

    upstream = DelayedResponse(_text_upstream().lines, events)
    client = FakeClient(upstream)
    _install_client(monkeypatch, responses_router, client)
    monkeypatch.setattr(responses_router, "SSE_KEEPALIVE_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(responses_router, "ensure_server_ready_by_id", ready)
    monkeypatch.setattr(
        responses_router.agent_manager,
        "get_agent",
        AsyncMock(return_value=SimpleNamespace(host="agent", port=8080)),
    )
    lease = FakeLease(events)
    stream = _responses_stream(lease)
    try:
        await _initial_responses_frames(stream)
        pending = asyncio.create_task(anext(stream))
        await asyncio.wait_for(entered.wait(), 1)
        assert await asyncio.wait_for(pending, 1) == ": keep-alive\n\n"
        assert not gate.is_set()
        gate.set()
        frames = [frame async for frame in stream]
        assert _has_text_delta(frames)
        assert any(frame.startswith("event: response.completed") for frame in frames)
        assert frames[-1] == "data: [DONE]\n\n"
        await _drain_release_tasks()
        assert events == ["upstream_closed", "released"]
        assert len(client.stream_calls) == 1
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_responses_discarded_lines_do_not_suppress_keepalive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    comments_seen = asyncio.Event()
    comments_finished = asyncio.Event()
    events: list[str] = []

    class NoisyResponse(FakeStreamResponse):
        def aiter_lines(self):
            async def iterate():
                for _ in range(20):
                    yield ": upstream queued"
                    yield ""
                    comments_seen.set()
                    await asyncio.sleep(0.003)
                comments_finished.set()
                await gate.wait()
                for line in self.lines:
                    yield line

            return iterate()

    client = FakeClient(NoisyResponse(_text_upstream().lines, events))
    _install_client(monkeypatch, responses_router, client)
    monkeypatch.setattr(responses_router, "SSE_KEEPALIVE_INTERVAL_SECONDS", 0.012)
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
    stream = _responses_stream(FakeLease(events))
    try:
        await _initial_responses_frames(stream)
        pending = asyncio.create_task(anext(stream))
        await asyncio.wait_for(comments_seen.wait(), 1)
        assert await asyncio.wait_for(pending, 1) == ": keep-alive\n\n"
        assert not comments_finished.is_set()
        assert not gate.is_set()
        gate.set()
        frames = [frame async for frame in stream]
        assert _has_text_delta(frames)
        assert any(frame.startswith("event: response.completed") for frame in frames)
        await _drain_release_tasks()
        assert events == ["upstream_closed", "released"]
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("close_reason", ["downstream", "lease_lost"])
async def test_responses_blocked_headers_cleanup(
    monkeypatch: pytest.MonkeyPatch, close_reason: str
) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    events: list[str] = []

    class BlockedResponse(FakeStreamResponse):
        async def __aenter__(self) -> FakeStreamResponse:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    client = FakeClient(BlockedResponse([], events))
    _install_client(monkeypatch, responses_router, client)
    monkeypatch.setattr(responses_router, "SSE_KEEPALIVE_INTERVAL_SECONDS", 0.01)
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
    lease = InferenceLeaseHandle("request-id", SimpleNamespace(), uuid.uuid4())
    lease.release = AsyncMock(side_effect=lambda *_: events.append("released"))
    stream = _responses_stream(lease)
    await _initial_responses_frames(stream)
    pending = asyncio.create_task(anext(stream))
    await asyncio.wait_for(entered.wait(), 1)
    assert await asyncio.wait_for(pending, 1) == ": keep-alive\n\n"
    if close_reason == "lease_lost":
        lease.lost.set()
        frames = [frame async for frame in stream]
        assert any(frame.startswith("event: response.failed") for frame in frames)
    else:
        await stream.aclose()
    await asyncio.wait_for(cancelled.wait(), 1)
    assert events == ["released"]
    assert client.closed is False
    lease.release.assert_awaited_once()
    lease.release.assert_awaited_once_with(
        "failed" if close_reason == "lease_lost" else "cancelled"
    )


@pytest.mark.asyncio
async def test_responses_cancel_task_while_waiting_for_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    events: list[str] = []

    class BlockedResponse(FakeStreamResponse):
        async def __aenter__(self) -> FakeStreamResponse:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    client = FakeClient(BlockedResponse([], events))
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
    lease = InferenceLeaseHandle("request-id", SimpleNamespace(), uuid.uuid4())
    lease.release = AsyncMock(side_effect=lambda *_: events.append("released"))
    stream = _responses_stream(lease)
    await _initial_responses_frames(stream)
    pending = asyncio.create_task(anext(stream))
    await asyncio.wait_for(entered.wait(), 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, 1)
    assert cancelled.is_set()
    assert events == ["released"]
    assert client.closed is False
    lease.release.assert_awaited_once_with("cancelled")


@pytest.mark.asyncio
async def test_responses_header_error_releases_lease_without_closing_shared_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FailingResponse(FakeStreamResponse):
        async def __aenter__(self) -> FakeStreamResponse:
            raise httpx.ReadTimeout("agent headers timed out")

    client = FakeClient(FailingResponse([]))
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
    lease = InferenceLeaseHandle("request-id", SimpleNamespace(), uuid.uuid4())
    lease.release = AsyncMock(side_effect=lambda *_: events.append("released"))
    stream = _responses_stream(lease)
    frames = [frame async for frame in stream]
    assert any(frame.startswith("event: response.failed") for frame in frames)
    assert frames[-1] == "data: [DONE]\n\n"
    assert events == ["released"]
    assert client.closed is False
    lease.release.assert_awaited_once_with("failed")


@pytest.mark.asyncio
async def test_responses_headers_arriving_at_disconnect_are_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = asyncio.Event()
    headers_arrived = asyncio.Event()
    events: list[str] = []

    class DelayedResponse(FakeStreamResponse):
        async def __aenter__(self) -> FakeStreamResponse:
            await gate.wait()
            headers_arrived.set()
            return self

    client = FakeClient(DelayedResponse([], events))
    _install_client(monkeypatch, responses_router, client)
    monkeypatch.setattr(responses_router, "SSE_KEEPALIVE_INTERVAL_SECONDS", 0.01)
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
    lease = FakeLease(events)
    stream = _responses_stream(lease)
    await _initial_responses_frames(stream)
    assert await asyncio.wait_for(anext(stream), 1) == ": keep-alive\n\n"
    gate.set()
    await asyncio.wait_for(headers_arrived.wait(), 1)
    await stream.aclose()
    assert events == ["upstream_closed", "released"]
    assert client.closed is False


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
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
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
    content = await asyncio.wait_for(anext(stream), 1)
    usage = await asyncio.wait_for(anext(stream), 1)

    assert '"text":"ok"' in content
    assert '"choices":[]' in usage
    assert events[:2] == ["upstream_closed", "released"]
    assert "backend_inference_stream_open" in caplog.text
    assert "request_id=fake-request-id" in caplog.text
    assert "close_reason=upstream_eof" in caplog.text
    assert client.stream_calls[0]["headers"] == {
        "X-Inference-Slot-Generation": "23",
        "X-Inference-Request-ID": "fake-request-id",
    }

    done = await anext(stream)
    assert done == "data: [DONE]\n\n"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "upstream_lines",
    [
        ['data: {"error":{"message":"invalid prompt"}}'],
        ['data: {"choices":[{"text":"partial"}]}'],
    ],
)
async def test_completion_stream_reports_agent_failure_instead_of_done(
    monkeypatch: pytest.MonkeyPatch, upstream_lines: list[str]
) -> None:
    events: list[str] = []
    _install_client(
        monkeypatch, v1_completions, FakeClient(FakeStreamResponse(upstream_lines))
    )
    monkeypatch.setattr(
        v1_completions.agent_manager,
        "get_agent",
        AsyncMock(return_value=SimpleNamespace(host="agent", port=8080)),
    )
    request = v1_completions.CompletionRequest(
        model="model", prompt="hello", stream=True
    )
    frames = [
        frame
        async for frame in v1_completions._stream_completion_via_agent(
            "agent-id", "server-id", request, "cmpl-1", FakeLease(events)
        )
    ]
    assert any('"error"' in frame for frame in frames)
    assert "data: [DONE]\n\n" not in frames
    assert events == ["released"]


@pytest.mark.asyncio
async def test_responses_sse_releases_before_persistence_without_gating_terminal(
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
            events.append("terminal")
        frames.append(frame)

    assert "terminal" in events
    assert events.index("released") < events.index("persisted")
    assert lease.upstream_started
    assert frames[-1] == "data: [DONE]\n\n"
    assert client.stream_calls[0]["headers"] == {
        "X-Inference-Slot-Generation": "23",
        "X-Inference-Request-ID": "fake-request-id",
    }


@pytest.mark.asyncio
async def test_responses_websocket_propagates_disconnect_and_releases(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    disconnected = asyncio.Event()
    disconnected.set()
    server = SimpleNamespace(id="server-id", agent_id="agent-id", model_id="model-id")
    model = SimpleNamespace(id="model-id")
    reservation_id = uuid.uuid4()
    lease = InferenceLeaseHandle(
        "request-id",
        server,
        uuid.uuid4(),
        slot_generation=31,
        reservation_id=reservation_id,
    )
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
    assert client.stream_calls[0]["headers"] == {
        "X-Inference-Slot-Generation": "31",
        "X-Inference-Request-ID": "request-id",
        "X-Inference-Reservation": "request-id",
        "X-Inference-Reservation-ID": str(reservation_id),
    }
    assert "responses_ws_stream_close" in caplog.text
    assert "close_reason=client_disconnect" in caplog.text


@pytest.mark.asyncio
async def test_responses_ws_halogen_flash_native_uses_responses_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WS transport for engine=halogen-flash opens the agent's native
    /v1/responses and translates typed events (no [DONE]) into broker WS
    messages."""
    disconnected = asyncio.Event()
    server = SimpleNamespace(
        id="server-id",
        agent_id="agent-id",
        model_id="model-id",
        engine="halogen-flash",
    )
    model = SimpleNamespace(id="model-id")
    lease = InferenceLeaseHandle("request-id", server, uuid.uuid4(), slot_generation=5)
    lease.release = AsyncMock()

    async def acquire(*_args: Any, **_kwargs: Any) -> InferenceLeaseHandle:
        return lease

    flash_lines = [
        'data: {"type":"response.output_text.delta","delta":"He"}',
        'data: {"type":"response.output_text.delta","delta":"y"}',
        'data: {"type":"response.completed","response":{"status":"completed",'
        '"usage":{"input_tokens":6,"output_tokens":2,"total_tokens":8}}}',
    ]
    client = FakeClient(FakeStreamResponse(flash_lines))
    _install_client(monkeypatch, responses_ws, client)
    monkeypatch.setattr(
        responses_ws, "resolve_target", AsyncMock(return_value=(server, model, None))
    )
    monkeypatch.setattr(responses_ws.inference_scheduler, "acquire", acquire)
    monkeypatch.setattr(
        responses_ws.agent_manager,
        "get_agent",
        AsyncMock(return_value=SimpleNamespace(host="agent", port=8080)),
    )
    sent: list[str] = []

    async def send_text(text: str) -> None:
        sent.append(text)

    websocket = SimpleNamespace(send_text=send_text)
    request = CreateResponseBody(model="model", input="hello", store=False)

    result = await responses_ws._stream_to_ws(
        websocket, request, disconnected=disconnected
    )

    assert client.stream_calls[0]["url"].endswith("/proxy/server-id/v1/responses")
    assert client.stream_calls[0]["json"]["model"] == "model"
    assert result["status"] == "completed"
    assert result["usage"]["input_tokens"] == 6
    deltas = [
        json.loads(m)["delta"]
        for m in sent
        if json.loads(m).get("type") == "response.output_text.delta"
    ]
    assert deltas == ["He", "y"]
    lease.release.assert_awaited()


@pytest.mark.asyncio
async def test_responses_ws_gufo_native_uses_responses_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WS transport for engine=gufo opens the agent's native /v1/responses and
    translates typed events (no [DONE]) into broker WS messages."""
    disconnected = asyncio.Event()
    server = SimpleNamespace(
        id="server-id", agent_id="agent-id", model_id="model-id", engine="gufo"
    )
    model = SimpleNamespace(id="model-id")
    lease = InferenceLeaseHandle("request-id", server, uuid.uuid4(), slot_generation=5)
    lease.release = AsyncMock()

    async def acquire(*_args: Any, **_kwargs: Any) -> InferenceLeaseHandle:
        return lease

    gufo_lines = [
        'data: {"type":"response.output_text.delta","delta":"He"}',
        'data: {"type":"response.output_text.delta","delta":"y"}',
        'data: {"type":"response.completed","response":{"status":"completed",'
        '"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}}',
    ]
    client = FakeClient(FakeStreamResponse(gufo_lines))
    _install_client(monkeypatch, responses_ws, client)
    monkeypatch.setattr(
        responses_ws, "resolve_target", AsyncMock(return_value=(server, model, None))
    )
    monkeypatch.setattr(responses_ws.inference_scheduler, "acquire", acquire)
    monkeypatch.setattr(
        responses_ws.agent_manager,
        "get_agent",
        AsyncMock(return_value=SimpleNamespace(host="agent", port=8080)),
    )
    sent: list[str] = []

    async def send_text(text: str) -> None:
        sent.append(text)

    websocket = SimpleNamespace(send_text=send_text)
    request = CreateResponseBody(model="model", input="hello", store=False)

    result = await responses_ws._stream_to_ws(
        websocket, request, disconnected=disconnected
    )

    assert client.stream_calls[0]["url"].endswith("/proxy/server-id/v1/responses")
    assert result["status"] == "completed"
    assert result["usage"]["input_tokens"] == 4
    deltas = [
        json.loads(m)["delta"]
        for m in sent
        if json.loads(m).get("type") == "response.output_text.delta"
    ]
    assert deltas == ["He", "y"]
    lease.release.assert_awaited()

"""Tests for transient upstream connection handling."""

import asyncio
import gc
import logging
import os
import subprocess
import sys
from unittest.mock import AsyncMock, Mock, patch

os.environ["AGENT_ID"] = "test-agent"
os.environ["FRONTEND_URL"] = "http://test:8000"
os.environ["MODELS_PATH"] = "/tmp/models"

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.main import app
from app.services.inference_operations import cancel_operation, get_operation
from app.services.llama_server import ServerConfig
from app.services.proxy import (
    ProxyOperationExists,
    ServerProxy,
)


class _BufferedUpstream:
    status_code = 200
    headers = {}
    request = httpx.Request("POST", "http://127.0.0.1:8091/v1/chat/completions")

    def __init__(self, enter):
        self.enter = enter

    async def __aenter__(self):
        await self.enter()
        return self

    async def __aexit__(self, *_args):
        pass

    async def aiter_bytes(self):
        yield b"{}"


@pytest.mark.asyncio
async def test_proxy_request_retries_connection_failures():
    manager = Mock(configs={"server-1": Mock(port=8091)}, _active_connections={})
    proxy = ServerProxy(manager)
    assert proxy.client.timeout.connect == 10.0
    response = Mock()
    proxy.client.request = AsyncMock(
        side_effect=[httpx.ConnectError("not ready"), response]
    )

    with patch("app.services.proxy.asyncio.sleep", new=AsyncMock()) as sleep:
        result = await proxy.proxy_request("server-1", "GET", "/health", {})

    assert result is response
    assert proxy.client.request.await_count == 2
    sleep.assert_awaited_once_with(0.5)
    assert manager._active_connections == {}


@pytest.mark.asyncio
async def test_proxy_request_logs_upstream_400_request_and_response(caplog):
    caplog.set_level(logging.ERROR)
    manager = Mock(configs={"server-1": Mock(port=8091)}, _active_connections={})
    proxy = ServerProxy(manager)
    response = httpx.Response(
        400,
        json={"error": {"message": "max_tokens exceeds cap"}},
        request=httpx.Request("POST", "http://127.0.0.1:8091/v1/chat/completions"),
    )
    proxy.client.request = AsyncMock(return_value=response)
    request_body = {"messages": [{"role": "user", "content": "hello"}]}

    await proxy.proxy_request(
        "server-1",
        "POST",
        "/v1/chat/completions",
        {},
        json=request_body,
    )

    assert "inference_upstream_bad_request" in caplog.text
    assert "max_tokens exceeds cap" in caplog.text
    assert str(request_body) in caplog.text


@pytest.mark.asyncio
async def test_proxy_stream_reports_upstream_400_as_sse_error(caplog):
    caplog.set_level(logging.ERROR)
    manager = Mock(
        configs={"server-1": Mock(port=8091, api_port=8091, slot_generation=1)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    error_body = b'{"error":{"message":"max_tokens exceeds cap"}}'
    request = httpx.Request("POST", "http://127.0.0.1:8091/v1/chat/completions")
    proxy.client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(400, content=error_body, request=request)
        )
    )
    request_body = {"messages": [{"role": "user", "content": "hello"}]}

    try:
        chunks = [
            chunk
            async for chunk in proxy.proxy_stream(
                "server-1",
                "POST",
                "/v1/chat/completions",
                json=request_body,
                enforce_capacity=True,
                expected_slot_generation=1,
                operation_id="bad-request-1",
            )
        ]
    finally:
        await proxy.client.aclose()

    assert chunks == [b'data: {"error": {"message": "max_tokens exceeds cap"}}\n\n']
    assert "inference_upstream_bad_request" in caplog.text
    assert "max_tokens exceeds cap" in caplog.text
    assert str(request_body) in caplog.text


@pytest.mark.asyncio
async def test_inference_proxy_admission_is_atomic_and_released_on_completion():
    manager = Mock(
        configs={"server-1": Mock(port=8091)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    entered_upstream = asyncio.Event()
    finish_upstream = asyncio.Event()
    second_entered = asyncio.Event()
    calls = 0

    async def upstream_request(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered_upstream.set()
            await finish_upstream.wait()
        else:
            second_entered.set()
        return Mock()

    proxy.client.stream = Mock(
        side_effect=lambda **_: _BufferedUpstream(upstream_request)
    )
    first = asyncio.create_task(
        proxy.proxy_request(
            "server-1",
            "POST",
            "/v1/chat/completions",
            {},
            json={},
            enforce_capacity=True,
        )
    )
    await entered_upstream.wait()
    # A reserved slot need not have an established upstream connection yet.
    assert manager._active_connections == {}

    second = asyncio.create_task(
        proxy.proxy_request(
            "server-1",
            "POST",
            "/v1/chat/completions",
            {},
            json={},
            enforce_capacity=True,
        )
    )
    await asyncio.sleep(0.02)
    assert not second_entered.is_set()
    assert manager._active_inference_requests == {"server-1": 1}
    assert manager._active_connections == {}

    finish_upstream.set()
    await first
    await second
    assert second_entered.is_set()
    assert manager._active_inference_requests == {}
    assert manager._active_connections == {}


@pytest.mark.asyncio
async def test_operation_id_prevents_duplicate_dispatch_and_can_cancel_upstream(caplog):
    caplog.set_level(logging.INFO)
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=4)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    entered_upstream = asyncio.Event()
    finish_upstream = asyncio.Event()

    async def upstream_request(**_kwargs):
        entered_upstream.set()
        await finish_upstream.wait()
        return Mock()

    proxy.client.stream = Mock(
        side_effect=lambda **_: _BufferedUpstream(upstream_request)
    )
    operation = asyncio.create_task(
        proxy.proxy_request(
            "server-1",
            "POST",
            "/v1/chat/completions",
            {},
            json={},
            enforce_capacity=True,
            operation_id="request-123",
        )
    )
    await entered_upstream.wait()

    status = get_operation(manager, "request-123", "server-1")
    assert status is not None
    assert status["status"] == "active"
    assert status["slot_generation"] == 4
    with pytest.raises(ProxyOperationExists):
        await proxy.proxy_request(
            "server-1",
            "POST",
            "/v1/chat/completions",
            {},
            json={},
            enforce_capacity=True,
            operation_id="request-123",
        )

    assert cancel_operation(manager, "request-123", "server-1") is True
    with pytest.raises(asyncio.CancelledError):
        await operation
    status = get_operation(manager, "request-123", "server-1")
    assert status is not None
    assert status["status"] == "cancelled"
    assert manager._active_inference_requests == {}
    assert "inference_proxy_http_close" in caplog.text
    assert "operation_id=request-123" in caplog.text


@pytest.mark.asyncio
async def test_queued_operation_is_visible_and_cancellable():
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=2)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    first_started = asyncio.Event()
    finish_first = asyncio.Event()

    async def upstream_request(**_kwargs):
        first_started.set()
        await finish_first.wait()
        return Mock()

    proxy.client.stream = Mock(
        side_effect=lambda **_: _BufferedUpstream(upstream_request)
    )
    first = asyncio.create_task(
        proxy.proxy_request(
            "server-1",
            "POST",
            "/v1/chat/completions",
            {},
            json={},
            enforce_capacity=True,
            operation_id="active-operation",
        )
    )
    await first_started.wait()
    queued = asyncio.create_task(
        proxy.proxy_request(
            "server-1",
            "POST",
            "/v1/chat/completions",
            {},
            json={},
            enforce_capacity=True,
            operation_id="queued-operation",
        )
    )
    await asyncio.sleep(0.02)
    status = get_operation(manager, "queued-operation", "server-1")
    assert status is not None
    assert status["status"] == "queued"

    assert cancel_operation(manager, "queued-operation", "server-1") is True
    with pytest.raises(asyncio.CancelledError):
        await queued
    finish_first.set()
    await first

    status = get_operation(manager, "queued-operation", "server-1")
    assert status is not None
    assert status["status"] == "cancelled"
    assert manager._active_inference_requests == {}


@pytest.mark.asyncio
async def test_protocol_three_reservation_is_atomic_idempotent_and_never_queues():
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=2)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    try:
        assert await proxy.try_reserve_inference_slot("server-1", "first", 2)
        assert await proxy.try_reserve_inference_slot("server-1", "first", 2)
        assert not await proxy.try_reserve_inference_slot("server-1", "second", 2)
        assert manager._active_inference_requests == {"server-1": 1}
        with pytest.raises(httpx.HTTPStatusError):
            await proxy.consume_inference_slot("server-1", "first", 3)
        assert await proxy.consume_inference_slot("server-1", "first", 2)
        assert not await proxy.consume_inference_slot("server-1", "first", 2)
        assert not await proxy.release_pending_reservation("server-1", "first", 2)
        await proxy._connection_finished("server-1", True, "first", generation=2)
        assert await proxy.try_reserve_inference_slot("server-1", "second", 2)
        assert await proxy.release_pending_reservation("server-1", "second", 2)
        assert manager._active_inference_requests == {}
    finally:
        await proxy.aclose()


@pytest.mark.asyncio
async def test_stale_reservation_token_cannot_cancel_or_dispatch_new_attempt():
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=2)},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    try:
        assert await proxy.try_reserve_inference_slot("server-1", "request", 2, "old")
        assert await proxy.release_pending_reservation("server-1", "request", 2, "old")
        assert await proxy.try_reserve_inference_slot("server-1", "request", 2, "new")
        assert not await proxy.release_pending_reservation(
            "server-1", "request", 2, "old"
        )
        assert not await proxy.consume_inference_slot("server-1", "request", 2, "old")
        assert await proxy.consume_inference_slot("server-1", "request", 2, "new")
        await proxy._connection_finished("server-1", True, "request", generation=2)
        assert manager._active_inference_requests == {}
    finally:
        await proxy.aclose()


@pytest.mark.asyncio
async def test_protocol_three_expired_reservation_can_be_retried_without_dispatch():
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=2)},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    try:
        assert await proxy.try_reserve_inference_slot("server-1", "first", 2)
        assert await proxy.release_pending_reservation("server-1", "first", 2)
        assert await proxy.try_reserve_inference_slot("server-1", "first", 2)
        assert await proxy.release_pending_reservation("server-1", "first", 2)
    finally:
        await proxy.aclose()


@pytest.mark.asyncio
async def test_cancelled_reservation_cannot_dispatch_after_response_headers():
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=2)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    try:
        assert await proxy.try_reserve_inference_slot("server-1", "first", 2)
        assert await proxy.consume_inference_slot("server-1", "first", 2)
        # The route task exits on returning StreamingResponse before its body
        # generator starts. Cancellation in that gap must not contact the LLM.
        manager._inference_operations["first"]["task"] = asyncio.create_task(
            asyncio.sleep(0)
        )
        await manager._inference_operations["first"]["task"]
        assert await proxy.release_pending_reservation("server-1", "first", 2)
        stream = proxy.proxy_stream(
            "server-1",
            "POST",
            "/v1/chat/completions",
            json={"stream": True},
            expected_slot_generation=2,
            enforce_capacity=True,
            capacity_reserved=True,
            operation_id="first",
        )
        with pytest.raises(ProxyOperationExists):
            await anext(stream)
        assert manager._active_inference_requests == {}
        assert manager._active_connections == {}
    finally:
        await proxy.aclose()


@pytest.mark.asyncio
async def test_stream_cancellation_after_slot_acquire_releases_exactly_once(caplog):
    caplog.set_level(logging.INFO)
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=7)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    acquired = asyncio.Event()
    hold_reservation_return = asyncio.Event()
    original_start = proxy._connection_started

    async def delayed_start(*args, **kwargs):
        await original_start(*args, **kwargs)
        acquired.set()
        await hold_reservation_return.wait()

    proxy._connection_started = delayed_start
    stream = proxy.proxy_stream(
        "server-1",
        "POST",
        "/v1/chat/completions",
        json={"stream": True},
        expected_slot_generation=7,
        enforce_capacity=True,
        operation_id="cancel-at-acquire",
    )
    consumer = asyncio.create_task(anext(stream))
    await acquired.wait()
    assert manager._active_inference_requests == {"server-1": 1}

    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert manager._active_inference_requests == {}
    assert manager._active_connections == {}
    status = get_operation(manager, "cancel-at-acquire", "server-1")
    assert status is not None
    assert status["status"] == "cancelled"
    assert "inference_stream_close" in caplog.text
    assert "close_reason=downstream_cancelled_before_llm_connect" in caplog.text


@pytest.mark.asyncio
async def test_cancel_operation_racing_with_reservation_never_dispatches():
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=11)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    acquired = asyncio.Event()
    resume = asyncio.Event()
    original = proxy._connection_started

    async def delayed(*args, **kwargs):
        await original(*args, **kwargs)
        acquired.set()
        await resume.wait()

    proxy._connection_started = delayed
    proxy.client.stream = Mock(side_effect=AssertionError("must not dispatch"))
    stream = proxy.proxy_stream(
        "server-1",
        "POST",
        "/v1/chat/completions",
        json={"stream": True},
        expected_slot_generation=11,
        enforce_capacity=True,
        operation_id="cancel-race-11",
    )
    consumer = asyncio.create_task(anext(stream))
    await acquired.wait()
    assert cancel_operation(manager, "cancel-race-11", "server-1")
    resume.set()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    assert manager._active_inference_requests == {}
    assert manager._active_connections == {}
    assert get_operation(manager, "cancel-race-11", "server-1")["status"] == "cancelled"


@pytest.mark.asyncio
async def test_reserved_and_open_counts_diverge_until_upstream_eof(caplog):
    caplog.set_level(logging.INFO)
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=3)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    finish = asyncio.Event()
    opened = asyncio.Event()

    class Upstream:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def aiter_bytes(self):
            opened.set()
            yield b"data: [DONE]\n\n"
            await finish.wait()

    proxy.client.stream = Mock(return_value=Upstream())
    stream = proxy.proxy_stream(
        "server-1",
        "POST",
        "/v1/chat/completions",
        json={"stream": True},
        expected_slot_generation=3,
        enforce_capacity=True,
        operation_id="upstream-eof-3",
    )
    assert await anext(stream) == b"data: [DONE]\n\n"
    await opened.wait()
    assert manager._active_inference_requests == {"server-1": 1}
    assert manager._active_connections == {"server-1": 1}
    finish.set()
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert manager._active_inference_requests == {}
    assert manager._active_connections == {}
    assert get_operation(manager, "upstream-eof-3", "server-1")["status"] == "completed"
    assert "operation_id=upstream-eof-3" in caplog.text


@pytest.mark.asyncio
async def test_upstream_eof_releases_slot_while_downstream_stops_reading():
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=12)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    finished = asyncio.Event()

    class Upstream:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            finished.set()

        async def aiter_bytes(self):
            yield b"data: first\n\n"
            yield b"data: [DONE]\n\n"

    proxy.client.stream = Mock(return_value=Upstream())
    downstream = proxy.proxy_stream_background(
        "server-1",
        "POST",
        "/v1/chat/completions",
        json={"stream": True},
        expected_slot_generation=12,
        enforce_capacity=True,
        operation_id="stalled-client-12",
    )
    assert await anext(downstream) == b"data: first\n\n"
    await asyncio.wait_for(finished.wait(), 1)
    # The consumer has not requested the second chunk yet.
    await asyncio.sleep(0)
    assert manager._active_connections == {}
    assert manager._active_inference_requests == {}
    assert (
        get_operation(manager, "stalled-client-12", "server-1")["status"] == "completed"
    )
    await downstream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body_chunks", "expected_reason", "done_marker"),
    [
        (
            [b"data: hello\n\n", b"data: [DONE]\n\n"],
            "llm_eof_after_done_marker",
            "True",
        ),
        ([b"data: hello\n\n"], "llm_eof_without_done_marker", "False"),
    ],
)
async def test_stream_close_log_distinguishes_llm_eof_from_done_marker(
    caplog, body_chunks, expected_reason, done_marker
):
    caplog.set_level(logging.INFO)
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=9)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)

    class UpstreamResponse:
        status_code = 200
        is_error = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def aiter_bytes(self):
            for chunk in body_chunks:
                yield chunk

    proxy.client.stream = Mock(return_value=UpstreamResponse())
    chunks = [
        chunk
        async for chunk in proxy.proxy_stream(
            "server-1",
            "POST",
            "/v1/chat/completions",
            json={"stream": True},
            expected_slot_generation=9,
            enforce_capacity=True,
            operation_id="stream-log-id",
        )
    ]

    assert chunks == body_chunks
    assert "operation_id=stream-log-id" in caplog.text
    assert f"close_reason={expected_reason}" in caplog.text
    assert f"done_marker={done_marker}" in caplog.text
    assert manager._active_inference_requests == {}


@pytest.mark.asyncio
async def test_stream_close_log_identifies_llm_connection_drop(caplog):
    caplog.set_level(logging.INFO)
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=10)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)

    class DroppedUpstream:
        status_code = 200
        is_error = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def aiter_bytes(self):
            yield b"data: partial\n\n"
            raise httpx.RemoteProtocolError("upstream closed mid-frame")

    proxy.client.stream = Mock(return_value=DroppedUpstream())
    stream = proxy.proxy_stream(
        "server-1",
        "POST",
        "/v1/chat/completions",
        json={"stream": True},
        expected_slot_generation=10,
        enforce_capacity=True,
        operation_id="llm-dropped-request",
    )

    with pytest.raises(httpx.RemoteProtocolError):
        async for _chunk in stream:
            pass

    assert "operation_id=llm-dropped-request" in caplog.text
    assert "error_type=RemoteProtocolError" in caplog.text
    assert "close_reason=llm_stream_error:RemoteProtocolError" in caplog.text
    assert manager._active_inference_requests == {}


@pytest.mark.asyncio
async def test_stream_idle_before_first_byte_retries_once_in_same_operation(caplog):
    caplog.set_level(logging.INFO)
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=13)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    attempts = 0

    class Upstream:
        status_code = 200

        async def __aenter__(self):
            nonlocal attempts
            attempts += 1
            assert manager._active_inference_requests == {"server-1": 1}
            assert manager._active_connections == {}
            return self

        async def __aexit__(self, *_args):
            pass

        async def aiter_bytes(self):
            if attempts == 1:
                raise httpx.ReadTimeout("LLM idle")
            yield b"data: [DONE]\n\n"

    proxy.client.stream = Mock(side_effect=lambda **_: Upstream())
    chunks = [
        chunk
        async for chunk in proxy.proxy_stream(
            "server-1",
            "POST",
            "/v1/chat/completions",
            json={"stream": True},
            enforce_capacity=True,
            expected_slot_generation=13,
            operation_id="idle-retry-13",
        )
    ]
    assert chunks == [b"data: [DONE]\n\n"]
    assert attempts == 2
    assert proxy.client.stream.call_args.kwargs["timeout"].read == (
        settings.UPSTREAM_IDLE_TIMEOUT_SECONDS
    )
    assert "inference_stream_idle_retry operation_id=idle-retry-13" in caplog.text
    assert "outcome=completed" in caplog.text
    assert manager._active_inference_requests == {}
    assert manager._active_connections == {}
    assert get_operation(manager, "idle-retry-13", "server-1")["status"] == "completed"


@pytest.mark.asyncio
async def test_idle_before_response_headers_retries_once():
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=20)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    attempts = 0

    class Upstream:
        status_code = 200

        async def __aenter__(self):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ReadTimeout("no headers")
            return self

        async def __aexit__(self, *_args):
            pass

        async def aiter_bytes(self):
            yield b"data: [DONE]\n\n"

    proxy.client.stream = Mock(side_effect=lambda **_: Upstream())
    frames = [
        frame
        async for frame in proxy.proxy_stream(
            "server-1",
            "POST",
            "/v1/chat/completions",
            json={"stream": True},
            enforce_capacity=True,
            expected_slot_generation=20,
            operation_id="headers-idle-20",
        )
    ]
    assert frames == [b"data: [DONE]\n\n"]
    assert attempts == 2
    assert (
        get_operation(manager, "headers-idle-20", "server-1")["status"] == "completed"
    )
    assert manager._active_connections == {}
    assert manager._active_inference_requests == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_stream_idle_exhaustion_or_partial_output_fails_without_replay(
    partial, caplog
):
    caplog.set_level(logging.INFO)
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=14)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    attempts = 0

    class Upstream:
        status_code = 200

        async def __aenter__(self):
            nonlocal attempts
            attempts += 1
            return self

        async def __aexit__(self, *_args):
            pass

        async def aiter_bytes(self):
            if partial:
                yield b"data: partial\n\n"
            raise httpx.ReadTimeout("LLM idle")

    proxy.client.stream = Mock(side_effect=lambda **_: Upstream())
    chunks = []
    with pytest.raises(httpx.ReadTimeout):
        async for chunk in proxy.proxy_stream(
            "server-1",
            "POST",
            "/v1/chat/completions",
            json={"stream": True},
            enforce_capacity=True,
            expected_slot_generation=14,
            operation_id="idle-failed-14",
        ):
            chunks.append(chunk)
    assert chunks == ([b"data: partial\n\n"] if partial else [])
    assert attempts == (1 if partial else 2)
    assert get_operation(manager, "idle-failed-14", "server-1")["status"] == "failed"
    assert manager._active_connections == {}
    assert manager._active_inference_requests == {}
    assert "operation_id=idle-failed-14" in caplog.text


@pytest.mark.asyncio
async def test_nonstream_idle_retries_only_before_any_body_bytes(caplog):
    caplog.set_level(logging.INFO)
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=15)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    attempts = 0

    class Upstream(_BufferedUpstream):
        async def __aenter__(self):
            nonlocal attempts
            attempts += 1
            return self

        async def aiter_bytes(self):
            if attempts == 1:
                raise httpx.ReadTimeout("LLM idle")
            yield b'{"ok":true}'

    proxy.client.stream = Mock(side_effect=lambda **_: Upstream(None))
    response = await proxy.proxy_request(
        "server-1",
        "POST",
        "/v1/chat/completions",
        {},
        json={},
        enforce_capacity=True,
        expected_slot_generation=15,
        operation_id="http-idle-15",
    )
    assert attempts == 2
    assert response.json() == {"ok": True}
    assert "inference_proxy_http_idle_retry operation_id=http-idle-15" in caplog.text
    assert get_operation(manager, "http-idle-15", "server-1")["status"] == "completed"
    assert manager._active_connections == {}


@pytest.mark.asyncio
async def test_nonstream_retry_works_with_real_httpx_stream_transport():
    manager = Mock(
        configs={"server-1": Mock(port=8091, api_port=8091, slot_generation=21)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    requests = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            if len(requests) == 1:
                raise httpx.ReadTimeout("first upstream stalled")
            yield b'{"ok":true}'

    def handler(request):
        requests.append(request)
        return httpx.Response(200, stream=Body())

    proxy = ServerProxy(manager)
    await proxy.client.aclose()
    proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        response = await proxy.proxy_request(
            "server-1",
            "POST",
            "/v1/chat/completions",
            {},
            json={},
            enforce_capacity=True,
            expected_slot_generation=21,
            operation_id="httpx-idle-21",
        )
        assert response.json() == {"ok": True}
        assert len(requests) == 2
        assert (
            get_operation(manager, "httpx-idle-21", "server-1")["status"] == "completed"
        )
        assert manager._active_inference_requests == {}
        assert manager._active_connections == {}
    finally:
        await proxy.client.aclose()


@pytest.mark.asyncio
async def test_nonstream_partial_body_idle_does_not_retry():
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=16)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    attempts = 0

    class Upstream(_BufferedUpstream):
        async def __aenter__(self):
            nonlocal attempts
            attempts += 1
            return self

        async def aiter_bytes(self):
            yield b'{"ok":'
            raise httpx.ReadTimeout("LLM idle")

    proxy.client.stream = Mock(side_effect=lambda **_: Upstream(None))
    with pytest.raises(httpx.ReadTimeout):
        await proxy.proxy_request(
            "server-1",
            "POST",
            "/v1/chat/completions",
            {},
            json={},
            enforce_capacity=True,
            expected_slot_generation=16,
            operation_id="partial-idle-16",
        )
    assert attempts == 1
    assert get_operation(manager, "partial-idle-16", "server-1")["status"] == "failed"
    assert manager._active_connections == {}
    assert manager._active_inference_requests == {}


@pytest.mark.asyncio
async def test_cancel_during_idle_retry_closes_same_operation():
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=17)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    second_started = asyncio.Event()
    attempts = 0

    class Upstream:
        status_code = 200

        async def __aenter__(self):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                second_started.set()
                await asyncio.Event().wait()
            return self

        async def __aexit__(self, *_args):
            pass

        async def aiter_bytes(self):
            raise httpx.ReadTimeout("LLM idle")
            yield b""  # pragma: no cover - marks this as an async iterator

    proxy.client.stream = Mock(side_effect=lambda **_: Upstream())
    stream = proxy.proxy_stream(
        "server-1",
        "POST",
        "/v1/chat/completions",
        json={"stream": True},
        enforce_capacity=True,
        expected_slot_generation=17,
        operation_id="cancel-retry-17",
    )
    consumer = asyncio.create_task(anext(stream))
    await asyncio.wait_for(second_started.wait(), 1)
    assert cancel_operation(manager, "cancel-retry-17", "server-1") is True
    with pytest.raises(asyncio.CancelledError):
        await consumer
    assert attempts == 2
    assert (
        get_operation(manager, "cancel-retry-17", "server-1")["status"] == "cancelled"
    )
    assert manager._active_inference_requests == {}
    assert manager._active_connections == {}


@pytest.mark.asyncio
async def test_generation_change_prevents_prebyte_idle_retry():
    manager = Mock(
        configs={"server-1": Mock(port=8091, slot_generation=18)},
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )
    proxy = ServerProxy(manager)
    attempts = 0

    class Upstream:
        status_code = 200

        async def __aenter__(self):
            nonlocal attempts
            attempts += 1
            return self

        async def __aexit__(self, *_args):
            pass

        async def aiter_bytes(self):
            manager.configs["server-1"].slot_generation = 19
            # Server stop fences the old generation's reservation counter.
            manager._active_inference_requests.clear()
            raise httpx.ReadTimeout("LLM idle")
            yield b""  # pragma: no cover - marks this as an async iterator

    proxy.client.stream = Mock(side_effect=lambda **_: Upstream())
    with pytest.raises(httpx.HTTPStatusError) as exc:
        async for _ in proxy.proxy_stream(
            "server-1",
            "POST",
            "/v1/chat/completions",
            json={"stream": True},
            enforce_capacity=True,
            expected_slot_generation=18,
            operation_id="old-generation-18",
        ):
            pass
    assert exc.value.response.status_code == 409
    assert attempts == 1
    assert manager._active_inference_requests == {}
    assert manager._active_connections == {}
    assert get_operation(manager, "old-generation-18", "server-1")["status"] == "failed"


@pytest.mark.asyncio
async def test_metrics_are_returned_as_prometheus_text(monkeypatch):
    from app.api.routes import proxy as proxy_route

    server_id = "server-1"
    proxy_route.server_manager.servers[server_id] = Mock()
    proxy_route.server_manager.configs[server_id] = ServerConfig(
        model_path="/models/test.gguf", port=8091, slot_generation=3
    )
    response = Mock(is_error=False, text="requests_processing 2\n")
    # The route uses the module-level shared proxy, so patch its method rather
    # than the ServerProxy class.
    monkeypatch.setattr(
        proxy_route.proxy, "proxy_request", AsyncMock(return_value=response)
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        result = await client.get(f"/proxy/{server_id}/metrics")

    assert result.status_code == 200
    assert result.json() == {"metrics": "requests_processing 2\n"}

    del proxy_route.server_manager.servers[server_id]
    del proxy_route.server_manager.configs[server_id]


@pytest.mark.asyncio
async def test_proxy_rejects_stale_slot_generation():
    from app.api.routes import proxy as proxy_route

    server_id = "generation-server"
    proxy_route.server_manager.servers[server_id] = Mock()
    proxy_route.server_manager.configs[server_id] = ServerConfig(
        model_path="/models/test.gguf", port=8091, slot_generation=8
    )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        result = await client.post(
            f"/proxy/{server_id}/v1/chat/completions",
            headers={"X-Inference-Slot-Generation": "7"},
            json={"messages": []},
        )

    assert result.status_code == 409
    del proxy_route.server_manager.servers[server_id]
    del proxy_route.server_manager.configs[server_id]


@pytest.mark.asyncio
async def test_operation_status_route_does_not_fall_through_to_proxy():
    from app.api.routes import proxy as proxy_route

    server_id = "operation-server"
    proxy_route.server_manager._inference_operations = {
        "operation-123": {
            "request_id": "operation-123",
            "server_id": server_id,
            "slot_generation": 5,
            "status": "active",
            "started_at": 1.0,
            "task": None,
        }
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/proxy/{server_id}/operations/operation-123")
        operation_list = await client.get(f"/proxy/{server_id}/operations")

    assert response.status_code == 200
    assert response.json()["status"] == "active"
    assert response.json()["slot_generation"] == 5
    assert operation_list.status_code == 200
    assert operation_list.json()["complete"] is True
    assert operation_list.json()["operations"][0]["request_id"] == "operation-123"
    del proxy_route.server_manager._inference_operations


def _counting_async_client_factory(counter: list, handler):
    """Return an AsyncClient subclass that records every construction.

    The transport is forced so a leaked per-request client cannot touch the
    network; the test measures only how many clients were built.
    """

    class CountingAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)
            counter.append(self)

    return CountingAsyncClient


@pytest.mark.asyncio
async def test_proxy_route_reuses_shared_client_no_per_request_leak(monkeypatch):
    """Request handling must not build a new httpx.AsyncClient per request."""
    from app.api.routes import proxy as proxy_route

    # The route must expose one module-level shared proxy created at import.
    shared_proxy = proxy_route.proxy
    assert isinstance(shared_proxy, ServerProxy)

    server_id = "shared-client-server"
    proxy_route.server_manager.servers[server_id] = Mock()
    proxy_route.server_manager.configs[server_id] = ServerConfig(
        model_path="/models/test.gguf", port=8091, slot_generation=4
    )
    proxied = []

    def handler(request):
        proxied.append(request)
        return httpx.Response(200, json={"model": "test"}, request=request)

    # Point the shared proxy at a mock transport so the real proxy_request path
    # runs without a live llama-server. Built before the counting window opens.
    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(shared_proxy, "client", mock_client)

    constructed: list = []
    counter = _counting_async_client_factory(constructed, handler)
    asgi_client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    try:
        # Any client built inside this window is a per-request leak.
        with patch("app.services.proxy.httpx.AsyncClient", counter):
            for _ in range(3):
                result = await asgi_client.get(f"/proxy/{server_id}/v1/models")
                assert result.status_code == 200
    finally:
        await asgi_client.aclose()
        await mock_client.aclose()

    assert constructed == []
    assert proxy_route.proxy is shared_proxy
    assert len(proxied) == 3

    del proxy_route.server_manager.servers[server_id]
    del proxy_route.server_manager.configs[server_id]


_KEEPALIVE_SERVER_SCRIPT = r"""
import socket
import sys
import threading
import time


def handle(conn):
    buf = b""
    try:
        while True:
            try:
                data = conn.recv(65536)
            except OSError:
                break
            if not data:
                break
            buf += data
            while b"\r\n\r\n" in buf:
                head, buf = buf.split(b"\r\n\r\n", 1)
                length = 0
                for line in head.split(b"\r\n")[1:]:
                    if line.lower().startswith(b"content-length:"):
                        length = int(line.split(b":", 1)[1].strip())
                while len(buf) < length:
                    more = conn.recv(65536)
                    if not more:
                        return
                    buf += more
                # Hold briefly so concurrent requests need distinct pooled
                # connections instead of being strictly serialized.
                time.sleep(0.02)
                body = b'{"ok":true}'
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                    b"Connection: keep-alive\r\n\r\n" + body
                )
    finally:
        try:
            conn.close()
        except OSError:
            pass


listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", 0))
listener.listen(128)
print(listener.getsockname()[1], flush=True)
while True:
    try:
        conn, _ = listener.accept()
    except OSError:
        break
    threading.Thread(target=handle, args=(conn,), daemon=True).start()
"""


class _KeepAliveUpstream:
    """Real HTTP/1.1 keep-alive server in a child process.

    Running it out of process keeps the server-side accepted sockets out of
    ``/proc/self/fd`` so the measurement reflects only agent-side descriptors.
    """

    def __init__(self):
        self._process = subprocess.Popen(
            [sys.executable, "-c", _KEEPALIVE_SERVER_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        line = self._process.stdout.readline().strip()
        if not line:
            raise RuntimeError("keep-alive upstream failed to start")
        self.port = int(line)

    def stop(self):
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except Exception:  # noqa: BLE001 - best-effort teardown
            self._process.kill()


def _fd_count() -> int | None:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return None


@pytest.mark.asyncio
async def test_shared_proxy_bounded_fds_under_keepalive(monkeypatch):
    """Descriptors must be capped by the pool bound, not by the request count."""
    from app.api.routes import proxy as proxy_route

    baseline = _fd_count()
    if baseline is None:
        pytest.skip("/proc/self/fd is unavailable")

    upstream = _KeepAliveUpstream()
    server_id = "keepalive-server"
    manager = proxy_route.server_manager
    manager.servers[server_id] = Mock()
    manager.configs[server_id] = ServerConfig(
        model_path="/models/test.gguf", port=upstream.port, slot_generation=1
    )
    shared_proxy = proxy_route.proxy
    # Tight bounds: 30 leaked per-request clients would blow straight past this.
    monkeypatch.setattr(settings, "PROXY_MAX_CONNECTIONS", 4)
    monkeypatch.setattr(settings, "PROXY_MAX_KEEPALIVE_CONNECTIONS", 4)
    bounded_proxy = ServerProxy(manager)
    original_client = shared_proxy.client
    monkeypatch.setattr(shared_proxy, "client", bounded_proxy.client)
    n_requests = 30
    try:
        before = _fd_count()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            results = await asyncio.gather(
                *(
                    client.get(f"/proxy/{server_id}/v1/models")
                    for _ in range(n_requests)
                )
            )
        assert all(r.status_code == 200 for r in results)
        after = _fd_count()
        assert after - before <= settings.PROXY_MAX_KEEPALIVE_CONNECTIONS + 2, (
            f"fd delta {after - before} exceeded the keepalive pool bound"
        )
        assert not shared_proxy.client.is_closed
    finally:
        monkeypatch.setattr(shared_proxy, "client", original_client)
        await bounded_proxy.aclose()
        upstream.stop()
        manager.servers.pop(server_id, None)
        manager.configs.pop(server_id, None)

    gc.collect()
    settled = _fd_count()
    assert settled - before <= 2, (
        f"closing the proxy did not release pooled sockets: delta {settled - before}"
    )


@pytest.mark.asyncio
async def test_shutdown_closes_shared_proxy():
    """The app shutdown handler must close the shared upstream client."""
    from app.api.routes import proxy as proxy_route

    assert callable(proxy_route.close_proxy)
    assert not proxy_route.proxy.client.is_closed

    try:
        for handler in app.router.on_shutdown:
            await handler()
        assert proxy_route.proxy.client.is_closed
        # Closing again is a no-op, not an error.
        await proxy_route.close_proxy()
        assert proxy_route.proxy.client.is_closed
    finally:
        proxy_route.proxy = ServerProxy(proxy_route.server_manager)
        assert not proxy_route.proxy.client.is_closed

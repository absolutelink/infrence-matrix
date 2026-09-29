"""Tests for transient upstream connection handling."""

import asyncio
import logging
import os
from unittest.mock import AsyncMock, Mock, patch

os.environ["AGENT_ID"] = "test-agent"
os.environ["FRONTEND_URL"] = "http://test:8000"
os.environ["MODELS_PATH"] = "/tmp/models"

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.inference_operations import cancel_operation, get_operation
from app.services.llama_server import ServerConfig
from app.services.proxy import (
    ProxyOperationExists,
    ServerProxy,
)


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

    proxy.client.request = AsyncMock(side_effect=upstream_request)
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

    proxy.client.request = AsyncMock(side_effect=upstream_request)
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

    proxy.client.request = AsyncMock(side_effect=upstream_request)
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
async def test_metrics_are_returned_as_prometheus_text(monkeypatch):
    from app.api.routes import proxy as proxy_route

    server_id = "server-1"
    proxy_route.server_manager.servers[server_id] = Mock()
    proxy_route.server_manager.configs[server_id] = ServerConfig(
        model_path="/models/test.gguf", port=8091, slot_generation=3
    )
    response = Mock(is_error=False, text="requests_processing 2\n")
    proxy = Mock()
    proxy.proxy_request = AsyncMock(return_value=response)
    monkeypatch.setattr(proxy_route, "ServerProxy", Mock(return_value=proxy))

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

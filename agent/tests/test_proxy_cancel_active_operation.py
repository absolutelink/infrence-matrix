"""Regression tests for cancelling a running (``active``) inference operation.

Production bug: when the backend released a lease on a downstream disconnect it
POSTed ``/proxy/{server_id}/operations/{request_id}/cancel`` with an
``X-Inference-Reservation-ID`` header. The plain proxy dispatch path
(``_connection_started`` -> ``activate_operation``) never stores a
``reservation_id`` on the running operation, so the route's auth gate
``operation.get("reservation_id") != reservation_id`` evaluated to
``None != <uuid>`` -> ``True`` and returned ``cancelled: false`` WITHOUT ever
calling ``cancel_operation``. The bound producer task (the ``proxy_stream`` drain
holding ``async with self.client.stream(...)``) kept generating the abandoned
response to completion, holding the llama.cpp inference slot.

These tests drive the real route handler and assert the in-flight upstream stream
is actually torn down and the slot is released, while the reservation auth is not
weakened.
"""

import asyncio
import os

os.environ.setdefault("AGENT_ID", "test-agent")
os.environ.setdefault("FRONTEND_URL", "http://localhost:8000")
os.environ.setdefault("MODELS_PATH", "/tmp/models")

from unittest.mock import Mock

import pytest

from app.api.routes import proxy as proxy_route
from app.services.inference_operations import get_operation
from app.services.proxy import ServerProxy


class _HoldingUpstream:
    """Upstream that yields one chunk then blocks forever until cancelled.

    ``closed`` is set from ``__aexit__`` so the test can prove the llama.cpp
    stream (and therefore the inference slot) is actually released on cancel.
    """

    status_code = 200
    headers: dict = {}

    def __init__(self, closed: asyncio.Event) -> None:
        self._closed = closed
        self.entered = asyncio.Event()

    async def __aenter__(self) -> "_HoldingUpstream":
        self.entered.set()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self._closed.set()

    async def aiter_bytes(self):
        yield b"data: hello\n\n"
        # Hold the connection open like a still-generating llama.cpp response.
        await asyncio.Event().wait()
        yield b""  # pragma: no cover - unreachable, keeps this an async iterator


def _make_manager(generation: int = 5) -> Mock:
    return Mock(
        servers={},
        configs={
            "server-1": Mock(port=8091, api_port=8091, slot_generation=generation)
        },
        _active_connections={},
        _active_inference_requests={},
        _inference_slot_conditions={},
        _inference_operations={},
        get_effective_capacity=Mock(return_value=1),
    )


async def _start_running_operation(
    proxy: ServerProxy, manager: Mock, operation_id: str, generation: int
) -> tuple[asyncio.Task, asyncio.Event]:
    """Launch a background stream and wait until it is ``active`` with a bound task.

    Mirrors the real halogen-flash dispatch: the operation reaches ``active`` via
    the plain ``_connection_started``/``activate_operation`` path, so the stored
    operation carries NO ``reservation_id`` and NO ``dispatched`` flag. Returns
    the bound producer task and an event that fires when the upstream stream is
    closed.
    """
    closed = asyncio.Event()
    upstreams: list[_HoldingUpstream] = []

    def stream_factory(**_kwargs):
        upstream = _HoldingUpstream(closed)
        upstreams.append(upstream)
        return upstream

    proxy.client.stream = Mock(side_effect=stream_factory)

    generator = proxy.proxy_stream_background(
        "server-1",
        "POST",
        "/v1/chat/completions",
        json={"stream": True},
        expected_slot_generation=generation,
        enforce_capacity=True,
        operation_id=operation_id,
    )
    # Pulling the first frame drives the producer through slot acquisition,
    # bind_operation_task, and into the held upstream stream.
    first = await anext(generator)
    assert first == b"data: hello\n\n"
    await asyncio.wait_for(upstreams[0].entered.wait(), 1)

    bound_task = manager._inference_operations[operation_id]["task"]
    assert bound_task is not None
    assert not bound_task.done()
    # Slot is held by the running ghost generation.
    assert manager._active_inference_requests == {"server-1": 1}
    return bound_task, closed


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_reservation_id", ["lease-owner-uuid", None])
async def test_cancel_token_less_active_operation_is_honored_and_frees_slot(
    monkeypatch, cancel_reservation_id
):
    """A running active op with no stored reservation_id must be cancellable.

    This is the production scenario: the plain dispatch path leaves the running
    operation without a reservation token, yet the lease owner must still be able
    to abort it. Before the fix the gate returned ``cancelled: false`` and the
    producer kept running to completion.
    """
    generation = 5
    manager = _make_manager(generation)
    proxy = ServerProxy(manager)
    monkeypatch.setattr(proxy_route, "server_manager", manager)
    monkeypatch.setattr(proxy_route, "proxy", proxy)
    try:
        operation_id = "ghost-active-1"
        bound_task, upstream_closed = await _start_running_operation(
            proxy, manager, operation_id, generation
        )

        # Mirror the real halogen-flash dispatch: no reservation_id on the op.
        stored = get_operation(manager, operation_id, "server-1")
        assert stored is not None
        assert stored["status"] == "active"
        assert "reservation_id" not in stored

        result = await proxy_route.cancel_inference_operation(
            "server-1", operation_id, reservation_id=cancel_reservation_id
        )

        # The cancel must be honored, not a no-op ``cancelled: false``.
        assert result["cancelled"] is True
        assert result["status"] in {"cancelling", "cancelled"}

        # The bound producer task must stop (the ghost generation is aborted,
        # not left running to completion).
        await asyncio.wait_for(bound_task, 1)
        assert bound_task.done()

        # The upstream llama stream is actually closed and the slot released
        # exactly once.
        await asyncio.wait_for(upstream_closed.wait(), 1)
        await asyncio.wait_for(_drain_slot(manager), 1)
        assert manager._active_inference_requests == {}
        assert manager._active_connections == {}
        final = get_operation(manager, operation_id, "server-1")
        assert final is not None
        assert final["status"] == "cancelled"
    finally:
        await proxy.aclose()


async def _drain_slot(manager: Mock) -> None:
    while manager._active_inference_requests:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_cancel_with_mismatched_reservation_id_is_rejected(monkeypatch):
    """Auth is NOT weakened: a mismatched reservation_id on an op that HAS one
    must still reject the cancel and leave the running task untouched.
    """
    generation = 6
    manager = _make_manager(generation)
    proxy = ServerProxy(manager)
    monkeypatch.setattr(proxy_route, "server_manager", manager)
    monkeypatch.setattr(proxy_route, "proxy", proxy)
    try:
        operation_id = "reserved-active-2"
        bound_task, _ = await _start_running_operation(
            proxy, manager, operation_id, generation
        )
        # Simulate a reservation-protocol operation that carries a token.
        manager._inference_operations[operation_id]["reservation_id"] = "good-token"

        result = await proxy_route.cancel_inference_operation(
            "server-1", operation_id, reservation_id="bad-token"
        )

        assert result["cancelled"] is False
        assert result["status"] == "active"
        # The running producer must NOT have been cancelled by the wrong token.
        assert not bound_task.done()
        # Slot still held.
        assert manager._active_inference_requests == {"server-1": 1}

        # The correct token still cancels it (happy path preserved).
        ok = await proxy_route.cancel_inference_operation(
            "server-1", operation_id, reservation_id="good-token"
        )
        assert ok["cancelled"] is True
        await asyncio.wait_for(bound_task, 1)
        await _drain_slot(manager)
        assert manager._active_inference_requests == {}
    finally:
        await proxy.aclose()


@pytest.mark.asyncio
async def test_cancel_completed_operation_is_not_resurrected(monkeypatch):
    """A finished operation must not be flipped back to cancelling/cancelled."""
    generation = 7
    manager = _make_manager(generation)
    proxy = ServerProxy(manager)
    monkeypatch.setattr(proxy_route, "server_manager", manager)
    monkeypatch.setattr(proxy_route, "proxy", proxy)
    try:
        operation_id = "done-op-3"
        manager._inference_operations[operation_id] = {
            "request_id": operation_id,
            "server_id": "server-1",
            "slot_generation": generation,
            "status": "completed",
            "started_at": 1.0,
            "finished_at": 2.0,
            "task": None,
        }
        result = await proxy_route.cancel_inference_operation(
            "server-1", operation_id, reservation_id="whatever"
        )
        assert result["cancelled"] is False
        assert result["status"] == "completed"
    finally:
        await proxy.aclose()

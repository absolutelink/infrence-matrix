"""Tests for the /ws/status WebSocket route.

The handler must JOIN (await) its heartbeat and sender tasks before returning,
not merely call ``Task.cancel()``. ``cancel()`` only schedules cancellation, so
without a join the handler returns while the tasks are still unwinding: their
exceptions are never retrieved ("Task exception was never retrieved" on every
reconnect) and the heartbeat can outlive the socket it writes to.
"""

import asyncio
import contextlib
from collections.abc import Callable
from concurrent.futures import CancelledError as FutureCancelledError

from starlette.testclient import TestClient

from app.api.routes import websocket as websocket_module
from app.main import create_app

# Long enough that "cancelled but not finished" is observable for many loop cycles.
TEARDOWN_DELAY = 0.1


class _FakeWebSocket:
    """Minimal WebSocket stand-in that can be flipped into a closed state."""

    def __init__(self) -> None:
        self.accepted = asyncio.Event()
        self.sent: list[dict] = []
        self.closed = False

    async def accept(self) -> None:
        self.accepted.set()

    async def send_json(self, payload: dict) -> None:
        if self.closed:
            raise RuntimeError("Cannot send on a closed WebSocket")
        self.sent.append(payload)


async def _slow_teardown_heartbeat(websocket: _FakeWebSocket) -> None:
    """Stands in for ``_heartbeat``: loops, unwinds slowly, then errors.

    The ``finally`` body mimics a real teardown against a dying socket: it awaits
    (so cancellation is not instantaneous) and then raises a non-CancelledError.
    """
    try:
        while True:
            await websocket.send_json({"event": "heartbeat", "data": {}})
            await asyncio.sleep(0.01)
    finally:
        await asyncio.sleep(TEARDOWN_DELAY)
        await websocket.send_json({"event": "heartbeat", "data": {}})


async def _slow_teardown_sender(
    websocket: _FakeWebSocket, queue: asyncio.Queue
) -> None:
    """Stands in for ``_send_events``: blocks on the queue, unwinds slowly."""
    try:
        while True:
            await websocket.send_json(await queue.get())
    finally:
        await asyncio.sleep(TEARDOWN_DELAY)
        await websocket.send_json({"event": "heartbeat", "data": {}})


class _CreateTaskSpy:
    """Drop-in for the module's ``asyncio`` reference that records created tasks.

    Only ``create_task`` is intercepted; every other attribute is delegated to
    the real module, so the patch stays scoped to ``app.api.routes.websocket``.
    """

    def __init__(self, records: list) -> None:
        self._records = records

    def __getattr__(self, name: str) -> object:
        return getattr(asyncio, name)

    def create_task(self, coro, **kwargs):  # noqa: ANN001, ANN201
        task = asyncio.create_task(coro, **kwargs)
        self._records.append(task)
        return task


async def _wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout)


async def test_ws_status_handler_joins_background_tasks(monkeypatch) -> None:  # noqa: ANN001
    """The handler must not return until both background tasks are done.

    Regression guard for a ``finally`` that called ``cancel()`` without awaiting
    the tasks: the handler returned while both tasks were still unwinding.
    """
    created: list = []
    monkeypatch.setattr(websocket_module, "asyncio", _CreateTaskSpy(created))
    monkeypatch.setattr(websocket_module, "_heartbeat", _slow_teardown_heartbeat)
    monkeypatch.setattr(websocket_module, "_send_events", _slow_teardown_sender)

    websocket = _FakeWebSocket()
    handler = asyncio.create_task(websocket_module.websocket_endpoint(websocket))
    try:
        await _wait_for(
            lambda: (
                websocket.accepted.is_set()
                and len(created) == 2
                and len(websocket.sent) >= 1
            )
        )
        heartbeat_task, sender_task = created
        assert not heartbeat_task.done()
        assert not sender_task.done()

        # The socket dies, then the connection is torn down the way a server
        # tears it down: cancellation of the handler task.
        websocket.closed = True
        handler.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await handler

        # Joined, not merely cancelled.
        assert heartbeat_task.done(), "handler returned while heartbeat was cancelling"
        assert sender_task.done(), "handler returned while sender was cancelling"
        # The dead-socket errors raised during teardown were retrieved.
        assert isinstance(heartbeat_task.exception(), RuntimeError)
        assert isinstance(sender_task.exception(), RuntimeError)
    finally:
        # Never leak tasks whose exceptions would be logged as unretrieved.
        for task in created:
            if not task.done():
                task.cancel()
        if created:
            await asyncio.gather(*created, return_exceptions=True)
        handler.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await handler


def test_ws_status_streams_heartbeat() -> None:
    """The endpoint still accepts and delivers heartbeats over a real WebSocket."""
    client = TestClient(create_app())
    message = None
    with contextlib.suppress(FutureCancelledError):
        with client.websocket_connect("/ws/status") as websocket:
            message = websocket.receive_json()
        # TestClient runs the app in an anyio blocking portal. Joining the route's
        # background tasks happens inside that portal's cancelled scope, so the
        # portal future reports the cancellation at teardown. The heartbeat frame
        # is already in hand above, so this is teardown noise, not a route failure.

    assert message is not None, "no frame received from /ws/status"
    assert message["event"] == "heartbeat"
    assert "timestamp" in message["data"]

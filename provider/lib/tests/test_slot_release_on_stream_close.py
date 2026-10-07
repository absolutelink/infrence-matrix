"""THE critical test: slot release is tied to the UPSTREAM stream lifetime.

A completed (or abandoned) backend request must never hold a slot open
because a downstream HTTP client connection lingers, and an abandoned
stream must close the upstream generator (running its `finally`) as the
release trigger.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from provider_lib.app_factory import BackendOverrides, create_provider_app
from provider_lib.backend import BackendDriver, BackendLifecycle
from provider_lib.config import ProviderSettings
from provider_lib.wire import BackendStatusValue


class RecordingDriver(BackendDriver):
    """Driver whose response stream records open/close and can block forever."""

    def __init__(self, *, infinite: bool = False, events: int = 3) -> None:
        self.infinite = infinite
        self.events = events
        self.opened = 0
        self.closed = 0
        self.exhausted = 0
        self.release_seen: list[int] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def health(self) -> bool:
        return True

    async def list_models(self) -> list[dict[str, Any]]:
        return [{"id": "rec-model", "object": "model"}]

    def stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[dict[str, Any]]:
        self.opened += 1
        try:
            i = 0
            while self.infinite or i < self.events:
                yield {"type": "response.output_text.delta", "delta": f"t{i}"}
                i += 1
                await asyncio.sleep(0.01)
            self.exhausted += 1
        finally:
            self.closed += 1


def _settings() -> ProviderSettings:
    return ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="m-slot",
        MACHINE_SECRET="tok",
        AGENT_ID="m-slot-agent",
        ADMIN_BASE_URL="http://admin:8000",
        CACHE_DIR="/tmp/prov_test_cache",
        MODELS_DIR="/tmp/prov_test_models",
    )


async def test_normal_exhaustion_releases_slot() -> None:
    driver = RecordingDriver(events=3)
    lifecycle = BackendLifecycle(driver, capacity=1)
    await lifecycle.start()

    stream = await lifecycle.stream_responses({})
    assert lifecycle.in_flight == 1
    assert lifecycle.backend_status == BackendStatusValue.IN_USE

    events = [e async for e in stream]
    assert len(events) == 3
    assert driver.closed == 1
    assert driver.exhausted == 1
    assert lifecycle.in_flight == 0
    assert lifecycle.backend_status == BackendStatusValue.RUNNING


async def test_abandoned_stream_closes_upstream_and_releases_slot() -> None:
    """Consumer takes one event then abandons; upstream must still close."""
    driver = RecordingDriver(infinite=True)
    lifecycle = BackendLifecycle(driver, capacity=1)
    await lifecycle.start()

    stream = await lifecycle.stream_responses({})
    assert lifecycle.in_flight == 1

    # Consume exactly one delta, then close the consumer generator early
    # (simulates the HTTP response being torn down mid-stream).
    it = stream.__aiter__()
    first = await it.__anext__()
    assert first["type"] == "response.output_text.delta"
    assert driver.closed == 0  # infinite upstream is still open
    await stream.aclose()

    # The close must propagate: upstream generator finally ran, slot freed.
    assert driver.closed == 1
    assert lifecycle.in_flight == 0
    assert lifecycle.backend_status == BackendStatusValue.RUNNING


async def test_slot_released_even_if_consumer_never_iterates() -> None:
    """Eager pump: the upstream is drained/closed without any consumption."""
    driver = RecordingDriver(events=2)
    lifecycle = BackendLifecycle(driver, capacity=1)
    await lifecycle.start()

    _stream = await lifecycle.stream_responses({})  # never iterated
    # Give the pump time to finish on its own.
    for _ in range(200):
        if lifecycle.in_flight == 0:
            break
        await asyncio.sleep(0.01)

    assert driver.closed == 1
    assert driver.exhausted == 1
    assert lifecycle.in_flight == 0
    assert lifecycle.backend_status == BackendStatusValue.RUNNING


async def test_http_disconnect_cancels_upstream_and_releases_slot() -> None:
    """Raw-ASGI simulation of a real uvicorn client disconnect.

    httpx's ASGITransport buffers the whole response, so TestClient
    cannot model a mid-stream disconnect. Instead we drive the real
    POST /v1/responses route at the ASGI layer, read the response start
    plus one SSE body message, then signal `http.disconnect` the way a
    server does when the client goes away mid-stream. The disconnect must
    propagate through the consumer to the pump, close the upstream
    generator (its `finally` runs), and release the slot.
    """
    driver = RecordingDriver(infinite=True)
    lifecycle = BackendLifecycle(driver, capacity=1)
    await lifecycle.start()

    app = create_provider_app(
        _settings(),
        BackendOverrides(provider_type="test", version="v", lifecycle=lifecycle),
    )

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "headers": [(b"content-type", b"application/json")],
        "scheme": "http",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "server": ("test", 80),
        "client": ("test", 1234),
    }
    disconnect = asyncio.Event()
    request_read = False
    sent: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def receive() -> dict[str, Any]:
        nonlocal request_read
        if not request_read:
            request_read = True
            return {
                "type": "http.request",
                "body": b'{"stream": true}',
                "more_body": False,
            }
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        await sent.put(message)

    task = asyncio.create_task(app(scope, receive, send))
    start_msg = await asyncio.wait_for(sent.get(), 5)
    assert start_msg["type"] == "http.response.start"
    assert start_msg["status"] == 200
    body1 = await asyncio.wait_for(sent.get(), 5)
    assert body1["type"] == "http.response.body"
    assert lifecycle.in_flight == 1
    assert driver.closed == 0  # infinite stream still live while client reads

    # Client disconnects mid-stream.
    disconnect.set()
    await asyncio.wait_for(task, 5)

    assert driver.closed == 1
    assert lifecycle.in_flight == 0
    assert lifecycle.backend_status == BackendStatusValue.RUNNING


async def test_capacity_two_streams_release_independently() -> None:
    driver = RecordingDriver(infinite=True)
    lifecycle = BackendLifecycle(driver, capacity=2)
    await lifecycle.start()

    s1 = await lifecycle.stream_responses({})
    s2 = await lifecycle.stream_responses({})
    assert lifecycle.in_flight == 2

    # Start consuming s1 so its consumer frame is live, then abandon it:
    # only its slot frees. (aclose() on a never-started async generator
    # would not run the consumer's finally, so we pull one event first.)
    it1 = s1.__aiter__()
    await it1.__anext__()
    await s1.aclose()
    await asyncio.sleep(0.05)
    assert lifecycle.in_flight == 1
    assert lifecycle.backend_status == BackendStatusValue.IN_USE

    # Abandon the second: back to RUNNING.
    it2 = s2.__aiter__()
    await it2.__anext__()
    await s2.aclose()
    await asyncio.sleep(0.05)
    assert lifecycle.in_flight == 0
    assert lifecycle.backend_status == BackendStatusValue.RUNNING
    assert driver.closed == 2


async def test_pump_cancelled_while_upstream_fails_still_releases() -> None:
    """Production-observed leak: the pump task is cancelled (client abort)
    while it sits INSIDE its finally's first await (upstream teardown that
    blocks after the upstream had already failed mid-iteration). Cancelled-
    Error is a BaseException, so the old finally — `suppress(Exception):
    await upstream.aclose()` — re-raised at that await and skipped both
    _STREAM_END and release_slot(): /health reported in_flight: 1 forever
    and every new request got a 429. The fix (shield + uncancel + suppress
    BaseException) must land the release DESPITE the pending cancellation."""
    import asyncio as _asyncio

    aclose_gate = _asyncio.Event()

    class _DyingDriver(RecordingDriver):
        async def stream_responses(
            self, request: dict[str, Any]
        ) -> AsyncIterator[dict[str, Any]]:
            self.opened += 1
            try:
                yield {"type": "response.created"}
                raise RuntimeError("peer closed connection")
            finally:
                # Simulate slow/hung upstream teardown: the generator's own
                # aclose blocks until released (the pump awaits this in its
                # finally via upstream.aclose()).
                await aclose_gate.wait()
                self.closed += 1

    driver = _DyingDriver()
    lifecycle = BackendLifecycle(driver, capacity=1)
    await lifecycle.start()
    await lifecycle.acquire_slot()

    upstream = driver.stream_responses({})
    queue: _asyncio.Queue[object] = _asyncio.Queue()
    pump = _asyncio.create_task(lifecycle._pump(upstream, queue))
    first = await _asyncio.wait_for(queue.get(), 2)
    assert first == {"type": "response.created"}

    # The upstream failure propagates into the pump's finally, which blocks
    # on the gated generator aclose.
    await _asyncio.sleep(0.05)
    assert not pump.done(), "pump should be suspended in finally on gate"

    # THE PRODUCTION RACE: cancel the pump while it is suspended inside its
    # own finally. Old code: CancelledError re-raised at that await →
    # release_slot never ran → slot leaked forever. Fixed code: the shield
    # keeps each finally-await alive, and the slot release lands even with
    # the cancellation pending and the gated aclose still unfinished.
    pump.cancel()
    await _asyncio.sleep(0.05)

    assert lifecycle.in_flight == 0, "slot leaked after pump cancel in finally"
    assert lifecycle.backend_status == BackendStatusValue.RUNNING

    # Drain state settles: open the teardown gate so the shielded aclose
    # child completes and the pump task finishes with its CancelledError.
    aclose_gate.set()
    await _asyncio.wait_for(_asyncio.gather(pump, return_exceptions=True), 5)
    await _asyncio.sleep(0.05)
    # NOTE: deliberately NO `driver.closed` assert here — aclose() throws
    # GeneratorExit into the generator while it is suspended at the gate,
    # so its post-gate line never runs. The contract under test is the
    # slot release above.

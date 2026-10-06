"""Phase 13 log streaming tests: batched flush, provider.log handler,
backend.logs.get command."""

import asyncio
import logging

from provider_lib.log_ring import CursorLogRing
from provider_lib.log_stream import (
    LogStreamer,
    ProviderLogHandler,
    install_log_streaming,
)
from provider_lib.wire import Frame


class FakeClient:
    def __init__(self, settings=None) -> None:
        self.events: list[tuple[str, dict]] = []
        self.handlers: dict = {}
        self.settings = settings if settings is not None else _Settings()

    def on_command(self, name, handler) -> None:
        self.handlers[name] = handler

    async def send_event(self, type_, payload) -> None:
        self.events.append((type_, payload))


class _Settings:
    LOG_FLUSH_INTERVAL_SECONDS = 0.05
    LOG_RING_LINES = 50


def _streamer(client, backend_ring, provider_ring, **kw):
    return LogStreamer(
        client,
        {
            "backend": lambda: backend_ring,
            "provider": lambda: provider_ring,
        },
        interval=kw.pop("interval", 0.05),
        **kw,
    )


async def test_flush_batches_lines_and_updates_cursor() -> None:
    client = FakeClient()
    backend = CursorLogRing(max_lines=100)
    provider = CursorLogRing(max_lines=100)
    streamer = _streamer(client, backend, provider)
    streamer.start()
    try:
        for i in range(5):
            backend.append("stdout", f"b{i}")
        provider.append("stdout", "p0")
        await asyncio.sleep(0.3)
    finally:
        await streamer.stop()

    back = [p for t, p in client.events if t == "backend.logs"]
    prov = [p for t, p in client.events if t == "provider.logs"]
    assert len(back) == 1
    assert len(prov) == 1
    assert [e["text"] for e in back[0]["lines"]] == ["b0", "b1", "b2", "b3", "b4"]
    assert back[0]["dropped"] == 0
    assert prov[0]["lines"][0]["text"] == "p0"
    assert all("ts" in e and "stream" in e for e in back[0]["lines"])

    # No new lines -> no new frames on subsequent ticks.
    n = len(client.events)
    await asyncio.sleep(0.2)
    assert len(client.events) == n


async def test_flush_caps_at_max_lines_per_frame() -> None:
    client = FakeClient()
    backend = CursorLogRing(max_lines=1000)
    streamer = _streamer(client, backend, CursorLogRing(10), max_lines_per_frame=100)
    for i in range(250):
        backend.append("stdout", f"l{i}")
    frames = 0
    for _ in range(3):
        frames += await streamer.flush_once()
    assert frames == 3  # 100 + 100 + 50
    sent = [
        e
        for t, payload in client.events
        if t == "backend.logs"
        for e in payload["lines"]
    ]
    assert len(sent) == 250
    assert [e["text"] for e in sent] == [f"l{i}" for i in range(250)]
    # second flush_once on drained ring sends nothing
    assert await streamer.flush_once() == 0


async def test_dropped_counter_propagates() -> None:
    client = FakeClient()
    backend = CursorLogRing(max_lines=5)
    streamer = _streamer(client, backend, CursorLogRing(5))
    for i in range(8):
        backend.append("stdout", f"l{i}")
    assert backend.dropped == 3
    await streamer.flush_once()
    payload = client.events[0][1]
    assert payload["dropped"] == 3
    # oldest 3 were evicted; the ring now holds l3..l7
    assert [e["text"] for e in payload["lines"]] == ["l3", "l4", "l5", "l6", "l7"]


async def test_no_backend_ring_skips_backend_logs() -> None:
    client = FakeClient()
    provider = CursorLogRing(10)
    provider.append("stdout", "only-provider")
    streamer = LogStreamer(
        client,
        {"backend": lambda: None, "provider": lambda: provider},
        interval=0.05,
    )
    await streamer.flush_once()
    assert [t for t, _ in client.events] == ["provider.logs"]


async def test_provider_log_handler_captures_records() -> None:
    ring = CursorLogRing(max_lines=100)
    handler = ProviderLogHandler(ring)
    test_logger = logging.getLogger("provider.unittest.tmp")
    test_logger.setLevel(logging.DEBUG)
    test_logger.addHandler(handler)
    try:
        test_logger.info("hello %s", "world")
        test_logger.debug("below level, ignored")
    finally:
        test_logger.removeHandler(handler)
    entries = ring.tail(10)
    assert len(entries) == 1
    assert entries[0].text == "hello world"
    assert entries[0].stream == "stdout"


async def test_provider_log_handler_no_recursion() -> None:
    ring = CursorLogRing(max_lines=100)
    handler = ProviderLogHandler(ring)

    # Simulate a producer that logs while the handler is mid-emit: the
    # thread-local busy guard must turn the re-entrant emit into a no-op.
    real_append = ring.append
    reentrant_calls = []

    def append_with_logging(stream, text):
        # A pathological append that itself logs to the captured logger.
        reentrant_calls.append(text)
        handler.emit(logging.LogRecord("x", logging.ERROR, __file__, 1, text, (), None))
        return real_append(stream, text)

    ring.append = append_with_logging  # type: ignore[method-assign]
    # Prime: one normal entry, then the busy guard is what stops recursion.
    ring.append = real_append  # type: ignore[method-assign]
    handler.emit(logging.LogRecord("x", logging.ERROR, __file__, 1, "first", (), None))
    assert [e.text for e in ring.tail(10)] == ["first"]

    # Direct re-entrancy: busy flag set -> emit is a no-op.
    handler._local.busy = True
    handler.emit(
        logging.LogRecord("x", logging.ERROR, __file__, 2, "ignored", (), None)
    )
    handler._local.busy = False
    assert [e.text for e in ring.tail(10)] == ["first"]

    # Exceptions inside emit are swallowed (never propagate to the
    # logging machinery / application code).
    broken = CursorLogRing(max_lines=5)

    def exploding_append(stream, text):  # noqa: ARG001
        raise RuntimeError("ring exploded")

    broken.append = exploding_append  # type: ignore[method-assign]
    h2 = ProviderLogHandler(broken)
    h2.emit(
        logging.LogRecord("x", logging.ERROR, __file__, 3, "nope", (), None)
    )  # no raise


class _QuietStderr:
    def __enter__(self):
        import io
        import sys

        self._old = sys.stderr
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, *a):
        import sys

        sys.stderr = self._old


def _quiet_stderr():
    return _QuietStderr()


async def test_logs_get_handler_kind_and_since() -> None:
    client = FakeClient()

    class Driver:
        def __init__(self) -> None:
            self.log_ring = CursorLogRing(max_lines=100)

    class Lifecycle:
        def __init__(self) -> None:
            self.driver = Driver()

    lifecycle = Lifecycle()
    bundle = install_log_streaming(client, lifecycle, attach_provider_handler=False)
    for i in range(3):
        lifecycle.driver.log_ring.append("stderr", f"be{i}")
    bundle.provider_ring.append("stdout", "pv0")

    ack = await bundle.handle_logs_get(
        Frame(type="backend.logs.get", id="c1", payload={"kind": "backend", "since": 0})
    )
    assert ack["ok"] is True
    assert [e["text"] for e in ack["detail"]["lines"]] == ["be0", "be1", "be2"]
    assert ack["detail"]["seq"] == 3
    assert ack["detail"]["dropped"] == 0

    ack = await bundle.handle_logs_get(
        Frame(type="backend.logs.get", id="c2", payload={"kind": "provider"})
    )
    assert [e["text"] for e in ack["detail"]["lines"]] == ["pv0"]

    ack = await bundle.handle_logs_get(
        Frame(type="backend.logs.get", id="c3", payload={"kind": "all", "since": 0})
    )
    texts = sorted(e["text"] for e in ack["detail"]["lines"])
    assert texts == ["be0", "be1", "be2", "pv0"]

    # since resumes: only newer entries
    ack = await bundle.handle_logs_get(
        Frame(type="backend.logs.get", id="c4", payload={"kind": "backend", "since": 2})
    )
    assert [e["text"] for e in ack["detail"]["lines"]] == ["be2"]

    # null since == whole buffer
    ack = await bundle.handle_logs_get(
        Frame(
            type="backend.logs.get", id="c5", payload={"kind": "backend", "since": None}
        )
    )
    assert len(ack["detail"]["lines"]) == 3

    # handler registered on the client
    assert "backend.logs.get" in client.handlers


async def test_logs_get_all_shared_counter_resume_no_gaps_no_dups() -> None:
    """H1 proof: backend+provider rings share one seq space; a single
    kind=all `since` cursor resumes both with no missed/duplicated
    lines, and the ack `seq` is the global max+1."""
    client = FakeClient()

    class Driver:
        def __init__(self) -> None:
            self.log_ring = CursorLogRing(max_lines=100)

    class Lifecycle:
        def __init__(self) -> None:
            self.driver = Driver()

    lifecycle = Lifecycle()
    bundle = install_log_streaming(client, lifecycle, attach_provider_handler=False)
    backend = lifecycle.driver.log_ring
    provider = bundle.provider_ring
    # The install unified the two rings into one seq space.
    assert backend.seq_source is provider.seq_source

    # Interleave appends across both rings.
    backend.append("stdout", "b1")  # seq 0
    provider.append("stdout", "p1")  # seq 1
    backend.append("stderr", "b2")  # seq 2
    provider.append("stdout", "p2")  # seq 3
    provider.append("stdout", "p3")  # seq 4

    ack = await bundle.handle_logs_get(
        Frame(type="backend.logs.get", id="c1", payload={"kind": "all", "since": 0})
    )
    first = ack["detail"]["lines"]
    assert [e["text"] for e in first] == ["b1", "p1", "b2", "p2", "p3"]
    assert ack["detail"]["seq"] == 5  # global max + 1

    # Resume from the returned cursor with two new lines on each ring.
    since = ack["detail"]["seq"]
    backend.append("stdout", "b3")  # seq 5
    provider.append("stdout", "p4")  # seq 6
    ack2 = await bundle.handle_logs_get(
        Frame(
            type="backend.logs.get",
            id="c2",
            payload={"kind": "all", "since": since},
        )
    )
    assert [e["text"] for e in ack2["detail"]["lines"]] == ["b3", "p4"]
    assert ack2["detail"]["seq"] == 7
    # No duplicates against the first batch, nothing missed.
    seen = {e["text"] for e in first} | {e["text"] for e in ack2["detail"]["lines"]}
    assert seen == {"b1", "p1", "b2", "p2", "p3", "b3", "p4"}


async def test_streamer_resumes_after_stop_start() -> None:
    client = FakeClient()
    backend = CursorLogRing(max_lines=100)
    streamer = _streamer(client, backend, CursorLogRing(10))
    backend.append("stdout", "one")
    await streamer.flush_once()
    # produced while "disconnected" (streamer stopped)
    backend.append("stdout", "two")
    streamer.start()
    await asyncio.sleep(0.2)
    await streamer.stop()
    texts = [
        e["text"] for t, p in client.events if t == "backend.logs" for e in p["lines"]
    ]
    assert texts == ["one", "two"]

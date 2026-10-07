"""Batched, throttled log transport for provider instances (Phase 13).

Wires the cursor rings (``provider_lib.log_ring``) to the admin WS:

- ``LogStreamer`` — background flush task that drains each registered
  ring since its last flushed cursor and emits ONE batched frame per
  chunk (~1s interval, max 100 lines per frame, docs/ws-protocol.md §4):
  ``{"lines": [{"ts", "stream", "text"}...], "dropped": int}``.
  Started on WS connect, stopped on disconnect; cursors persist across
  connections so lines produced while disconnected are sent on
  reconnect (or served via ``backend.logs.get``).
- ``ProviderLogHandler`` — a ``logging.Handler`` that captures the
  provider process's own ``provider.*`` logger output into a separate
  ring, flushed as ``provider.logs`` by the same streamer.
- ``install_log_streaming`` — one call per provider package: builds the
  provider ring + handler + streamer and registers the
  ``backend.logs.get`` catch-up command handler on the client.

Recursion safety: ``ProviderLogHandler.emit`` only appends to the ring —
it never calls a logger, and a thread-local busy guard makes accidental
re-entry (e.g. from ``handleError`` paths) a no-op. The streamer's own
warnings flow through the captured ``provider.log_stream`` logger, but
each failure yields at most one extra ring entry per tick (bounded, not
self-amplifying).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable
from typing import Any

from provider_lib.log_ring import DEFAULT_RING_LINES, CursorLogRing, SeqCounter
from provider_lib.wire import FrameKind

logger = logging.getLogger("provider.log_stream")

DEFAULT_FLUSH_INTERVAL_SECONDS = 1.0
DEFAULT_MAX_LINES_PER_FRAME = 100
# Safety cap on frames per ring per tick so a huge backlog cannot flood
# the WS queue in one go; the remainder ships on the next tick.
MAX_FRAMES_PER_TICK = 10

# Ring name -> event type emitted for its entries.
_EVENT_BY_RING = {
    "backend": FrameKind.BACKEND_LOGS,
    "provider": FrameKind.PROVIDER_LOGS,
}


class LogStreamer:
    """Drains registered rings into batched WS log events.

    ``rings`` maps a ring name (``"backend"`` / ``"provider"``) to a
    zero-arg getter returning the current ``CursorLogRing`` or ``None``
    (a driver without a ring — e.g. mock — simply never emits
    ``backend.logs``). Getters are re-evaluated every tick so a driver
    swapped in after install is picked up automatically.
    """

    def __init__(
        self,
        client: Any,
        rings: dict[str, Callable[[], CursorLogRing | None]],
        *,
        interval: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
        max_lines_per_frame: int = DEFAULT_MAX_LINES_PER_FRAME,
        backend_instance_id: Callable[[], str | None] | None = None,
    ) -> None:
        self._client = client
        self._rings = rings
        self._interval = interval
        self._max_lines = max_lines_per_frame
        # H2: backend.logs frames must carry the owning instance_id so the
        # admin keys them under im:logs:backend:{instance_id} (where the REST
        # log tail reads them), not the agent id.
        self._backend_instance_id = backend_instance_id
        self._cursors: dict[str, int] = dict.fromkeys(rings, 0)
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start the periodic flush loop (no-op if already running).

        Resumes from the last flushed cursors — never resets the rings.
        """
        if self.running:
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.flush_once()
            except Exception:  # noqa: BLE001 - logs are best-effort
                logger.warning("log flush tick failed", exc_info=True)

    async def flush_once(self) -> int:
        """Drain every ring once; returns the number of frames queued."""
        sent = 0
        for name, getter in self._rings.items():
            ring = getter()
            if ring is None:
                continue
            event_type = _EVENT_BY_RING[name]
            for _ in range(MAX_FRAMES_PER_TICK):
                # Single-lock after(): entries + dropped are a consistent
                # snapshot of the ring (M3).
                entries, _gap, dropped = ring.after(
                    self._cursors[name], self._max_lines
                )
                if not entries:
                    break
                payload = {
                    "lines": [
                        {"ts": e.ts, "stream": e.stream, "text": e.text}
                        for e in entries
                    ],
                    "dropped": dropped,
                }
                if name == "backend" and self._backend_instance_id is not None:
                    iid = self._backend_instance_id()
                    if iid is not None:
                        payload["instance_id"] = iid
                await self._client.send_event(event_type, payload)
                self._cursors[name] = entries[-1].seq + 1
                sent += 1
                if len(entries) < self._max_lines:
                    break
        return sent


class ProviderLogHandler(logging.Handler):
    """Captures the provider's own log records into a CursorLogRing.

    Attach to the ``provider`` logger namespace; every record at or
    above ``level`` becomes a ``("stdout", message)`` ring entry.
    """

    def __init__(self, ring: CursorLogRing, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self._ring = ring
        # Re-entrancy guard: anything logged while emit() runs must not
        # recurse back into emit().
        self._local = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._local, "busy", False):
            return
        self._local.busy = True
        try:
            self._ring.append("stdout", record.getMessage())
        except Exception:  # noqa: BLE001 - never break the log emitter
            self.handleError(record)
        finally:
            self._local.busy = False


class LogStreamingBundle:
    """Handles returned by :func:`install_log_streaming`."""

    def __init__(
        self,
        streamer: LogStreamer,
        provider_ring: CursorLogRing,
        backend_ring_getter: Callable[[], CursorLogRing | None],
    ) -> None:
        self.streamer = streamer
        self.provider_ring = provider_ring
        self.backend_ring_getter = backend_ring_getter

    def start(self) -> None:
        self.streamer.start()

    async def stop(self) -> None:
        await self.streamer.stop()

    async def handle_logs_get(self, frame: Any) -> dict[str, Any]:
        """``backend.logs.get`` ack detail: {lines, dropped, seq}.

        ``kind`` selects backend / provider / all rings; ``since``
        resumes at the ring sequence (null/0 = whole current buffer).
        ``seq`` in the reply is the cursor the caller should resume from
        (max last-sent seq + 1 across the returned rings).

        NOTE (H1): the backend and provider rings installed by
        :func:`install_log_streaming` share ONE ``SeqCounter``, so a
        single ``since``/``seq`` cursor resumes both correctly for
        ``kind="all"``. This provider-side seq space is UNRELATED to the
        admin REST ``GET /logs?since=`` cursor (assigned by the admin's
        own ``im:logs:seq:{id}`` Redis counter at ingest) — clients must
        never mix the two.
        """
        payload = getattr(frame, "payload", None) or {}
        kind = payload.get("kind", "backend")
        since_raw = payload.get("since")
        try:
            since = int(since_raw) if since_raw is not None else 0
        except (TypeError, ValueError):  # fmt: skip
            since = 0
        try:
            limit = int(payload.get("limit", 5000))
        except (TypeError, ValueError):  # fmt: skip
            limit = 5000

        rings: list[CursorLogRing] = []
        if kind in ("backend", "all"):
            backend = self.backend_ring_getter()
            if backend is not None:
                rings.append(backend)
        if kind in ("provider", "all"):
            rings.append(self.provider_ring)

        entries_all: list[Any] = []
        dropped = 0
        seq = 0
        for ring in rings:
            entries, _gap, ring_dropped = ring.after(since, limit)
            entries_all.extend(entries)
            dropped += ring_dropped
            seq = max(seq, entries[-1].seq + 1 if entries else ring.last_cursor)
        # Shared SeqCounter => seq is the canonical interleaved order
        # across the two rings (ts can tie or skew across threads).
        entries_all.sort(key=lambda e: e.seq)
        lines = [{"ts": e.ts, "stream": e.stream, "text": e.text} for e in entries_all]
        return {"ok": True, "detail": {"lines": lines, "dropped": dropped, "seq": seq}}


def install_log_streaming(
    client: Any,
    lifecycle: Any,
    *,
    interval: float | None = None,
    max_lines_per_frame: int = DEFAULT_MAX_LINES_PER_FRAME,
    provider_ring: CursorLogRing | None = None,
    attach_provider_handler: bool = True,
) -> LogStreamingBundle:
    """Wire Phase 13 log streaming onto a provider package.

    - Builds (or adopts) the provider-owned ring and attaches a
      ``ProviderLogHandler`` to the ``provider`` logger namespace.
    - Resolves the backend ring from the lifecycle's driver attribute
      ``log_ring`` (a driver without one — mock — simply never produces
      ``backend.logs``).
    - Registers the ``backend.logs.get`` command handler on the client.
    - Returns the bundle; the caller starts/stops the streamer with the
      WS connection lifecycle (``on_connected`` / ``on_disconnected``).
    """
    if interval is None:
        interval = float(
            getattr(client.settings, "LOG_FLUSH_INTERVAL_SECONDS", 0.0)
            or DEFAULT_FLUSH_INTERVAL_SECONDS
        )

    def backend_ring_getter() -> CursorLogRing | None:
        ring = getattr(getattr(lifecycle, "driver", None), "log_ring", None)
        return ring if isinstance(ring, CursorLogRing) else None

    # H1: put BOTH rings in ONE monotonic seq space so the
    # backend.logs.get kind="all" single since/seq cursor resumes both
    # correctly. The driver's ring exists before install (created in
    # __init__ and never swapped for the process lifetime), so we adopt
    # its SeqCounter as the shared source; without a backend ring (mock)
    # the provider ring just owns its own counter.
    if provider_ring is None:
        backend = backend_ring_getter()
        shared = backend.seq_source if backend is not None else SeqCounter()
        provider_ring = CursorLogRing(
            int(getattr(client.settings, "LOG_RING_LINES", 0) or DEFAULT_RING_LINES),
            seq_source=shared,
        )

    if attach_provider_handler:
        provider_logger = logging.getLogger("provider")
        # Idempotent: don't stack handlers across reinstalls.
        if not any(isinstance(h, ProviderLogHandler) for h in provider_logger.handlers):
            provider_logger.addHandler(ProviderLogHandler(provider_ring))

    def backend_ring_getter() -> CursorLogRing | None:
        ring = getattr(getattr(lifecycle, "driver", None), "log_ring", None)
        return ring if isinstance(ring, CursorLogRing) else None

    streamer = LogStreamer(
        client,
        {"backend": backend_ring_getter, "provider": lambda: provider_ring},
        interval=interval,
        max_lines_per_frame=max_lines_per_frame,
        backend_instance_id=lambda: getattr(lifecycle, "instance_id", None),
    )
    bundle = LogStreamingBundle(streamer, provider_ring, backend_ring_getter)
    client.on_command("backend.logs.get", bundle.handle_logs_get)
    return bundle

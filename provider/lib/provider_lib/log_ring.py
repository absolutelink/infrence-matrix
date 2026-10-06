"""Cursor-addressable log ring shared by all provider packages (Phase 13).

Single canonical implementation replacing the four per-provider copies.
A bounded ring of log entries carrying a monotonic sequence number so the
batched live forwarder (``provider_lib.log_stream.LogStreamer``) and the
``backend.logs.get`` catch-up command share the same cursors, plus a
cumulative ``dropped`` counter for lines evicted beyond ``max_lines``.

Entry shape on the wire (docs/ws-protocol.md §4):
``{"seq": int, "ts": iso-utc, "stream": str, "text": str}``.

Thread safety: the subprocess readers run ``stream.readline`` in
``asyncio.to_thread`` but call ``append`` back on the event loop; the
``ProviderLogHandler`` can emit from any thread. A plain ``threading.Lock``
around the small critical sections keeps producers and the flusher from
racing without blocking the loop meaningfully.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

DEFAULT_RING_LINES = 2000


class SeqCounter:
    """Lock-protected shared monotonic sequence source.

    Pass the SAME instance to several ``CursorLogRing``s so their seqs
    come from one ordered space — this is what makes the
    ``backend.logs.get`` ``kind="all"`` single-cursor resume correct
    (backend and provider rings interleave in one sequence instead of
    two independent ~0-start spaces).
    """

    def __init__(self, start: int = 0) -> None:
        self._next = start
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            value = self._next
            self._next += 1
            return value

    @property
    def value(self) -> int:
        """Cursor just past the last issued seq."""
        with self._lock:
            return self._next


@dataclass(frozen=True)
class LogEntry:
    seq: int
    ts: str
    stream: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


class CursorLogRing:
    """Ring buffer of log entries with a monotonic per-feed cursor.

    ``seq_source`` lets several rings share one monotonic sequence space
    (see ``SeqCounter``); by default each ring owns a private counter.
    """

    def __init__(
        self,
        max_lines: int = DEFAULT_RING_LINES,
        *,
        seq_source: SeqCounter | None = None,
    ) -> None:
        self.max_lines = max_lines
        self._seq = seq_source if seq_source is not None else SeqCounter()
        self._entries: list[LogEntry] = []
        self._dropped: int = 0
        self._lock = threading.Lock()

    @property
    def first_cursor(self) -> int:
        with self._lock:
            return self._entries[0].seq if self._entries else self._seq.value

    @property
    def last_cursor(self) -> int:
        """Cursor just past the newest seq ever drawn by this ring.

        With a shared ``SeqCounter`` this is the global next-seq (>=
        every seq any sharing ring holds), which keeps cross-ring
        resume cursors monotonic.
        """
        return self._seq.value

    @property
    def dropped(self) -> int:
        """Cumulative lines lost to eviction since ring creation."""
        with self._lock:
            return self._dropped

    @property
    def seq_source(self) -> SeqCounter:
        """This ring's sequence source (share it to unify seq spaces)."""
        return self._seq

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def append(self, stream: str, text: str) -> LogEntry:
        # The seq is drawn and the entry stored under this ring's lock so
        # per-ring seq order always matches ts order; the SeqCounter's own
        # lock keeps cross-ring interleaving atomic too (nested lock here
        # is cheap and deadlock-free: nothing ever takes our lock first).
        with self._lock:
            seq = self._seq.next()
            ts = datetime.now(UTC).isoformat()
            entry = LogEntry(seq=seq, ts=ts, stream=stream, text=text)
            self._entries.append(entry)
            overflow = len(self._entries) - self.max_lines
            if overflow > 0:
                del self._entries[:overflow]
                self._dropped += overflow
            return entry

    def extend(self, stream: str, text: str) -> list[LogEntry]:
        """Append every line of a raw chunk; blank lines are dropped."""
        return [self.append(stream, line) for line in text.splitlines() if line]

    def after(self, cursor: int, limit: int = 500) -> tuple[list[LogEntry], bool, int]:
        """Entries with ``seq >= cursor``, capped at ``limit``.

        Returns ``(entries, gap, dropped)`` — all read under a single
        lock so a flushed frame is internally consistent (M3): the
        ``dropped`` count belongs to the same ring state as the entries,
        never a later eviction. ``gap`` is True when ``cursor`` points
        before the oldest retained entry (lines were evicted).
        """
        with self._lock:
            first = self._entries[0].seq if self._entries else self._seq.value
            gap = cursor < first
            entries = [e for e in self._entries if e.seq >= cursor]
            return entries[:limit], gap, self._dropped

    def tail(self, lines: int) -> list[LogEntry]:
        if lines <= 0:
            return []
        with self._lock:
            return list(self._entries[-lines:])

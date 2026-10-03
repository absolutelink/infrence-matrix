"""Cursor-addressable log rings for reliable streaming.

Each log feed (llama-server instance, halogen instance, benchmark run) gets a
ring of lines carrying a monotonic sequence number. Live events and polled
history share the same cursors so consumers can merge without content-based
matching and can detect both gaps (evicted lines) and reconnect holes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LogEntry:
    seq: int
    stream: str
    line: str

    def to_dict(self) -> dict:
        return asdict(self)


class CursorLogRing:
    """Ring buffer of log lines with a monotonic per-feed cursor.

    A cursor is the absolute sequence number of the next unread line.
    ``after(cursor)`` returns lines with ``seq >= cursor``. When lines are
    evicted, callers detect the hole via the ``gap`` flag and fall back to
    a tail bootstrap.
    """

    def __init__(self, max_lines: int = 2000) -> None:
        self.max_lines = max_lines
        self._entries: list[LogEntry] = []
        self._next_seq: int = 0

    @property
    def first_cursor(self) -> int:
        """Cursor of the oldest retained line (== last_cursor when empty)."""
        return self._entries[0].seq if self._entries else self._next_seq

    @property
    def last_cursor(self) -> int:
        """Cursor just past the most recently appended line."""
        return self._next_seq

    def __len__(self) -> int:
        return len(self._entries)

    def append(self, stream: str, line: str) -> LogEntry:
        entry = LogEntry(seq=self._next_seq, stream=stream, line=line)
        self._next_seq += 1
        self._entries.append(entry)
        if len(self._entries) > self.max_lines:
            del self._entries[: len(self._entries) - self.max_lines]
        return entry

    def extend(self, stream: str, text: str) -> list[LogEntry]:
        """Append every line of a raw chunk; blank lines are dropped."""
        return [self.append(stream, line) for line in text.splitlines() if line]

    def after(self, cursor: int, limit: int = 500) -> tuple[list[LogEntry], bool]:
        """Entries with ``seq >= cursor``, capped at ``limit``.

        Returns ``(entries, gap)`` where ``gap`` is True when ``cursor``
        points before the oldest retained entry (lines were evicted).
        """
        gap = cursor < self.first_cursor
        entries = [e for e in self._entries if e.seq >= cursor]
        return entries[:limit], gap

    def tail(self, lines: int) -> list[LogEntry]:
        if lines <= 0:
            return []
        return self._entries[-lines:]

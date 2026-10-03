"""Tests for the cursor-addressable log ring."""

from app.services.log_buffers import CursorLogRing


def test_append_and_tail_are_sequential() -> None:
    ring = CursorLogRing(max_lines=10)
    first = ring.append("stdout", "a")
    second = ring.append("stderr", "b")

    assert (first.seq, second.seq) == (0, 1)
    assert ring.last_cursor == 2
    assert [e.stream for e in ring.tail(10)] == ["stdout", "stderr"]


def test_after_returns_incremental_entries() -> None:
    ring = CursorLogRing(max_lines=10)
    for i in range(5):
        ring.append("stdout", f"line {i}")

    entries, gap = ring.after(3)
    assert gap is False
    assert [e.line for e in entries] == ["line 3", "line 4"]

    empty, gap = ring.after(ring.last_cursor)
    assert empty == []
    assert gap is False


def test_after_reports_gap_when_lines_evicted() -> None:
    ring = CursorLogRing(max_lines=3)
    for i in range(5):
        ring.append("stdout", f"line {i}")

    entries, gap = ring.after(0)
    assert gap is True
    assert [e.line for e in entries] == ["line 2", "line 3", "line 4"]
    assert ring.first_cursor == 2


def test_after_respects_limit() -> None:
    ring = CursorLogRing(max_lines=100)
    for i in range(50):
        ring.append("stdout", f"line {i}")

    entries, _ = ring.after(0, limit=10)
    assert len(entries) == 10
    assert entries[0].line == "line 0"


def test_extend_splits_chunks_and_drops_blanks() -> None:
    ring = CursorLogRing(max_lines=10)
    entries = ring.extend("stdout", "one\ntwo\n\n")
    assert [e.line for e in entries] == ["one", "two"]
    assert ring.last_cursor == 2

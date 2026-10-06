"""CursorLogRing tests: seq addressing, ts, drop counting, after/tail."""

import threading

from provider_lib.log_ring import CursorLogRing, SeqCounter


def test_append_assigns_monotonic_seq_and_ts() -> None:
    ring = CursorLogRing(max_lines=10)
    e0 = ring.append("stdout", "a")
    e1 = ring.append("stderr", "b")
    assert (e0.seq, e1.seq) == (0, 1)
    assert e0.stream == "stdout" and e1.stream == "stderr"
    assert e0.text == "a"
    # ISO-8601 UTC timestamp parseable.
    from datetime import datetime

    assert datetime.fromisoformat(e0.ts).tzinfo is not None
    assert ring.last_cursor == 2
    assert ring.first_cursor == 0
    assert ring.dropped == 0


def test_after_returns_entries_and_gap() -> None:
    ring = CursorLogRing(max_lines=10)
    for i in range(5):
        ring.append("stdout", f"l{i}")
    entries, gap, _dropped = ring.after(2, limit=100)
    assert [e.seq for e in entries] == [2, 3, 4]
    assert gap is False
    entries, gap, _dropped = ring.after(0, limit=3)
    assert [e.seq for e in entries] == [0, 1, 2]
    assert gap is False


def test_eviction_increments_dropped_and_moves_first_cursor() -> None:
    ring = CursorLogRing(max_lines=3)
    for i in range(5):
        ring.append("stdout", f"l{i}")
    assert len(ring) == 3
    assert ring.dropped == 2
    assert ring.first_cursor == 2
    assert ring.last_cursor == 5
    entries, gap, _dropped = ring.after(1, limit=10)
    assert gap is True  # seq 1 was evicted
    assert [e.seq for e in entries] == [2, 3, 4]


def test_after_beyond_end_is_empty_no_gap() -> None:
    ring = CursorLogRing(max_lines=3)
    ring.append("stdout", "x")
    entries, gap, _dropped = ring.after(1, limit=10)
    assert entries == []
    assert gap is False


def test_tail_and_extend() -> None:
    ring = CursorLogRing(max_lines=10)
    entries = ring.extend("stderr", "a\n\nb\nc")
    assert [e.text for e in entries] == ["a", "b", "c"]
    assert all(e.stream == "stderr" for e in entries)
    assert [e.text for e in ring.tail(2)] == ["b", "c"]
    assert ring.tail(0) == []
    assert ring.tail(-1) == []


def test_empty_ring_cursors() -> None:
    ring = CursorLogRing(max_lines=4)
    assert ring.first_cursor == 0
    assert ring.last_cursor == 0
    entries, gap, _dropped = ring.after(0, limit=10)
    assert entries == []
    assert gap is False


def test_concurrent_appends_keep_seq_unique() -> None:
    ring = CursorLogRing(max_lines=1000)
    n_threads = 8
    per = 50

    def worker() -> None:
        for i in range(per):
            ring.append("stdout", f"t{i}")

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seqs = [e.seq for e in ring.tail(len(ring))]
    assert len(seqs) == n_threads * per
    assert len(set(seqs)) == n_threads * per
    assert ring.last_cursor == n_threads * per


def test_shared_seq_counter_unifies_two_rings() -> None:
    counter = SeqCounter()
    backend = CursorLogRing(max_lines=10, seq_source=counter)
    provider = CursorLogRing(max_lines=10, seq_source=counter)
    b0 = backend.append("stdout", "b0")
    p0 = provider.append("stdout", "p0")
    b1 = backend.append("stderr", "b1")
    p1 = provider.append("stdout", "p1")
    # Interleaved appends draw from ONE space: strictly increasing, unique.
    seqs = [e.seq for e in (b0, p0, b1, p1)]
    assert seqs == sorted(seqs) and len(set(seqs)) == 4
    # Each ring's cursors reflect the shared space.
    assert backend.last_cursor == provider.last_cursor == 4
    assert counter.value == 4


def test_after_returns_dropped_atomically() -> None:
    ring = CursorLogRing(max_lines=2)
    for i in range(4):
        ring.append("stdout", f"l{i}")
    entries, gap, dropped = ring.after(0, limit=10)
    assert dropped == 2
    assert [e.seq for e in entries] == [2, 3]
    assert gap is True

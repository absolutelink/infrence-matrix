"""Tests for cross-thread event bus delivery."""

import asyncio
import threading
import time

import pytest

from app.services.event_bus import publish_event, subscribe, unsubscribe


@pytest.mark.asyncio
async def test_publish_from_worker_thread_wakes_subscriber() -> None:
    q = subscribe()
    arrivals = {}

    async def consumer() -> None:
        while True:
            ev = await q.get()
            arrivals[ev["data"]["i"]] = time.monotonic()

    c = asyncio.create_task(consumer())
    await asyncio.sleep(0.2)  # let the consumer park in get()

    put_times = {}
    n = 5

    def worker() -> None:
        for i in range(n):
            put_times[i] = time.monotonic()
            publish_event("download.progress", {"i": i})
            time.sleep(0.1)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    await asyncio.sleep(1.2)  # ONLY scheduled wakeup is this timer
    th.join(timeout=3)
    c.cancel()
    unsubscribe(q)

    assert len(arrivals) == n, f"only {len(arrivals)}/{n} events delivered"
    for i in range(n):
        lat = arrivals[i] - put_times[i]
        assert lat < 0.3, (
            f"event {i} starved: {lat:.2f}s (foreign-thread put did not wake the loop)"
        )


@pytest.mark.asyncio
async def test_publish_on_loop_still_delivers() -> None:
    q = subscribe()
    try:
        publish_event("server.started", {"server_id": "abc"})
        ev = await asyncio.wait_for(q.get(), timeout=1.0)
        assert ev["event"] == "server.started"
        assert ev["data"] == {"server_id": "abc"}
    finally:
        unsubscribe(q)

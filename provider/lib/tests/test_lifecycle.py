"""BackendLifecycle state machine + slot admission tests (fake driver)."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from provider_lib.backend import (
    BackendBusy,
    BackendDriver,
    BackendLifecycle,
    BackendNotReady,
)
from provider_lib.wire import BackendStatusValue


class FakeDriver(BackendDriver):
    def __init__(
        self,
        *,
        start_raises: bool = False,
        healthy: bool = True,
        stop_raises: bool = False,
    ) -> None:
        self.start_raises = start_raises
        self.healthy = healthy
        self.stop_raises = stop_raises
        self.start_calls = 0
        self.stop_calls = 0
        self.events: list[dict[str, Any]] = []

    async def start(self) -> None:
        self.start_calls += 1
        if self.start_raises:
            raise RuntimeError("boom")

    async def stop(self) -> None:
        self.stop_calls += 1
        if self.stop_raises:
            raise RuntimeError("stop boom")

    async def health(self) -> bool:
        return self.healthy

    async def list_models(self) -> list[dict[str, Any]]:
        return [{"id": "fake-model", "object": "model"}]

    def stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        async def gen() -> AsyncIterator[dict[str, Any]]:
            yield {"type": "response.created"}
            yield {"type": "response.completed", "usage": {"total_tokens": 1}}

        return gen()


def _collect_lifecycle(
    driver: FakeDriver | None = None, **kwargs: Any
) -> tuple[BackendLifecycle, list[tuple[str, str | None]]]:
    emitted: list[tuple[str, str | None]] = []

    async def cb(status: str, reason: str | None) -> None:
        emitted.append((status, reason))

    return (
        BackendLifecycle(driver or FakeDriver(**kwargs), status_callback=cb),
        emitted,
    )


async def test_start_transitions_to_running() -> None:
    lifecycle, emitted = _collect_lifecycle()
    assert lifecycle.backend_status == BackendStatusValue.STOPPED
    await lifecycle.start()
    assert lifecycle.backend_status == BackendStatusValue.RUNNING
    assert [s for s, _ in emitted] == [
        BackendStatusValue.STARTING,
        BackendStatusValue.RUNNING,
    ]


async def test_start_is_idempotent_when_running() -> None:
    driver = FakeDriver()
    lifecycle, _ = _collect_lifecycle(driver)
    await lifecycle.start()
    await lifecycle.start()
    await lifecycle.ensure_started()
    assert driver.start_calls == 1
    assert lifecycle.backend_status == BackendStatusValue.RUNNING


async def test_start_failure_emits_error() -> None:
    lifecycle, emitted = _collect_lifecycle(start_raises=True)
    with pytest.raises(RuntimeError, match="boom"):
        await lifecycle.start()
    assert lifecycle.backend_status == BackendStatusValue.ERROR
    assert emitted[-1][0] == BackendStatusValue.ERROR


async def test_unhealthy_after_start_emits_error() -> None:
    lifecycle, emitted = _collect_lifecycle(healthy=False)
    with pytest.raises(RuntimeError, match="health"):
        await lifecycle.start()
    assert lifecycle.backend_status == BackendStatusValue.ERROR


async def test_stop_path() -> None:
    lifecycle, emitted = _collect_lifecycle()
    await lifecycle.start()
    emitted.clear()
    await lifecycle.stop()
    assert lifecycle.backend_status == BackendStatusValue.STOPPED
    assert [s for s, _ in emitted] == [
        BackendStatusValue.STOPPING,
        BackendStatusValue.STOPPED,
    ]
    # stop from STOPPED is a no-op
    await lifecycle.stop()
    assert [s for s, _ in emitted] == [
        BackendStatusValue.STOPPING,
        BackendStatusValue.STOPPED,
    ]


async def test_stop_failure_emits_error() -> None:
    lifecycle, emitted = _collect_lifecycle(stop_raises=True)
    await lifecycle.start()
    emitted.clear()
    with pytest.raises(RuntimeError, match="stop boom"):
        await lifecycle.stop()
    assert lifecycle.backend_status == BackendStatusValue.ERROR


async def test_acquire_requires_running() -> None:
    lifecycle, _ = _collect_lifecycle()
    with pytest.raises(BackendNotReady):
        await lifecycle.acquire_slot()
    await lifecycle.stop()  # STOPPED -> no-op
    with pytest.raises(BackendNotReady):
        await lifecycle.acquire_slot()


async def test_slot_context_manager_in_use_running_cycle() -> None:
    lifecycle, emitted = _collect_lifecycle()
    await lifecycle.start()
    emitted.clear()
    assert lifecycle.in_flight == 0
    async with lifecycle.slot():
        assert lifecycle.in_flight == 1
        assert lifecycle.backend_status == BackendStatusValue.IN_USE
        assert lifecycle.last_request_at is not None
    assert lifecycle.in_flight == 0
    assert lifecycle.backend_status == BackendStatusValue.RUNNING
    assert [s for s, _ in emitted] == [
        BackendStatusValue.IN_USE,
        BackendStatusValue.RUNNING,
    ]


async def test_capacity_backend_busy() -> None:
    lifecycle, _ = _collect_lifecycle()
    lifecycle.capacity = 2
    await lifecycle.start()
    await lifecycle.acquire_slot()
    await lifecycle.acquire_slot()
    with pytest.raises(BackendBusy):
        await lifecycle.acquire_slot()
    await lifecycle.release_slot()
    await lifecycle.acquire_slot()  # room again
    assert lifecycle.in_flight == 2


async def test_stop_if_idle_refuses_while_slot_held() -> None:
    """SF-1: stop_if_idle raises BackendBusy when a slot is held and the
    backend is NOT stopped."""
    driver = FakeDriver()
    lifecycle, _ = _collect_lifecycle(driver)
    await lifecycle.start()
    await lifecycle.acquire_slot()  # RUNNING -> IN_USE
    with pytest.raises(BackendBusy) as exc_info:
        await lifecycle.stop_if_idle()
    assert exc_info.value.in_flight == 1
    assert lifecycle.backend_status == BackendStatusValue.IN_USE
    assert driver.stop_calls == 0
    await lifecycle.release_slot()
    assert lifecycle.backend_status == BackendStatusValue.RUNNING
    # Now idle -> stops cleanly.
    await lifecycle.stop_if_idle()
    assert lifecycle.backend_status == BackendStatusValue.STOPPED
    assert driver.stop_calls == 1


async def test_stop_if_idle_stops_when_idle() -> None:
    driver = FakeDriver()
    lifecycle, _ = _collect_lifecycle(driver)
    await lifecycle.start()
    await lifecycle.stop_if_idle()
    assert lifecycle.backend_status == BackendStatusValue.STOPPED
    # Already stopped -> no-op (no extra driver.stop call).
    await lifecycle.stop_if_idle()
    assert driver.stop_calls == 1


async def test_stop_if_idle_atomic_against_concurrent_acquire() -> None:
    """SF-1: with capacity 1 and a running backend, a concurrent
    acquire_slot and stop_if_idle never both succeed-and-kill: exactly one
    of them wins the lock first, and the loser sees the new state."""
    driver = FakeDriver()
    lifecycle, _ = _collect_lifecycle(driver)
    await lifecycle.start()

    acquire_failed = 0
    stop_refused = 0

    async def racer_acquire() -> None:
        nonlocal acquire_failed
        try:
            await lifecycle.acquire_slot()
        except BackendBusy, BackendNotReady:
            acquire_failed += 1

    async def racer_stop() -> None:
        nonlocal stop_refused
        try:
            await lifecycle.stop_if_idle()
        except BackendBusy:
            stop_refused += 1

    await asyncio.gather(racer_acquire(), racer_stop())
    # If the acquire won, the stop must have been refused (busy), and the
    # driver was never stopped under a live slot. If the stop won, the
    # acquire must have failed (not serving).
    assert (acquire_failed, stop_refused) in {(0, 1), (1, 0)}
    if stop_refused:
        assert lifecycle.in_flight == 1
        assert driver.stop_calls == 0
    else:
        assert lifecycle.backend_status == BackendStatusValue.STOPPED


async def test_release_tolerates_extra_releases() -> None:
    lifecycle, emitted = _collect_lifecycle()
    await lifecycle.start()
    emitted.clear()
    await lifecycle.release_slot()  # nothing held
    assert lifecycle.in_flight == 0
    assert emitted == []


async def test_last_request_at_updates_per_acquire() -> None:
    lifecycle, _ = _collect_lifecycle()
    await lifecycle.start()
    first = None
    async with lifecycle.slot():
        first = lifecycle.last_request_at
    await asyncio.sleep(0.01)
    async with lifecycle.slot():
        assert lifecycle.last_request_at is not None
        assert lifecycle.last_request_at >= first  # type: ignore[operator]

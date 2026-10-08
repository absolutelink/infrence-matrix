"""provider_lib.ops: manual backend control + provider.initialize.

Covers the two contracts the admin UI depends on:

* `wait_for_running: false` acks "accepted" and boots in the BACKGROUND
  (a cold engine download must not hold a WS ack window or an HTTP
  request), while the default (true) keeps the scheduler's
  "acked == /v1 is live" contract.
* `provider.initialize` re-registers first, then reboots and publishes
  `backend.metadata` (stamped with the owning `instance_id`).

Uses a gated fake driver so "in flight" is deterministic.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from config_fakes import RecordingClient

from provider_lib.backend import BackendDriver, BackendLifecycle
from provider_lib.config import ProviderSettings
from provider_lib.ops import emit_backend_status_snapshot, install_backend_ops
from provider_lib.wire import BackendStatusValue, InstanceStatusValue


class GatedDriver(BackendDriver):
    """Driver whose start() blocks on `gate`, with call counters."""

    def __init__(self, *, gate: asyncio.Event | None = None) -> None:
        self.gate = gate
        self.start_calls = 0
        self.stop_calls = 0
        self.started = False
        self.models: list[dict[str, Any]] = [{"id": "gated", "object": "model"}]
        self.list_models_raises = False
        self.applied: list[dict[str, Any]] = []

    async def start(self) -> None:
        self.start_calls += 1
        if self.gate is not None:
            await self.gate.wait()
        self.started = True

    async def stop(self) -> None:
        self.stop_calls += 1
        self.started = False

    async def health(self) -> bool:
        return self.started

    async def list_models(self) -> list[dict[str, Any]]:
        if self.list_models_raises:
            raise RuntimeError("metadata scrape blew up")
        return list(self.models)

    def stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        async def gen() -> AsyncIterator[dict[str, Any]]:
            yield {"type": "response.completed"}

        return gen()

    def apply_config(self, backend_config: dict[str, Any]) -> None:
        self.applied.append(dict(backend_config))


def _setup(
    *,
    gate: asyncio.Event | None = None,
    re_register: Any = None,
    instance_id: str | None = None,
) -> tuple[RecordingClient, BackendLifecycle, GatedDriver, Any]:
    client = RecordingClient(ProviderSettings())
    driver = GatedDriver(gate=gate)
    lifecycle = BackendLifecycle(driver, capacity=2, instance_id=instance_id)
    ops = install_backend_ops(
        client, lifecycle, backend_name="gated", re_register=re_register
    )
    return client, lifecycle, driver, ops


async def test_default_start_blocks_until_running_and_publishes_metadata() -> None:
    """Scheduler contract: the ack means /v1 is live (unchanged)."""
    client, lifecycle, driver, ops = _setup()
    ack = await client.dispatch("backend.start", {})
    assert ack["ok"] is True
    assert driver.start_calls == 1
    assert lifecycle.backend_status == BackendStatusValue.RUNNING
    assert ack["detail"]["waited_for_running"] is True
    assert ack["detail"]["capacity"] == 2
    assert ("backend.metadata", {"models": [{"id": "gated", "object": "model"}]}) in (
        client.events
    )
    assert ops.busy is False


async def test_accept_style_start_acks_immediately_and_boots_in_background() -> None:
    """`wait_for_running: false` — the manual button's contract.

    The ack must return BEFORE the engine is up (that is the whole point:
    a download-bound boot cannot hold the socket open), and the terminal
    state must arrive as events.
    """
    gate = asyncio.Event()
    client, lifecycle, driver, ops = _setup(gate=gate)
    ack = await client.dispatch("backend.start", {"wait_for_running": False})
    assert ack["ok"] is True
    assert ack["detail"]["accepted"] is True
    assert lifecycle.backend_status != BackendStatusValue.RUNNING
    assert client.events == []  # nothing terminal emitted yet

    gate.set()
    await ops.join()
    assert lifecycle.backend_status == BackendStatusValue.RUNNING
    types = [t for t, _ in client.events]
    assert "backend.metadata" in types
    assert (
        "provider.status",
        {
            "instance_status": InstanceStatusValue.RUNNING,
            "backend_status": BackendStatusValue.RUNNING,
        },
    ) in client.events


async def test_background_failure_reports_error_status() -> None:
    """An accepted boot that later dies must not fail silently: the ack is
    already gone, so the outcome travels as provider.status error."""
    client, _lifecycle, driver, ops = _setup()

    async def exploding_start() -> None:
        driver.start_calls += 1
        raise RuntimeError("engine exited during download")

    driver.start = exploding_start  # type: ignore[method-assign]
    ack = await client.dispatch("backend.start", {"wait_for_running": False})
    assert ack["ok"] is True
    await ops.join()
    errors = [(t, p) for t, p in client.events if t == "provider.status"]
    assert errors, client.events
    assert errors[-1][1]["instance_status"] == InstanceStatusValue.ERROR
    assert "engine exited during download" in errors[-1][1]["error_message"]


async def test_second_accept_style_start_is_refused_not_duplicated() -> None:
    gate = asyncio.Event()
    client, _lifecycle, driver, ops = _setup(gate=gate)
    first = await client.dispatch("backend.start", {"wait_for_running": False})
    assert first["ok"] is True
    await asyncio.sleep(0)  # let the spawned boot reach driver.start()
    assert driver.start_calls == 1
    second = await client.dispatch("backend.start", {"wait_for_running": False})
    assert second["ok"] is False
    assert second["error"] == "boot_in_progress"
    assert second["detail"]["retry_after"]
    assert driver.start_calls == 1  # no redundant spawn

    gate.set()
    await ops.join()


async def test_waiting_start_joins_an_in_flight_boot() -> None:
    """Scheduler arrives while the operator's download boot is running:
    it must ride that boot, not fail and not double-spawn."""
    gate = asyncio.Event()
    client, lifecycle, driver, ops = _setup(gate=gate)
    accepted = await client.dispatch("backend.start", {"wait_for_running": False})
    assert accepted["ok"] is True

    waiter = asyncio.create_task(client.dispatch("backend.start", {}))
    await asyncio.sleep(0)  # let the waiter reach the join
    assert not waiter.done()
    gate.set()
    ack = await waiter
    assert ack["ok"] is True
    assert driver.start_calls == 1
    assert lifecycle.backend_status == BackendStatusValue.RUNNING


async def test_backend_metadata_stamps_instance_id() -> None:
    """H1: backend.metadata carries the owning instance_id so the admin keys
    it to the right backend (frames without it are dropped)."""
    client, _lifecycle, _driver, _ops = _setup(instance_id="inst-42")
    await client.dispatch("backend.start", {"instance_id": "inst-42"})
    assert (
        "backend.metadata",
        {"models": [{"id": "gated", "object": "model"}], "instance_id": "inst-42"},
    ) in client.events


async def test_multi_backend_dispatch_routes_by_instance_id() -> None:
    """H3: a 2-backend agent boots the CORRECT backend by instance_id — a
    start for backend #2 must not boot backend #1."""
    from provider_lib.registry import BackendHandle, BackendRegistry

    client = RecordingClient(ProviderSettings())
    d1, d2 = GatedDriver(), GatedDriver()
    lc1 = BackendLifecycle(d1, capacity=1, instance_id="inst-1")
    lc2 = BackendLifecycle(d2, capacity=1, instance_id="inst-2")
    registry = BackendRegistry()
    registry.add(BackendHandle("inst-1", lc1))
    registry.add(BackendHandle("inst-2", lc2))
    install_backend_ops(client, registry, backend_name="gated")

    ack = await client.dispatch("backend.start", {"instance_id": "inst-2"})
    assert ack["ok"] is True
    assert d2.start_calls == 1
    assert d1.start_calls == 0  # the other backend was NOT booted

    # An id this agent does not host is NAK'd, not silently mis-served.
    ack = await client.dispatch("backend.start", {"instance_id": "inst-999"})
    assert ack["ok"] is False
    assert ack["error"] == "unknown_instance"

    # A multi-backend agent cannot guess an id-less command.
    ack = await client.dispatch("backend.stop", {})
    assert ack["ok"] is False
    assert ack["error"] == "unknown_instance"


async def test_single_handle_rejects_foreign_instance_id() -> None:
    """H3 hardening: a single-backend agent must NOT boot its one backend for
    a command naming a DIFFERENT (foreign) instance_id — that would silently
    mis-serve. Absent id still passes through; a matching id serves."""
    from provider_lib.registry import BackendHandle, BackendRegistry

    client = RecordingClient(ProviderSettings())
    driver = GatedDriver()
    lc = BackendLifecycle(driver, capacity=1, instance_id="mine")
    registry = BackendRegistry()
    registry.add(BackendHandle("", lc))  # key predates registration ("")
    install_backend_ops(client, registry, backend_name="gated")

    # Foreign id -> NAK, backend NOT booted.
    ack = await client.dispatch("backend.start", {"instance_id": "someone-else"})
    assert ack["ok"] is False
    assert ack["error"] == "unknown_instance"
    assert driver.start_calls == 0

    # The agent's own id (matched via the live lifecycle.instance_id) -> serves.
    ack = await client.dispatch("backend.start", {"instance_id": "mine"})
    assert ack["ok"] is True
    assert driver.start_calls == 1

    # No id at all -> single-backend convenience serves.
    await client.dispatch("backend.stop", {})
    ack = await client.dispatch("backend.start", {})
    assert ack["ok"] is True


async def test_stop_is_awaited_and_confirms_stopped() -> None:
    client, lifecycle, driver, _ops = _setup()
    await client.dispatch("backend.start", {})
    ack = await client.dispatch("backend.stop", {})
    assert ack["ok"] is True
    assert ack["detail"]["backend_status"] == BackendStatusValue.STOPPED
    assert driver.stop_calls == 1


async def test_stop_refused_while_a_background_boot_runs() -> None:
    gate = asyncio.Event()
    client, _lifecycle, _driver, ops = _setup(gate=gate)
    await client.dispatch("backend.start", {"wait_for_running": False})
    ack = await client.dispatch("backend.stop", {})
    assert ack["ok"] is False
    assert ack["error"] == "boot_in_progress"
    gate.set()
    await ops.join()


async def test_restart_drains_then_reboots() -> None:
    client, lifecycle, driver, ops = _setup()
    await client.dispatch("backend.start", {})
    await lifecycle.acquire_slot()  # live request holds a slot
    ack = await client.dispatch("backend.restart", {"wait_for_running": False})
    assert ack["ok"] is False
    assert ack["error"] == "backend_in_use"
    assert ack["detail"]["in_flight"] == 1
    await lifecycle.release_slot()
    ack = await client.dispatch("backend.restart", {"wait_for_running": False})
    assert ack["ok"] is True and ack["detail"]["accepted"] is True
    await ops.join()
    assert driver.start_calls == 2  # original boot + restart boot


async def test_restart_blocking_path() -> None:
    """`wait_for_running: true` restart stops, boots, and only then acks."""
    client, lifecycle, driver, _ops = _setup()
    await client.dispatch("backend.start", {})
    ack = await client.dispatch("backend.restart", {"wait_for_running": True})
    assert ack["ok"] is True
    assert ack["detail"]["waited_for_running"] is True
    assert driver.stop_calls == 1
    assert driver.start_calls == 2
    assert lifecycle.backend_status == BackendStatusValue.RUNNING


async def test_initialize_re_registers_then_reboots_and_publishes_metadata() -> None:
    calls: list[str] = []

    async def re_register() -> dict[str, Any]:
        calls.append("re-register")
        return {"alias": "rocinante", "backend_config": {"context": {"ctx": 32}}}

    client, lifecycle, driver, ops = _setup(re_register=re_register)
    await client.dispatch("backend.start", {})
    calls_order_before = list(calls)
    assert calls_order_before == []

    ack = await client.dispatch("provider.initialize", {})
    assert ack["ok"] is True
    assert ack["detail"]["accepted"] is True
    assert calls == ["re-register"]
    await ops.join()
    assert driver.start_calls == 2  # first boot + the initialize reboot
    assert driver.stop_calls == 1
    assert lifecycle.backend_status == BackendStatusValue.RUNNING
    assert ("backend.metadata", {"models": [{"id": "gated", "object": "model"}]}) in (
        client.events
    )


async def test_initialize_re_registers_and_boots() -> None:
    """provider.initialize re-registers then boots in the background (the
    Phase 14 shell/no-config fence is gone — a definition always has a
    config now)."""

    async def re_register() -> dict[str, Any] | None:
        return {"backend_config": {"context": {"ctx": 8}}}

    client, _lifecycle, driver, ops = _setup(re_register=re_register)
    ack = await client.dispatch("provider.initialize", {})
    assert ack["ok"] is True
    assert ack["detail"]["accepted"] is True
    await ops.join()
    assert driver.start_calls == 1  # the initialize boot ran


async def test_initialize_re_register_failure_is_a_nak() -> None:
    async def re_register() -> dict[str, Any]:
        raise RuntimeError("409 version mismatch")

    client, _lifecycle, driver, _ops = _setup(re_register=re_register)
    ack = await client.dispatch("provider.initialize", {})
    assert ack["ok"] is False
    assert "409 version mismatch" in ack["error"]
    assert ack["detail"]["step"] == "initialize"
    assert driver.start_calls == 0


async def test_initialize_refuses_while_requests_are_live() -> None:
    async def re_register() -> dict[str, Any]:
        return {"backend_config": {}}

    client, lifecycle, _driver, _ops = _setup(re_register=re_register)
    await client.dispatch("backend.start", {})
    await lifecycle.acquire_slot()
    ack = await client.dispatch("provider.initialize", {})
    assert ack["ok"] is False
    assert ack["error"] == "backend_in_use"


async def test_metadata_scrape_failure_does_not_fail_the_boot() -> None:
    client, lifecycle, driver, _ops = _setup()
    driver.list_models_raises = True
    ack = await client.dispatch("backend.start", {})
    assert ack["ok"] is True
    assert lifecycle.backend_status == BackendStatusValue.RUNNING
    assert not [e for e in client.events if e[0] == "backend.metadata"]


async def test_start_detail_carries_driver_ports() -> None:
    """The shared ack builder echoes whatever the driver exposes, so a
    provider needs no local handler to keep its port info in the UI."""
    client, _lifecycle, driver, _ops = _setup()
    driver.api_port = 8288  # type: ignore[attr-defined]
    driver.engine_port = 8289  # type: ignore[attr-defined]
    ack = await client.dispatch("backend.start", {})
    assert ack["detail"]["api_port"] == 8288
    assert ack["detail"]["engine_port"] == 8289
    assert "backend_port" not in ack["detail"]


# ---------------------------------------------------------------------------
# Connect-time backend.status snapshot (Fix A: reconcile the admin mirror)
# ---------------------------------------------------------------------------
async def test_emit_backend_status_snapshot_one_frame_per_handle() -> None:
    """A restarted agent must re-announce every hosted backend's CURRENT
    status so the admin heals its DB mirror (the lifecycle only emits
    backend.status on transitions, so nothing else would)."""
    from provider_lib.registry import BackendHandle, BackendRegistry

    client = RecordingClient(ProviderSettings())
    stopped = BackendLifecycle(GatedDriver(), capacity=1, instance_id="inst-stopped")
    running_driver = GatedDriver()
    running = BackendLifecycle(running_driver, capacity=1, instance_id="inst-running")
    await running.start()  # STOPPED -> RUNNING
    assert running.backend_status == BackendStatusValue.RUNNING

    registry = BackendRegistry()
    registry.add(BackendHandle("inst-stopped", stopped))
    registry.add(BackendHandle("inst-running", running))

    await emit_backend_status_snapshot(client, registry)

    frames = [p for t, p in client.events if t == "backend.status"]
    assert len(frames) == 2
    by_id = {f["instance_id"]: f for f in frames}
    assert by_id["inst-stopped"]["backend_status"] == BackendStatusValue.STOPPED
    assert by_id["inst-running"]["backend_status"] == BackendStatusValue.RUNNING
    assert all(f["reason"] == "connect snapshot" for f in frames)


async def test_emit_backend_status_snapshot_skips_placeholder_instance_id() -> None:
    """A handle whose key predates registration (empty) and whose lifecycle has
    no instance_id yet is skipped — the admin drops frames without one."""
    from provider_lib.registry import BackendHandle, BackendRegistry

    client = RecordingClient(ProviderSettings())
    placeholder = BackendLifecycle(GatedDriver(), capacity=1, instance_id=None)
    real = BackendLifecycle(GatedDriver(), capacity=1, instance_id="inst-real")
    registry = BackendRegistry()
    registry.add(BackendHandle("", placeholder))
    registry.add(BackendHandle("inst-real", real))

    await emit_backend_status_snapshot(client, registry)

    frames = [p for t, p in client.events if t == "backend.status"]
    assert [f["instance_id"] for f in frames] == ["inst-real"]


async def test_emit_backend_status_snapshot_accepts_bare_lifecycle() -> None:
    """Single-backend convenience: a bare BackendLifecycle is wrapped via
    registry_from_lifecycle (mirrors install_backend_ops)."""
    client = RecordingClient(ProviderSettings())
    lifecycle = BackendLifecycle(GatedDriver(), capacity=1, instance_id="solo")

    await emit_backend_status_snapshot(client, lifecycle)

    assert client.events == [
        (
            "backend.status",
            {
                "instance_id": "solo",
                "backend_status": BackendStatusValue.STOPPED,
                "reason": "connect snapshot",
            },
        )
    ]

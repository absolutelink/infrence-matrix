"""provider_lib.ops: manual backend control + provider.initialize.

Covers the two contracts the admin UI depends on:

* `wait_for_running: false` acks "accepted" and boots in the BACKGROUND
  (a cold engine download must not hold a WS ack window or an HTTP
  request), while the default (true) keeps the scheduler's
  "acked == /v1 is live" contract.
* `provider.initialize` re-registers first, then reboots and publishes
  `backend.metadata`; it never boots an unconfigured (shell) definition.

Uses a gated fake driver so "in flight" is deterministic.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from config_fakes import RecordingClient

from provider_lib.backend import BackendDriver, BackendLifecycle
from provider_lib.config import ProviderSettings
from provider_lib.ops import install_backend_ops
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
    no_config: bool = False,
) -> tuple[RecordingClient, BackendLifecycle, GatedDriver, Any]:
    client = RecordingClient(ProviderSettings())
    if no_config:

        async def _nak(_frame: Any) -> dict[str, Any] | None:
            return {
                "ok": False,
                "error": "no_config",
                "detail": {"step": "validate"},
            }

        client.no_config_nak = _nak
    driver = GatedDriver(gate=gate)
    lifecycle = BackendLifecycle(driver, capacity=2)
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


async def test_start_refused_for_unconfigured_shell_definition() -> None:
    """Phase 14 fence lives in config_update; ops must honor it."""
    client, _lifecycle, driver, _ops = _setup(no_config=True)
    for payload in ({}, {"wait_for_running": False}):
        ack = await client.dispatch("backend.start", payload)
        assert ack["ok"] is False
        assert ack["error"] == "no_config"
    assert driver.start_calls == 0


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


async def test_initialize_without_config_refreshes_but_never_boots() -> None:
    """A shell definition may be refreshed (that is how it re-reads the
    admin's row) but must NOT boot on defaults."""

    async def re_register() -> dict[str, Any] | None:
        return None  # still unconfigured

    client, _lifecycle, driver, ops = _setup(re_register=re_register, no_config=True)
    ack = await client.dispatch("provider.initialize", {})
    assert ack["ok"] is True
    assert ack["detail"]["no_config"] is True
    await ops.join()
    assert driver.start_calls == 0


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

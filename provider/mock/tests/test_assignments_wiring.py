"""Phase 16 slice 5 mock wiring: N-backend assignments reconcile + per-backend
config addressing.

The mock hosts a ``BackendRegistry`` of several lifecycles, so it exercises the
two slice-5 provider contracts end to end against the real mock driver:

* ``agent.assignments.update`` adds a newly-placed backend and retires a
  dropped one (busy-safe), demonstrating the reconcile the admin drives;
* ``provider.config.update`` addressed by ``instance_id`` reaches the CORRECT
  lifecycle on a 2-backend agent (the ``resolve_target`` path) — a config edit
  for backend #2 must not touch backend #1.
"""

import asyncio
from typing import Any

from provider_lib.admin_client import AdminClient
from provider_lib.config import ProviderSettings
from provider_lib.config_update import ConfigState
from provider_lib.registry import BackendHandle, BackendRegistry
from provider_lib.wire import BackendStatusValue, Frame

from provider_mock.main import install_command_handlers, make_lifecycle


def _settings(tmp_path: Any) -> ProviderSettings:
    return ProviderSettings(  # type: ignore[call-arg]
        MACHINE_UID="mock-asg",
        MACHINE_SECRET="tok",
        AGENT_ID="mock-asg-agent",
        ADMIN_BASE_URL="http://localhost:9999",
        CACHE_DIR=tmp_path / "cache",
        MODELS_DIR=tmp_path / "models",
    )


async def _dispatch(client: AdminClient, type_: str, payload: dict) -> dict[str, Any]:
    await client._dispatch(Frame(type=type_, id="cmd-1", epoch=1, payload=payload))
    for _ in range(80):
        frame = await asyncio.wait_for(client._outgoing.get(), timeout=5.0)
        if frame.type == "ack" and frame.reply_to == "cmd-1":
            return frame.payload
    raise AssertionError("no ack produced")


def _entry(iid: str, **over: Any) -> dict[str, Any]:
    entry = {
        "instance_id": iid,
        "provider_definition_id": f"def-{iid}",
        "alias": f"alias-{iid}",
        "backend_config": {"delta_count": 2},
        "config_fingerprint": f"fp-{iid}",
        "port": 8081,
        "capacity": 1,
        "idle_timeout_seconds": 300,
        "vram_required_bytes": 0,
    }
    entry.update(over)
    return entry


def _two_backend_registry(client: AdminClient) -> BackendRegistry:
    registry = BackendRegistry()
    for iid in ("i1", "i2"):
        lifecycle = make_lifecycle(client, instance_id=iid)
        registry.add(BackendHandle(iid, lifecycle, ConfigState("fp-seed")))
    return registry


def test_assignment_handler_installed_on_registry_agent() -> None:
    client = AdminClient(_settings(__import__("pathlib").Path("/tmp")))
    registry = _two_backend_registry(client)
    install_command_handlers(client, registry)
    assert "agent.assignments.update" in client._command_handlers


async def test_assignments_add_and_remove_on_mock(tmp_path: Any) -> None:
    client = AdminClient(_settings(tmp_path))
    registry = _two_backend_registry(client)
    install_command_handlers(client, registry)

    # Add a third backend, drop i1 (idle) -> i1 stopped+removed, i3 added.
    ack = await _dispatch(
        client,
        "agent.assignments.update",
        {"assignments": [_entry("i2"), _entry("i3")], "max_running_backends": 0},
    )
    assert ack["ok"] is True
    assert ack["added"] == ["i3"]
    assert ack["removed"] == ["i1"]
    assert "i1" not in registry
    assert "i2" in registry and "i3" in registry
    # The added backend's lifecycle is keyed to its instance id and adopted the
    # pushed config (alias -> model, backend_config applied).
    h3 = registry.get("i3")
    assert h3 is not None
    assert h3.lifecycle.instance_id == "i3"
    assert h3.lifecycle.driver.model == "alias-i3"
    assert h3.config_state.applied_fingerprint == "fp-i3"


async def test_assignments_busy_removal_refused_on_mock(tmp_path: Any) -> None:
    client = AdminClient(_settings(tmp_path))
    registry = _two_backend_registry(client)
    install_command_handlers(client, registry)
    h1 = registry.get("i1")
    assert h1 is not None
    await h1.lifecycle.start()
    await h1.lifecycle.acquire_slot()  # IN_USE

    ack = await _dispatch(
        client, "agent.assignments.update", {"assignments": [_entry("i2")]}
    )
    assert {"instance_id": "i1", "reason": "backend_in_use"} in ack["refused"]
    assert "i1" in registry
    assert h1.lifecycle.backend_status == BackendStatusValue.IN_USE
    await h1.lifecycle.release_slot()


async def test_config_update_targets_correct_backend_on_two_agent(
    tmp_path: Any,
) -> None:
    """Per-backend addressing: a provider.config.update for i2 applies to i2's
    driver only — i1's driver is untouched."""
    client = AdminClient(_settings(tmp_path))
    registry = _two_backend_registry(client)
    install_command_handlers(client, registry)
    h1 = registry.get("i1")
    h2 = registry.get("i2")
    assert h1 is not None and h2 is not None

    ack = await _dispatch(
        client,
        "provider.config.update",
        {
            "instance_id": "i2",
            "backend_config": {"delta_count": 9},
            "config_fingerprint": "fp-i2-new",
            "capacity": 1,
        },
    )
    assert ack["ok"] is True
    assert ack["detail"]["config_fingerprint"] == "fp-i2-new"
    # i2 applied the new config; i1 did not.
    assert {"delta_count": 9} in h2.lifecycle.driver.applied_configs
    assert {"delta_count": 9} not in h1.lifecycle.driver.applied_configs
    assert h2.config_state.applied_fingerprint == "fp-i2-new"
    assert h1.config_state.applied_fingerprint == "fp-seed"

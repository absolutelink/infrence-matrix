"""Phase 16 slice 6: a llama-cpp AGENT hosts N drivable backends, each on its
own admin-facing port (``base_port + offset``) with a distinct engine port.

Exercises the real llama-cpp driver through the registry path:

* ``build_registry`` / ``make_handle`` key each lifecycle to its backend id and
  thread the assignment ``port`` into the driver so two backends on one agent
  never collide on the llama-server port (``serve_port + 1``);
* ``agent.assignments.update`` spawns a fresh lifecycle for a newly-placed
  backend (with its own port) and retires a dropped one;
* ``provider.config.update`` addressed by ``instance_id`` reaches the CORRECT
  backend's driver only.
"""

import asyncio
from pathlib import Path
from typing import Any

from conftest import make_settings
from provider_lib.admin_client import AdminClient
from provider_lib.config_update import ConfigState
from provider_lib.registry import BackendHandle, BackendRegistry
from provider_lib.wire import Frame

from provider_llama_cpp.main import (
    build_registry,
    install_command_handlers,
    make_handle,
    make_lifecycle,
)

INSTANCE_A = "aaaaaaaa-0000-0000-0000-000000000001"
INSTANCE_B = "bbbbbbbb-0000-0000-0000-000000000002"
INSTANCE_C = "cccccccc-0000-0000-0000-000000000003"


class _NoopEmitter:
    def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def _entry(iid: str, port: int, model: str, **over: Any) -> dict[str, Any]:
    entry = {
        "instance_id": iid,
        "provider_definition_id": f"def-{iid}",
        "alias": f"alias-{iid}",
        "backend_config": {"artifacts": {"model": {"path": model}}},
        "config_fingerprint": f"fp-{iid}",
        "port": port,
        "capacity": 1,
        "idle_timeout_seconds": 300,
        "vram_required_bytes": 0,
    }
    entry.update(over)
    return entry


def _reg_entry(iid: str, port: int, model: str) -> dict[str, Any]:
    # Registration-response backend shape (nested under "definition").
    return {
        "instance_id": iid,
        "port": port,
        "definition": {
            "alias": f"alias-{iid}",
            "capacity": 1,
            "config_fingerprint": f"fp-{iid}",
            "backend_config": {"artifacts": {"model": {"path": model}}},
        },
    }


async def _dispatch(client: AdminClient, type_: str, payload: dict) -> dict[str, Any]:
    await client._dispatch(Frame(type=type_, id="cmd-1", epoch=1, payload=payload))
    for _ in range(80):
        frame = await asyncio.wait_for(client._outgoing.get(), timeout=5.0)
        if frame.type == "ack" and frame.reply_to == "cmd-1":
            return frame.payload
    raise AssertionError("no ack produced")


def test_build_registry_threads_ports_into_drivers(
    tmp_path: Path, fake_llama_binary: str, local_model_file: str
) -> None:
    settings = make_settings(tmp_path, fake_llama_binary)
    client = AdminClient(settings)
    result = _FakeResult(
        [
            _reg_entry(INSTANCE_A, 8181, local_model_file),
            _reg_entry(INSTANCE_B, 8182, local_model_file),
        ]
    )
    registry = build_registry(client, result)
    assert len(registry) == 2
    ha, hb = registry.get(INSTANCE_A), registry.get(INSTANCE_B)
    assert ha is not None and hb is not None
    # Each handle carries its assignment port...
    assert ha.port == 8181 and hb.port == 8182
    # ...and the driver derives a DISTINCT engine port (serve_port + 1) so the
    # two llama-server subprocesses never collide on one agent.
    assert ha.lifecycle.driver.backend_port == 8182
    assert hb.lifecycle.driver.backend_port == 8183
    assert ha.lifecycle.instance_id == INSTANCE_A
    assert hb.lifecycle.instance_id == INSTANCE_B


def test_make_handle_builds_drivable_backend_for_new_assignment(
    tmp_path: Path, fake_llama_binary: str, local_model_file: str
) -> None:
    settings = make_settings(tmp_path, fake_llama_binary)
    client = AdminClient(settings)
    handle = make_handle(client, _entry(INSTANCE_C, 8190, local_model_file))
    assert isinstance(handle, BackendHandle)
    assert handle.instance_id == INSTANCE_C
    assert handle.port == 8190
    assert handle.lifecycle.instance_id == INSTANCE_C
    assert handle.lifecycle.driver.backend_port == 8191  # serve_port + 1
    assert handle.config_state.applied_fingerprint == f"fp-{INSTANCE_C}"


async def test_assignments_add_and_remove_on_two_agent(
    tmp_path: Path, fake_llama_binary: str, local_model_file: str
) -> None:
    settings = make_settings(tmp_path, fake_llama_binary)
    client = AdminClient(settings)
    registry = BackendRegistry()
    for iid, port in ((INSTANCE_A, 8181), (INSTANCE_B, 8182)):
        lifecycle = make_lifecycle(
            client,
            {"artifacts": {"model": {"path": local_model_file}}},
            instance_id=iid,
            serve_port=port,
        )
        registry.add(BackendHandle(iid, lifecycle, ConfigState("fp-seed"), port=port))
    install_command_handlers(client, registry, _NoopEmitter())  # type: ignore[arg-type]

    # Drop A (idle), add C on a new port -> A removed, C added with its own port.
    ack = await _dispatch(
        client,
        "agent.assignments.update",
        {
            "assignments": [
                _entry(INSTANCE_B, 8182, local_model_file),
                _entry(INSTANCE_C, 8183, local_model_file),
            ],
            "max_running_backends": 0,
        },
    )
    assert ack["ok"] is True
    assert ack["added"] == [INSTANCE_C]
    assert ack["removed"] == [INSTANCE_A]
    assert INSTANCE_A not in registry and INSTANCE_C in registry
    hc = registry.get(INSTANCE_C)
    assert hc is not None
    assert hc.port == 8183
    assert hc.lifecycle.driver.backend_port == 8184


async def test_config_update_targets_correct_backend_on_two_agent(
    tmp_path: Path, fake_llama_binary: str, local_model_file: str
) -> None:
    settings = make_settings(tmp_path, fake_llama_binary)
    client = AdminClient(settings)
    registry = BackendRegistry()
    for iid, port in ((INSTANCE_A, 8181), (INSTANCE_B, 8182)):
        lifecycle = make_lifecycle(
            client,
            {"artifacts": {"model": {"path": local_model_file}}},
            instance_id=iid,
            serve_port=port,
        )
        registry.add(BackendHandle(iid, lifecycle, ConfigState("fp-seed"), port=port))
    install_command_handlers(client, registry, _NoopEmitter())  # type: ignore[arg-type]
    ha, hb = registry.get(INSTANCE_A), registry.get(INSTANCE_B)
    assert ha is not None and hb is not None
    # Boot B so its driver has a resolved artifact set; A stays stopped.
    await hb.lifecycle.start()
    try:
        new_cfg = {
            "artifacts": {"model": {"path": local_model_file}},
            "context": {"ctx": 8192},
        }
        ack = await _dispatch(
            client,
            "provider.config.update",
            {
                "instance_id": INSTANCE_B,
                "backend_config": new_cfg,
                "config_fingerprint": "fp-b-new",
                "capacity": 1,
            },
        )
        assert ack["ok"] is True, ack
        assert hb.lifecycle.driver._config == new_cfg
        assert hb.config_state.applied_fingerprint == "fp-b-new"
        # A untouched.
        assert ha.lifecycle.driver._config != new_cfg
        assert ha.config_state.applied_fingerprint == "fp-seed"
    finally:
        await hb.lifecycle.stop()
        await hb.lifecycle.driver.aclose()


class _FakeResult:
    """Minimal stand-in for RegistrationResult exposing .backends."""

    def __init__(self, backends: list[dict[str, Any]]) -> None:
        self.backends = backends

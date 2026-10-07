"""Phase 16 slice 6: a gufo AGENT hosts N drivable backends, each on its own
admin-facing port (``base_port + offset``) with a distinct engine port.

Mirrors the llama-cpp slice-6 wiring test against the real gufo driver:
``build_registry`` / ``make_handle`` thread each assignment ``port`` into the
driver (engine binds ``serve_port + 1``), ``agent.assignments.update`` spawns a
fresh lifecycle for a newly-placed backend, and per-backend ``config.update``
reaches the correct driver only.
"""

import asyncio
from pathlib import Path
from typing import Any

from conftest import make_settings
from provider_lib.admin_client import AdminClient
from provider_lib.config_update import ConfigState
from provider_lib.registry import BackendHandle, BackendRegistry
from provider_lib.wire import Frame

from provider_gufo.main import (
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


class _FakeResult:
    def __init__(self, backends: list[dict[str, Any]]) -> None:
        self.backends = backends


async def _dispatch(client: AdminClient, type_: str, payload: dict) -> dict[str, Any]:
    await client._dispatch(Frame(type=type_, id="cmd-1", epoch=1, payload=payload))
    for _ in range(80):
        frame = await asyncio.wait_for(client._outgoing.get(), timeout=5.0)
        if frame.type == "ack" and frame.reply_to == "cmd-1":
            return frame.payload
    raise AssertionError("no ack produced")


def _two_registry(client: AdminClient, model: str) -> BackendRegistry:
    registry = BackendRegistry()
    for iid, port in ((INSTANCE_A, 8181), (INSTANCE_B, 8182)):
        lifecycle = make_lifecycle(
            client,
            {"artifacts": {"model": {"path": model}}},
            instance_id=iid,
            serve_port=port,
        )
        registry.add(BackendHandle(iid, lifecycle, ConfigState("fp-seed"), port=port))
    return registry


def test_build_registry_threads_ports_into_drivers(
    tmp_path: Path, fake_gufo_binary: str, local_model_file: str
) -> None:
    settings = make_settings(tmp_path, fake_gufo_binary)
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
    assert ha.port == 8181 and hb.port == 8182
    # gufo engine binds serve_port + 1 -> distinct per backend.
    assert ha.lifecycle.driver.backend_port == 8182
    assert hb.lifecycle.driver.backend_port == 8183
    assert ha.lifecycle.instance_id == INSTANCE_A
    assert hb.lifecycle.instance_id == INSTANCE_B


def test_make_handle_builds_drivable_backend_for_new_assignment(
    tmp_path: Path, fake_gufo_binary: str, local_model_file: str
) -> None:
    settings = make_settings(tmp_path, fake_gufo_binary)
    client = AdminClient(settings)
    handle = make_handle(client, _entry(INSTANCE_C, 8190, local_model_file))
    assert handle.instance_id == INSTANCE_C
    assert handle.port == 8190
    assert handle.lifecycle.driver.backend_port == 8191
    assert handle.config_state.applied_fingerprint == f"fp-{INSTANCE_C}"


async def test_assignments_add_and_remove_on_two_agent(
    tmp_path: Path, fake_gufo_binary: str, local_model_file: str
) -> None:
    settings = make_settings(tmp_path, fake_gufo_binary)
    client = AdminClient(settings)
    registry = _two_registry(client, local_model_file)
    install_command_handlers(client, registry, _NoopEmitter())  # type: ignore[arg-type]

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
    hc = registry.get(INSTANCE_C)
    assert hc is not None
    assert hc.port == 8183
    assert hc.lifecycle.driver.backend_port == 8184


async def test_cache_clear_skips_shared_dir_when_instance_id_unset(
    tmp_path: Path, fake_gufo_binary: str, local_model_file: str
) -> None:
    """M2: a gufo backend whose driver has no instance_id yet must NOT fall back
    to clearing the shared ``CACHE_DIR/<MACHINE_UID>`` dir (two such backends
    would clobber each other). Once the id is known, its own per-instance dir is
    cleared."""
    settings = make_settings(tmp_path, fake_gufo_binary)
    client = AdminClient(settings)
    # A lifecycle built without an instance id: driver.instance_id stays None.
    lifecycle = make_lifecycle(
        client, {"artifacts": {"model": {"path": local_model_file}}}
    )
    assert lifecycle.driver.instance_id is None
    registry = BackendRegistry()
    registry.add(BackendHandle("i1", lifecycle, ConfigState("fp-seed"), port=8181))
    install_command_handlers(client, registry, _NoopEmitter())  # type: ignore[arg-type]

    shared = settings.CACHE_DIR / settings.MACHINE_UID
    shared.mkdir(parents=True, exist_ok=True)
    marker = shared / "slot.bin"
    marker.write_bytes(b"x" * 16)

    ack = await _dispatch(client, "cache.clear", {"instance_id": "i1"})
    assert ack["ok"] is True, ack
    # The shared MACHINE_UID dir is NOT touched when the id is unset.
    assert marker.exists()

    # Now adopt a real instance id: its per-instance dir IS cleared.
    lifecycle.driver.set_instance_id("i1")
    own = settings.CACHE_DIR / "i1"
    own.mkdir(parents=True, exist_ok=True)
    own_marker = own / "slot.bin"
    own_marker.write_bytes(b"y" * 16)

    ack = await _dispatch(client, "cache.clear", {"instance_id": "i1"})
    assert ack["ok"] is True, ack
    assert not own_marker.exists()
    # The shared dir is still untouched (never in the extra set now).
    assert marker.exists()

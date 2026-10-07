"""Phase 16 slice 6: a halogen AGENT hosts N drivable backends, each on its own
admin-facing port (``base_port + offset``) with a distinct (api, engine) pair.

Mirrors the llama-cpp/gufo slice-6 wiring test against the real halogen driver:
``build_registry`` / ``make_handle`` thread each assignment ``port`` into the
driver (api binds ``serve_port + 1``, engine ``serve_port + 2``), and
``agent.assignments.update`` spawns a fresh lifecycle for a newly-placed backend.
"""

import asyncio
from pathlib import Path
from typing import Any

from conftest import make_settings
from provider_lib.admin_client import AdminClient
from provider_lib.config_update import ConfigState
from provider_lib.registry import BackendHandle, BackendRegistry
from provider_lib.wire import Frame

from provider_halogen.main import (
    build_registry,
    install_command_handlers,
    make_handle,
    make_lifecycle,
)

INSTANCE_A = "aaaaaaaa-0000-0000-0000-000000000001"
INSTANCE_B = "bbbbbbbb-0000-0000-0000-000000000002"
INSTANCE_C = "cccccccc-0000-0000-0000-000000000003"


def _cfg(artifacts: dict[str, str]) -> dict[str, Any]:
    return {
        "artifacts": {
            "model": {"path": artifacts["checkpoint"]},
            "tokenizer": {"path": artifacts["tokenizer"]},
        }
    }


class _NoopEmitter:
    def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


def _entry(
    iid: str, port: int, artifacts: dict[str, str], **over: Any
) -> dict[str, Any]:
    entry = {
        "instance_id": iid,
        "provider_definition_id": f"def-{iid}",
        "alias": f"alias-{iid}",
        "backend_config": _cfg(artifacts),
        "config_fingerprint": f"fp-{iid}",
        "port": port,
        "capacity": 1,
        "idle_timeout_seconds": 300,
        "vram_required_bytes": 0,
    }
    entry.update(over)
    return entry


def _reg_entry(iid: str, port: int, artifacts: dict[str, str]) -> dict[str, Any]:
    return {
        "instance_id": iid,
        "port": port,
        "definition": {
            "alias": f"alias-{iid}",
            "capacity": 1,
            "config_fingerprint": f"fp-{iid}",
            "backend_config": _cfg(artifacts),
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


def test_build_registry_threads_ports_into_drivers(
    tmp_path: Path, fake_halogen_binary: str, local_artifacts: dict[str, str]
) -> None:
    settings = make_settings(tmp_path, fake_halogen_binary)
    client = AdminClient(settings)
    result = _FakeResult(
        [
            _reg_entry(INSTANCE_A, 8181, local_artifacts),
            _reg_entry(INSTANCE_B, 8182, local_artifacts),
        ]
    )
    registry = build_registry(client, result)
    assert len(registry) == 2
    ha, hb = registry.get(INSTANCE_A), registry.get(INSTANCE_B)
    assert ha is not None and hb is not None
    assert ha.port == 8181 and hb.port == 8182
    # halogen derives (api, engine) = (serve_port+1, serve_port+2) -> distinct.
    assert ha.lifecycle.driver.api_port == 8182
    assert ha.lifecycle.driver.engine_port == 8183
    assert hb.lifecycle.driver.api_port == 8183
    assert hb.lifecycle.driver.engine_port == 8184


def test_make_handle_builds_drivable_backend_for_new_assignment(
    tmp_path: Path, fake_halogen_binary: str, local_artifacts: dict[str, str]
) -> None:
    settings = make_settings(tmp_path, fake_halogen_binary)
    client = AdminClient(settings)
    handle = make_handle(client, _entry(INSTANCE_C, 8190, local_artifacts))
    assert handle.instance_id == INSTANCE_C
    assert handle.port == 8190
    assert handle.lifecycle.driver.api_port == 8191
    assert handle.lifecycle.driver.engine_port == 8192
    assert handle.config_state.applied_fingerprint == f"fp-{INSTANCE_C}"


async def test_assignments_add_and_remove_on_two_agent(
    tmp_path: Path, fake_halogen_binary: str, local_artifacts: dict[str, str]
) -> None:
    settings = make_settings(tmp_path, fake_halogen_binary)
    client = AdminClient(settings)
    registry = BackendRegistry()
    for iid, port in ((INSTANCE_A, 8181), (INSTANCE_B, 8182)):
        lifecycle = make_lifecycle(
            client, _cfg(local_artifacts), instance_id=iid, serve_port=port
        )
        registry.add(BackendHandle(iid, lifecycle, ConfigState("fp-seed"), port=port))
    install_command_handlers(client, registry, _NoopEmitter())  # type: ignore[arg-type]

    ack = await _dispatch(
        client,
        "agent.assignments.update",
        {
            "assignments": [
                _entry(INSTANCE_B, 8182, local_artifacts),
                _entry(INSTANCE_C, 8183, local_artifacts),
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
    assert hc.lifecycle.driver.api_port == 8184

"""Port model overhaul: a gufo AGENT hosts N drivable backends behind ONE
admin-facing ``/v1`` (``PROVIDER_PORT``), routed by the request ``model``
(= definition alias). Each backend's gufo engine binds its own OS-assigned
private local port the admin never sees.

Mirrors the llama-cpp wiring test against the real gufo driver:
``build_registry`` / ``make_handle`` stamp the definition ``alias`` on each
handle (no per-backend port; the driver self-allocates its engine port at
start), ``agent.assignments.update`` spawns a fresh lifecycle for a newly-placed
backend, and per-backend ``config.update`` reaches the correct driver only.
"""

import asyncio
from pathlib import Path
from typing import Any

import httpx
from conftest import make_settings
from provider_lib.admin_client import AdminClient
from provider_lib.app_factory import BackendOverrides, create_provider_app
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


def _entry(iid: str, model: str, **over: Any) -> dict[str, Any]:
    entry = {
        "instance_id": iid,
        "provider_definition_id": f"def-{iid}",
        "alias": f"alias-{iid}",
        "backend_config": {"artifacts": {"model": {"path": model}}},
        "config_fingerprint": f"fp-{iid}",
        "capacity": 1,
        "idle_timeout_seconds": 300,
        "vram_required_bytes": 0,
    }
    entry.update(over)
    return entry


def _reg_entry(iid: str, model: str) -> dict[str, Any]:
    return {
        "instance_id": iid,
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
    for iid in (INSTANCE_A, INSTANCE_B):
        lifecycle = make_lifecycle(
            client,
            {"artifacts": {"model": {"path": model}}},
            instance_id=iid,
        )
        registry.add(
            BackendHandle(iid, lifecycle, ConfigState("fp-seed"), alias=f"alias-{iid}")
        )
    return registry


def test_build_registry_sets_alias_handles(
    tmp_path: Path, fake_gufo_binary: str, local_model_file: str
) -> None:
    settings = make_settings(tmp_path, fake_gufo_binary)
    client = AdminClient(settings)
    result = _FakeResult(
        [
            _reg_entry(INSTANCE_A, local_model_file),
            _reg_entry(INSTANCE_B, local_model_file),
        ]
    )
    registry = build_registry(client, result)
    assert len(registry) == 2
    ha, hb = registry.get(INSTANCE_A), registry.get(INSTANCE_B)
    assert ha is not None and hb is not None
    # Each handle carries its definition alias (for model routing)...
    assert ha.alias == f"alias-{INSTANCE_A}" and hb.alias == f"alias-{INSTANCE_B}"
    # ...and NO per-backend port (the admin no longer allocates one).
    assert ha.port is None and hb.port is None
    # The engine port is allocated at start, so it is unset on a fresh handle.
    assert ha.lifecycle.driver.backend_port is None
    assert hb.lifecycle.driver.backend_port is None
    assert ha.lifecycle.instance_id == INSTANCE_A
    assert hb.lifecycle.instance_id == INSTANCE_B
    assert registry.resolve_by_model(f"alias-{INSTANCE_A}") is ha
    assert registry.resolve_by_model(f"alias-{INSTANCE_B}") is hb


def test_make_handle_builds_drivable_backend_for_new_assignment(
    tmp_path: Path, fake_gufo_binary: str, local_model_file: str
) -> None:
    settings = make_settings(tmp_path, fake_gufo_binary)
    client = AdminClient(settings)
    handle = make_handle(client, _entry(INSTANCE_C, local_model_file))
    assert handle.instance_id == INSTANCE_C
    assert handle.alias == f"alias-{INSTANCE_C}"
    assert handle.port is None
    assert handle.lifecycle.driver.backend_port is None
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
                _entry(INSTANCE_B, local_model_file),
                _entry(INSTANCE_C, local_model_file),
            ],
            "max_running_backends": 0,
        },
    )
    assert ack["ok"] is True
    assert ack["added"] == [INSTANCE_C]
    assert ack["removed"] == [INSTANCE_A]
    hc = registry.get(INSTANCE_C)
    assert hc is not None
    assert hc.alias == f"alias-{INSTANCE_C}"
    assert hc.port is None
    assert hc.lifecycle.driver.backend_port is None


async def test_two_backends_distinct_engine_ports(
    tmp_path: Path, fake_gufo_binary: str, local_model_file: str
) -> None:
    """Two started backends on one agent bind DISTINCT OS-assigned engine ports
    (no collision, no config/env derivation)."""
    settings = make_settings(tmp_path, fake_gufo_binary)
    client = AdminClient(settings)
    la = make_lifecycle(
        client,
        {"artifacts": {"model": {"path": local_model_file}}},
        instance_id=INSTANCE_A,
    )
    lb = make_lifecycle(
        client,
        {"artifacts": {"model": {"path": local_model_file}}},
        instance_id=INSTANCE_B,
    )
    try:
        await la.start()
        await lb.start()
        pa = la.driver.backend_port
        pb = lb.driver.backend_port
        assert isinstance(pa, int) and pa > 0
        assert isinstance(pb, int) and pb > 0
        assert pa != pb
    finally:
        await la.stop()
        await la.driver.aclose()
        await lb.stop()
        await lb.driver.aclose()


async def test_single_env_port_routes_by_alias(
    tmp_path: Path, fake_gufo_binary: str, local_model_file: str
) -> None:
    """The agent's ONE /v1 surface routes each request to the backend whose
    alias matches the request ``model``; an unknown model is a 404."""
    settings = make_settings(tmp_path, fake_gufo_binary)
    client = AdminClient(settings)
    registry = BackendRegistry()
    for iid in (INSTANCE_A, INSTANCE_B):
        lifecycle = make_lifecycle(
            client,
            {"artifacts": {"model": {"path": local_model_file}}},
            instance_id=iid,
        )
        await lifecycle.start()
        registry.add(
            BackendHandle(iid, lifecycle, ConfigState("fp-seed"), alias=f"alias-{iid}")
        )
    app = create_provider_app(
        settings,
        BackendOverrides(provider_type="gufo", version="dev", registry=registry),
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://provider"
        ) as v1:
            for iid in (INSTANCE_A, INSTANCE_B):
                resp = await v1.post(
                    "/v1/responses",
                    json={"model": f"alias-{iid}", "input": "go", "stream": True},
                )
                assert resp.status_code == 200, iid
                assert resp.headers["content-type"].startswith("text/event-stream")
            assert (
                await v1.post("/v1/responses", json={"model": "nope", "input": "go"})
            ).status_code == 404
    finally:
        for handle in registry.handles():
            await handle.lifecycle.stop()
            await handle.lifecycle.driver.aclose()


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
    registry.add(BackendHandle("i1", lifecycle, ConfigState("fp-seed"), alias="a-i1"))
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

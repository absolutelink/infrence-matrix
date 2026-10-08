"""Port model overhaul: a llama-cpp AGENT hosts N drivable backends behind ONE
admin-facing ``/v1`` (``PROVIDER_PORT``), routed by the request ``model``
(= definition alias). Each backend's llama-server engine binds its own
OS-assigned private local port the admin never sees.

Exercises the real llama-cpp driver through the registry path:

* ``build_registry`` / ``make_handle`` key each lifecycle to its backend id and
  stamp the definition ``alias`` on the handle (no per-backend port; the driver
  self-allocates its engine port at start);
* ``agent.assignments.update`` spawns a fresh lifecycle for a newly-placed
  backend and retires a dropped one;
* ``provider.config.update`` addressed by ``instance_id`` reaches the CORRECT
  backend's driver only;
* two started backends get DISTINCT engine ports and are BOTH routable by alias
  on the single env-port surface.
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
    # Registration-response backend shape (nested under "definition").
    return {
        "instance_id": iid,
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


def test_build_registry_sets_alias_handles(
    tmp_path: Path, fake_llama_binary: str, local_model_file: str
) -> None:
    settings = make_settings(tmp_path, fake_llama_binary)
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
    # resolve_by_model routes each alias to the right handle.
    assert registry.resolve_by_model(f"alias-{INSTANCE_A}") is ha
    assert registry.resolve_by_model(f"alias-{INSTANCE_B}") is hb


def test_make_handle_builds_drivable_backend_for_new_assignment(
    tmp_path: Path, fake_llama_binary: str, local_model_file: str
) -> None:
    settings = make_settings(tmp_path, fake_llama_binary)
    client = AdminClient(settings)
    handle = make_handle(client, _entry(INSTANCE_C, local_model_file))
    assert isinstance(handle, BackendHandle)
    assert handle.instance_id == INSTANCE_C
    assert handle.alias == f"alias-{INSTANCE_C}"
    assert handle.port is None
    assert handle.lifecycle.instance_id == INSTANCE_C
    assert handle.lifecycle.driver.backend_port is None
    assert handle.config_state.applied_fingerprint == f"fp-{INSTANCE_C}"


def test_modality_threads_into_driver(
    tmp_path: Path, fake_llama_binary: str, local_model_file: str
) -> None:
    """Phase 18 slice 3: the definition/entry modality reaches the built driver
    (assignment entry via make_handle, registration definition via
    build_registry); an absent modality defaults to "llm"."""
    settings = make_settings(tmp_path, fake_llama_binary)
    client = AdminClient(settings)

    # Assignment entry -> make_handle.
    emb_handle = make_handle(
        client, _entry(INSTANCE_C, local_model_file, modality="embedding")
    )
    assert emb_handle.lifecycle.driver.modality == "embedding"
    assert emb_handle.config_state.modality == "embedding"
    # Absent modality -> llm default.
    default_handle = make_handle(client, _entry(INSTANCE_A, local_model_file))
    assert default_handle.lifecycle.driver.modality == "llm"

    # Registration definition -> build_registry.
    reg_entry = _reg_entry(INSTANCE_B, local_model_file)
    reg_entry["definition"]["modality"] = "embedding"
    registry = build_registry(client, _FakeResult([reg_entry]))
    hb = registry.get(INSTANCE_B)
    assert hb is not None
    assert hb.lifecycle.driver.modality == "embedding"
    assert hb.config_state.modality == "embedding"


async def test_assignments_add_and_remove_on_two_agent(
    tmp_path: Path, fake_llama_binary: str, local_model_file: str
) -> None:
    settings = make_settings(tmp_path, fake_llama_binary)
    client = AdminClient(settings)
    registry = BackendRegistry()
    for iid in (INSTANCE_A, INSTANCE_B):
        lifecycle = make_lifecycle(
            client,
            {"artifacts": {"model": {"path": local_model_file}}},
            instance_id=iid,
        )
        registry.add(
            BackendHandle(iid, lifecycle, ConfigState("fp-seed"), alias=f"alias-{iid}")
        )
    install_command_handlers(client, registry, _NoopEmitter())  # type: ignore[arg-type]

    # Drop A (idle), add C -> A removed, C added with its own alias.
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
    assert INSTANCE_A not in registry and INSTANCE_C in registry
    hc = registry.get(INSTANCE_C)
    assert hc is not None
    assert hc.alias == f"alias-{INSTANCE_C}"
    assert hc.port is None
    assert hc.lifecycle.driver.backend_port is None


async def test_config_update_targets_correct_backend_on_two_agent(
    tmp_path: Path, fake_llama_binary: str, local_model_file: str
) -> None:
    settings = make_settings(tmp_path, fake_llama_binary)
    client = AdminClient(settings)
    registry = BackendRegistry()
    for iid in (INSTANCE_A, INSTANCE_B):
        lifecycle = make_lifecycle(
            client,
            {"artifacts": {"model": {"path": local_model_file}}},
            instance_id=iid,
        )
        registry.add(
            BackendHandle(iid, lifecycle, ConfigState("fp-seed"), alias=f"alias-{iid}")
        )
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


async def test_two_backends_distinct_engine_ports(
    tmp_path: Path, fake_llama_binary: str, local_model_file: str
) -> None:
    """Two started backends on one agent bind DISTINCT OS-assigned engine ports
    (no collision, no config/env derivation)."""
    settings = make_settings(tmp_path, fake_llama_binary)
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
    tmp_path: Path, fake_llama_binary: str, local_model_file: str
) -> None:
    """The agent's ONE /v1 surface routes each request to the backend whose
    alias matches the request ``model``; an unknown model is a 404."""
    settings = make_settings(tmp_path, fake_llama_binary)
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
        BackendOverrides(provider_type="llama-cpp", version="dev", registry=registry),
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


class _FakeResult:
    """Minimal stand-in for RegistrationResult exposing .backends."""

    def __init__(self, backends: list[dict[str, Any]]) -> None:
        self.backends = backends

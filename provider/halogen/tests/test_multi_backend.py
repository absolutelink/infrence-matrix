"""Port model overhaul: a halogen AGENT hosts N drivable backends behind ONE
admin-facing ``/v1`` (``PROVIDER_PORT``), routed by the request ``model``
(= definition alias). Each backend's halogen engine binds its own OS-assigned
private ``(api, engine)`` port pair the admin never sees.

Mirrors the llama-cpp/gufo wiring test against the real halogen driver:
``build_registry`` / ``make_handle`` stamp the definition ``alias`` on each
handle (no per-backend port; the driver self-allocates its engine pair at
start), and ``agent.assignments.update`` spawns a fresh lifecycle for a
newly-placed backend.
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


def _entry(iid: str, artifacts: dict[str, str], **over: Any) -> dict[str, Any]:
    entry = {
        "instance_id": iid,
        "provider_definition_id": f"def-{iid}",
        "alias": f"alias-{iid}",
        "backend_config": _cfg(artifacts),
        "config_fingerprint": f"fp-{iid}",
        "capacity": 1,
        "idle_timeout_seconds": 300,
        "vram_required_bytes": 0,
    }
    entry.update(over)
    return entry


def _reg_entry(iid: str, artifacts: dict[str, str]) -> dict[str, Any]:
    return {
        "instance_id": iid,
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


def test_build_registry_sets_alias_handles(
    tmp_path: Path, fake_halogen_binary: str, local_artifacts: dict[str, str]
) -> None:
    settings = make_settings(tmp_path, fake_halogen_binary)
    client = AdminClient(settings)
    result = _FakeResult(
        [
            _reg_entry(INSTANCE_A, local_artifacts),
            _reg_entry(INSTANCE_B, local_artifacts),
        ]
    )
    registry = build_registry(client, result)
    assert len(registry) == 2
    ha, hb = registry.get(INSTANCE_A), registry.get(INSTANCE_B)
    assert ha is not None and hb is not None
    assert ha.alias == f"alias-{INSTANCE_A}" and hb.alias == f"alias-{INSTANCE_B}"
    # Engine ports are allocated at start, so unset on a fresh handle.
    assert ha.lifecycle.driver.api_port is None
    assert hb.lifecycle.driver.api_port is None
    assert registry.resolve_by_model(f"alias-{INSTANCE_A}") is ha
    assert registry.resolve_by_model(f"alias-{INSTANCE_B}") is hb


def test_make_handle_builds_drivable_backend_for_new_assignment(
    tmp_path: Path, fake_halogen_binary: str, local_artifacts: dict[str, str]
) -> None:
    settings = make_settings(tmp_path, fake_halogen_binary)
    client = AdminClient(settings)
    handle = make_handle(client, _entry(INSTANCE_C, local_artifacts))
    assert handle.instance_id == INSTANCE_C
    assert handle.alias == f"alias-{INSTANCE_C}"
    assert handle.lifecycle.driver.api_port is None
    assert handle.config_state.applied_fingerprint == f"fp-{INSTANCE_C}"


async def test_assignments_add_and_remove_on_two_agent(
    tmp_path: Path, fake_halogen_binary: str, local_artifacts: dict[str, str]
) -> None:
    settings = make_settings(tmp_path, fake_halogen_binary)
    client = AdminClient(settings)
    registry = BackendRegistry()
    for iid in (INSTANCE_A, INSTANCE_B):
        lifecycle = make_lifecycle(client, _cfg(local_artifacts), instance_id=iid)
        registry.add(
            BackendHandle(iid, lifecycle, ConfigState("fp-seed"), alias=f"alias-{iid}")
        )
    install_command_handlers(client, registry, _NoopEmitter())  # type: ignore[arg-type]

    ack = await _dispatch(
        client,
        "agent.assignments.update",
        {
            "assignments": [
                _entry(INSTANCE_B, local_artifacts),
                _entry(INSTANCE_C, local_artifacts),
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
    assert hc.lifecycle.driver.api_port is None


async def test_two_backends_distinct_engine_ports(
    tmp_path: Path, fake_halogen_binary: str, local_artifacts: dict[str, str]
) -> None:
    """Two started backends on one agent get DISTINCT (api, engine) pairs."""
    settings = make_settings(tmp_path, fake_halogen_binary)
    client = AdminClient(settings)
    la = make_lifecycle(client, _cfg(local_artifacts), instance_id=INSTANCE_A)
    lb = make_lifecycle(client, _cfg(local_artifacts), instance_id=INSTANCE_B)
    try:
        await la.start()
        await lb.start()
        pa = (la.driver.api_port, la.driver.engine_port)
        pb = (lb.driver.api_port, lb.driver.engine_port)
        assert all(isinstance(p, int) and p > 0 for p in pa + pb)
        assert pa[0] != pa[1] and pb[0] != pb[1]
        # The two backends' four ports are all distinct (no collision).
        assert len({*pa, *pb}) == 4
    finally:
        await la.stop()
        await la.driver.aclose()
        await lb.stop()
        await lb.driver.aclose()


async def test_single_env_port_routes_by_alias(
    tmp_path: Path, fake_halogen_binary: str, local_artifacts: dict[str, str]
) -> None:
    """The agent's ONE /v1 surface routes each request to the backend whose
    alias matches the request ``model``; an unknown model is a 404."""
    settings = make_settings(tmp_path, fake_halogen_binary)
    client = AdminClient(settings)
    registry = BackendRegistry()
    for iid in (INSTANCE_A, INSTANCE_B):
        lifecycle = make_lifecycle(client, _cfg(local_artifacts), instance_id=iid)
        await lifecycle.start()
        registry.add(
            BackendHandle(iid, lifecycle, ConfigState("fp-seed"), alias=f"alias-{iid}")
        )
    app = create_provider_app(
        settings,
        BackendOverrides(provider_type="halogen", version="dev", registry=registry),
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

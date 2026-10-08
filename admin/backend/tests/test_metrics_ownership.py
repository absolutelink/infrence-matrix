"""Machine metrics ownership (agent-keyed, Phase 16): assign on connect,
single owner per machine, snapshot storage from owner only, release on
disconnect, takeover after the owner leaves, no assignment without declared
categories."""

import json
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.models import Machine, ProviderAgent, ProviderDefinition
from app.services import metrics_service, redis_keys
from app.services.wire import Frame


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def assign_calls(monkeypatch):
    """Record metrics.assign commands instead of pushing over a live WS."""
    calls: list[tuple[str, dict]] = []

    async def fake_send_command(agent_id, _type_, _payload, timeout=30.0):  # noqa: ARG001
        calls.append((agent_id, dict(_payload)))
        return Frame(type="ack", reply_to="x", payload={"ok": True})

    monkeypatch.setattr(metrics_service.manager, "send_command", fake_send_command)
    return calls


def _owner(clean_redis, machine_uid: str) -> str | None:
    return clean_redis.get(redis_keys.metrics_owner_key(machine_uid))


def _make_stack(session: Session, machine_uid: str) -> None:
    if session.query(Machine).filter(Machine.uid == machine_uid).first() is None:
        session.add(
            Machine(
                uid=machine_uid,
                name=f"name-{machine_uid}",
                registration_secret="mach-secret",
            )
        )
    alias = f"alias-{machine_uid}"
    if (
        session.query(ProviderDefinition)
        .filter(ProviderDefinition.alias == alias)
        .first()
        is None
    ):
        session.add(
            ProviderDefinition(
                alias=alias,
                provider_type="mock",
                backend_config={"model": {"file": "m.gguf"}},
            )
        )
    session.commit()


def _register(
    client: TestClient, machine_uid: str, agent_id: str, categories: list[str]
) -> dict:
    resp = client.post(
        "/admin/api/providers/register",
        json={
            "machine_uid": machine_uid,
            "machine_secret": "mach-secret",
            "agent_id": agent_id,
            "provider_type": "mock",
            "schema": {"type": "object"},
            "version": settings.VERSION,
            "base_port": 8081,
            "hardware": {"gpus": [], "total_vram_bytes": 1},
            "metrics_categories": categories,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _connect(client: TestClient, reg: dict):
    return client.websocket_connect(
        f"/provider/ws?agent_id={reg['agent_id']}",
        headers={"Authorization": f"Bearer {reg['agent_secret']}"},
    )


def _wait_for(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("condition not met in time")


def test_registration_persists_categories(
    client: TestClient, session: Session, clean_redis
) -> None:
    _make_stack(session, "mm-reg")
    reg = _register(client, "mm-reg", "agent-a", ["cpu", "vram"])
    raw = clean_redis.get(redis_keys.metrics_cats_key(reg["agent_id"]))
    assert json.loads(raw) == ["cpu", "vram"]


def test_first_connected_agent_owns_machine(
    client: TestClient, session: Session, assign_calls, clean_redis
) -> None:
    _make_stack(session, "mm-one")
    reg_a = _register(client, "mm-one", "agent-a", ["cpu"])
    reg_b = _register(client, "mm-one", "agent-b", ["cpu"])

    with _connect(client, reg_a) as ws:
        hello = Frame.model_validate_json(ws.receive_text())
        assert hello.type == "provider.hello"

        _wait_for(lambda: len(assign_calls) >= 1)
        assigned_id, payload = assign_calls[0]
        assert assigned_id == reg_a["agent_id"]
        assert payload["machine_uid"] == "mm-one"
        assert payload["categories"] == ["cpu"]
        assert _owner(clean_redis, "mm-one") == reg_a["agent_id"]

        # B connects on the SAME machine: must NOT be assigned.
        with _connect(client, reg_b) as ws_b:
            ws_b.receive_text()  # hello
            time.sleep(0.2)
            assert [c[0] for c in assign_calls] == [reg_a["agent_id"]]

    # A disconnected: ownership released.
    _wait_for(lambda: _owner(clean_redis, "mm-one") is None)


def test_owner_snapshot_stored_nonowner_dropped(
    client: TestClient, session: Session, assign_calls, clean_redis
) -> None:
    _make_stack(session, "mm-snap")
    reg_a = _register(client, "mm-snap", "agent-a", ["cpu"])
    reg_b = _register(client, "mm-snap", "agent-b", ["cpu"])
    key = redis_keys.metrics_machine_key("mm-snap")

    with _connect(client, reg_a) as ws:
        hello = Frame.model_validate_json(ws.receive_text())
        _wait_for(lambda: len(assign_calls) >= 1)

        ws.send_text(
            Frame(
                type="metrics.machine",
                id="m-1",
                epoch=hello.epoch,
                payload={"cpu": {"cores": 8}},
            ).to_json()
        )
        _wait_for(lambda: clean_redis.get(key) is not None)
        assert json.loads(clean_redis.get(key))["cpu"]["cores"] == 8

    # Now B is connected while the lease still names A: B's snapshot drops.
    _wait_for(lambda: _owner(clean_redis, "mm-snap") is None)
    clean_redis.set(redis_keys.metrics_owner_key("mm-snap"), reg_a["agent_id"], ex=60)
    with _connect(client, reg_b) as ws_b:
        hello_b = Frame.model_validate_json(ws_b.receive_text())
        ws_b.send_text(
            Frame(
                type="metrics.machine",
                id="m-2",
                epoch=hello_b.epoch,
                payload={"cpu": {"cores": 16}},
            ).to_json()
        )
        # Sync: the pong arrives only after the metrics frame was handled.
        ws_b.send_text(Frame(type="ping", id="p-2", epoch=hello_b.epoch).to_json())
        pong = Frame.model_validate_json(ws_b.receive_text())
        assert pong.type == "pong"
        assert json.loads(clean_redis.get(key))["cpu"]["cores"] == 8
        # Lease untouched (still A, not refreshed by B).
        assert _owner(clean_redis, "mm-snap") == reg_a["agent_id"]


def test_takeover_after_owner_disconnects(
    client: TestClient, session: Session, assign_calls, clean_redis
) -> None:
    _make_stack(session, "mm-take")
    reg_a = _register(client, "mm-take", "agent-a", ["cpu"])
    # Phase 17 M1: only a machine-wide-capable agent can take over the owner
    # lease (a vram-only agent no longer claims it), so B declares cpu.
    reg_b = _register(client, "mm-take", "agent-b", ["cpu"])

    with _connect(client, reg_a) as ws:
        ws.receive_text()
        _wait_for(lambda: len(assign_calls) >= 1)

    _wait_for(lambda: _owner(clean_redis, "mm-take") is None)

    with _connect(client, reg_b) as ws_b:
        ws_b.receive_text()
        _wait_for(lambda: len(assign_calls) >= 2)
        assert assign_calls[-1][0] == reg_b["agent_id"]
        assert assign_calls[-1][1]["categories"] == ["cpu"]
        assert _owner(clean_redis, "mm-take") == reg_b["agent_id"]


def test_no_categories_means_no_assignment(
    client: TestClient,
    session: Session,
    assign_calls,
    clean_redis,  # noqa: ARG001
) -> None:
    _make_stack(session, "mm-none")
    reg = _register(client, "mm-none", "agent-a", [])

    with _connect(client, reg) as ws:
        ws.receive_text()
        time.sleep(0.2)
        assert assign_calls == []
        assert clean_redis.get(redis_keys.metrics_owner_key("mm-none")) is None


async def test_assign_rolls_back_lease_when_command_fails(
    session: Session, monkeypatch
) -> None:
    """If metrics.assign can't be delivered, the lease must not stick."""
    import os

    import redis.asyncio as aioredis

    machine = Machine(uid="mm-rb", name="mm-rb-machine")
    definition = ProviderDefinition(
        alias="rb-a",
        provider_type="mock",
        backend_config={"model": {"file": "m.gguf"}},
    )
    session.add_all([machine, definition])
    session.commit()
    agent = ProviderAgent(
        machine_id=machine.id, provider_type="mock", agent_id="rb-agent"
    )
    session.add(agent)
    session.commit()
    agent_id = str(agent.id)

    aredis = aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    await aredis.set(redis_keys.metrics_cats_key(agent_id), '["cpu"]')

    async def boom(*_args, **_kwargs):  # noqa: ARG001
        raise ConnectionError("no live connection")

    monkeypatch.setattr(metrics_service.manager, "send_command", boom)
    assigned = await metrics_service.assign_ownership(aredis, agent_id)
    assert assigned is False
    assert await aredis.get(redis_keys.metrics_owner_key("mm-rb")) is None

    # And with a working command, ownership is acquired.
    async def ok(*_args, **_kwargs):  # noqa: ARG001
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(metrics_service.manager, "send_command", ok)
    assert await metrics_service.assign_ownership(aredis, agent_id) is True
    assert await aredis.get(redis_keys.metrics_owner_key("mm-rb")) == agent_id
    await aredis.aclose()


# ---------------------------------------------------------------------------
# Phase 17 slice 3: per-GPU merge (non-owner GPU frames stored, machine-wide
# still owner-gated) + read_machine_metrics merge-on-read.
# ---------------------------------------------------------------------------


def _vram_payload(uuid: str, gid: int, total: int, used: int, util: float) -> dict:
    """A single-GPU metrics.machine payload shaped like the provider emitter."""
    gpu = {
        "id": gid,
        "uuid": uuid,
        "name": f"card{gid}",
        "vendor": "nvidia",
        "vram_total": total,
        "vram_used": used,
        "vram_free": total - used,
        "utilization": util,
        "temperature": 42.0,
    }
    return {
        "vram": {
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": total - used,
            "gpu_count": 1,
            "gpus": [gpu],
        },
        "gpu_usage": {
            "utilization_percent": util,
            "gpus": [{"id": gid, "utilization": util}],
        },
        "assigned_gpus": [uuid],
    }


def _make_agents(
    session: Session, machine_uid: str, names: list[str]
) -> dict[str, str]:
    """Create a machine + provider agents; return {name: agent PK uuid str}."""
    if session.query(Machine).filter(Machine.uid == machine_uid).first() is None:
        session.add(
            Machine(
                uid=machine_uid,
                name=f"name-{machine_uid}",
                registration_secret="mach-secret",
            )
        )
        session.commit()
    machine = session.query(Machine).filter(Machine.uid == machine_uid).one()
    out: dict[str, str] = {}
    for name in names:
        agent = ProviderAgent(
            machine_id=machine.id, provider_type="mock", agent_id=name
        )
        session.add(agent)
        session.commit()
        out[name] = str(agent.id)
    return out


async def _aredis():
    import os

    import redis.asyncio as aioredis

    return aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)


async def test_nonowner_gpu_frame_is_stored(session: Session) -> None:
    """A non-owner's GPU-category frame writes a per-agent partial (not dropped)."""
    ids = _make_agents(session, "mm-gpu-nonowner", ["a", "b"])
    r = await _aredis()
    try:
        # Lease names agent a; b is a non-owner but still reports its GPU.
        await r.set(redis_keys.metrics_owner_key("mm-gpu-nonowner"), ids["a"], ex=60)
        stored = await metrics_service.handle_machine_metrics(
            r, ids["b"], _vram_payload("GPU-B", 0, 200, 20, 40.0)
        )
        assert stored is True
        partial_key = redis_keys.metrics_agent_partial_key("mm-gpu-nonowner", ids["b"])
        raw = await r.get(partial_key)
        assert raw is not None
        part = json.loads(raw)
        assert part["vram"]["gpus"][0]["uuid"] == "GPU-B"
        # N2: the partial stays minimal — assigned_gpus is not stored (read
        # derives attribution from the unioned vram.gpus uuids).
        assert "assigned_gpus" not in part
        # The non-owner did NOT touch the machine-wide snapshot or the lease.
        assert await r.get(redis_keys.metrics_machine_key("mm-gpu-nonowner")) is None
        assert await r.get(redis_keys.metrics_owner_key("mm-gpu-nonowner")) == ids["a"]
    finally:
        await r.aclose()


async def test_nonowner_machine_only_frame_dropped(session: Session) -> None:
    """A non-owner's machine-only (cpu) frame is still dropped entirely."""
    ids = _make_agents(session, "mm-mw-nonowner", ["a", "b"])
    r = await _aredis()
    try:
        await r.set(redis_keys.metrics_owner_key("mm-mw-nonowner"), ids["a"], ex=60)
        stored = await metrics_service.handle_machine_metrics(
            r, ids["b"], {"cpu": {"cores": 16}}
        )
        assert stored is False
        assert (
            await r.get(
                redis_keys.metrics_agent_partial_key("mm-mw-nonowner", ids["b"])
            )
            is None
        )
        assert await r.get(redis_keys.metrics_machine_key("mm-mw-nonowner")) is None
    finally:
        await r.aclose()


async def test_owner_machinewide_refreshes_lease_and_writes_snapshot(
    session: Session,
) -> None:
    """The owner's machine-wide frame refreshes the lease + writes the snapshot."""
    ids = _make_agents(session, "mm-owner-mw", ["a"])
    r = await _aredis()
    try:
        owner_key = redis_keys.metrics_owner_key("mm-owner-mw")
        await r.set(owner_key, ids["a"], ex=60)
        stored = await metrics_service.handle_machine_metrics(
            r, ids["a"], {"os_ram": {"total_bytes": 1000}, "cpu": {"cores": 8}}
        )
        assert stored is True
        snap = json.loads(await r.get(redis_keys.metrics_machine_key("mm-owner-mw")))
        assert snap["cpu"]["cores"] == 8
        assert snap["os_ram"]["total_bytes"] == 1000
        assert snap["owner_agent_id"] == ids["a"]
        # Lease still held by the owner.
        assert await r.get(owner_key) == ids["a"]
    finally:
        await r.aclose()


async def test_read_machine_metrics_merges_two_agents(session: Session) -> None:
    """Two agents each reporting a different GPU merge into one union."""
    ids = _make_agents(session, "mm-merge", ["a", "b"])
    r = await _aredis()
    try:
        await metrics_service.handle_machine_metrics(
            r, ids["a"], _vram_payload("GPU-A", 0, 100, 10, 20.0)
        )
        await metrics_service.handle_machine_metrics(
            r, ids["b"], _vram_payload("GPU-B", 0, 200, 20, 40.0)
        )
        merged = await metrics_service.read_machine_metrics(r, "mm-merge")
        assert merged["vram"]["gpu_count"] == 2
        assert merged["vram"]["total_bytes"] == 300
        assert merged["vram"]["used_bytes"] == 30
        assert merged["vram"]["free_bytes"] == 270
        by_uuid = {g["uuid"]: g for g in merged["vram"]["gpus"]}
        assert by_uuid["GPU-A"]["agent_id"] == ids["a"]
        assert by_uuid["GPU-B"]["agent_id"] == ids["b"]
        # gpu_usage derived from the union: mean of per-gpu utilization.
        assert merged["gpu_usage"]["utilization_percent"] == 30.0
        usage_by_uuid = {g["uuid"]: g for g in merged["gpu_usage"]["gpus"]}
        assert usage_by_uuid["GPU-A"]["agent_id"] == ids["a"]
        assert set(merged["reporting_agents"]) == {ids["a"], ids["b"]}
    finally:
        await r.aclose()


async def test_read_machine_metrics_overlays_machinewide(session: Session) -> None:
    """Owner machine-wide sections overlay the merged GPU union on read."""
    ids = _make_agents(session, "mm-merge-mw", ["a", "b"])
    r = await _aredis()
    try:
        await r.set(redis_keys.metrics_owner_key("mm-merge-mw"), ids["a"], ex=60)
        await metrics_service.handle_machine_metrics(
            r, ids["a"], _vram_payload("GPU-A", 0, 100, 10, 20.0)
        )
        await metrics_service.handle_machine_metrics(
            r, ids["b"], _vram_payload("GPU-B", 0, 200, 20, 40.0)
        )
        await metrics_service.handle_machine_metrics(
            r, ids["a"], {"cpu": {"cores": 8}, "os_ram": {"total_bytes": 4096}}
        )
        merged = await metrics_service.read_machine_metrics(r, "mm-merge-mw")
        assert merged["vram"]["gpu_count"] == 2
        assert merged["cpu"]["cores"] == 8
        assert merged["os_ram"]["total_bytes"] == 4096
        assert merged["owner_agent_id"] == ids["a"]
    finally:
        await r.aclose()


async def test_read_machine_metrics_empty(session: Session) -> None:
    """No partials + no snapshot => minimal dict with no GPU sections."""
    _make_agents(session, "mm-empty", ["a"])
    r = await _aredis()
    try:
        merged = await metrics_service.read_machine_metrics(r, "mm-empty")
        assert merged == {"machine_uid": "mm-empty"}
        assert "vram" not in merged
        assert "gpu_usage" not in merged
    finally:
        await r.aclose()


# ---------------------------------------------------------------------------
# Phase 17 slice 3 review fixes: M1 assign gate, M2 reporting, L1 uuid keying,
# L2 numeric coercion.
# ---------------------------------------------------------------------------


async def test_assign_gate_gpu_only_agent_does_not_claim_lease(
    session: Session, monkeypatch
) -> None:
    """M1: only agents reporting a machine-wide category claim the owner lease."""
    ids = _make_agents(session, "mm-assign-gate", ["gpuonly", "cpucap"])
    r = await _aredis()

    async def ok(*_args, **_kwargs):  # noqa: ARG001
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(metrics_service.manager, "send_command", ok)
    try:
        # GPU-only agent declares vram/gpu_usage but no machine-wide category.
        await r.set(
            redis_keys.metrics_cats_key(ids["gpuonly"]), '["vram", "gpu_usage"]'
        )
        assert await metrics_service.assign_ownership(r, ids["gpuonly"]) is False
        assert await r.get(redis_keys.metrics_owner_key("mm-assign-gate")) is None

        # A machine-wide-capable agent can then claim the still-free lease.
        await r.set(redis_keys.metrics_cats_key(ids["cpucap"]), '["cpu", "vram"]')
        assert await metrics_service.assign_ownership(r, ids["cpucap"]) is True
        assert (
            await r.get(redis_keys.metrics_owner_key("mm-assign-gate")) == ids["cpucap"]
        )
    finally:
        await r.aclose()


async def test_read_gpu_usage_only_partial_not_counted(session: Session) -> None:
    """M2: a partial contributing no vram GPU is absent from reporting_agents."""
    ids = _make_agents(session, "mm-m2", ["a", "b"])
    r = await _aredis()
    try:
        await metrics_service.handle_machine_metrics(
            r,
            ids["a"],
            {
                "gpu_usage": {
                    "utilization_percent": 50.0,
                    "gpus": [{"id": 0, "utilization": 50.0}],
                }
            },
        )
        await metrics_service.handle_machine_metrics(
            r, ids["b"], _vram_payload("GPU-B", 0, 200, 20, 40.0)
        )
        merged = await metrics_service.read_machine_metrics(r, "mm-m2")
        assert merged["vram"]["gpu_count"] == 1
        assert merged["reporting_agents"] == [ids["b"]]
        assert ids["a"] not in merged["reporting_agents"]
    finally:
        await r.aclose()


async def test_read_skips_gpu_without_uuid(session: Session) -> None:
    """L1: a vram GPU entry with no uuid is skipped (agrees with hardware.py)."""
    ids = _make_agents(session, "mm-l1", ["a"])
    r = await _aredis()
    try:
        partial = {
            "vram": {
                "gpus": [{"id": 0, "vram_total": 100, "utilization": 10.0}]  # no uuid
            }
        }
        await r.set(
            redis_keys.metrics_agent_partial_key("mm-l1", ids["a"]),
            json.dumps(partial),
            ex=60,
        )
        merged = await metrics_service.read_machine_metrics(r, "mm-l1")
        assert "vram" not in merged
        assert "reporting_agents" not in merged
    finally:
        await r.aclose()


async def test_read_coerces_nonnumeric_fields(session: Session) -> None:
    """L2: non-numeric vram/utilization fields coerce to 0 instead of raising."""
    ids = _make_agents(session, "mm-l2", ["a"])
    r = await _aredis()
    try:
        partial = {
            "vram": {
                "gpus": [
                    {
                        "uuid": "GPU-A",
                        "id": 0,
                        "vram_total": "oops",
                        "vram_used": None,
                        "vram_free": 50,
                        "utilization": None,
                    }
                ]
            }
        }
        await r.set(
            redis_keys.metrics_agent_partial_key("mm-l2", ids["a"]),
            json.dumps(partial),
            ex=60,
        )
        merged = await metrics_service.read_machine_metrics(r, "mm-l2")
        assert merged["vram"]["total_bytes"] == 0
        assert merged["vram"]["used_bytes"] == 0
        assert merged["vram"]["free_bytes"] == 50
        assert merged["gpu_usage"]["utilization_percent"] == 0.0
    finally:
        await r.aclose()

"""Machine metrics ownership: assign on connect, single owner per machine,
snapshot storage from owner only, release on disconnect, takeover after
the owner leaves, no assignment without declared categories."""

import json
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import settings
from app.models import Machine, ProviderDefinition, ProviderInstance
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

    async def fake_send_command(instance_id, _type_, _payload, timeout=30.0):  # noqa: ARG001
        calls.append((instance_id, dict(_payload)))
        return Frame(type="ack", reply_to="x", payload={"ok": True})

    monkeypatch.setattr(metrics_service.manager, "send_command", fake_send_command)
    return calls


def _owner(clean_redis, machine_uid: str) -> str | None:
    return clean_redis.get(redis_keys.metrics_owner_key(machine_uid))


def _make_stack(session: Session, machine_uid: str) -> None:
    if session.query(Machine).filter(Machine.uid == machine_uid).first() is None:
        session.add(Machine(uid=machine_uid, name=f"name-{machine_uid}"))
    for suffix in ("a", "b"):
        alias = f"alias-{machine_uid}-{suffix}"
        token = f"tok-{machine_uid}-{suffix}"
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
                    registration_token=token,
                    backend_config={"model": {"file": "m.gguf"}},
                )
            )
    session.commit()


def _register(
    client: TestClient, machine_uid: str, suffix: str, categories: list[str]
) -> dict:
    resp = client.post(
        "/admin/api/providers/register",
        json={
            "machine_uid": machine_uid,
            "registration_token": f"tok-{machine_uid}-{suffix}",
            "provider_type": "mock",
            "schema": {"type": "object"},
            "version": settings.VERSION,
            "port": 8081,
            "hardware": {"gpus": [], "total_vram_bytes": 1},
            "metrics_categories": categories,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _connect(client: TestClient, reg: dict):
    return client.websocket_connect(
        f"/provider/ws?instance_id={reg['instance_id']}",
        headers={"Authorization": f"Bearer {reg['instance_secret']}"},
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
    reg = _register(client, "mm-reg", "a", ["cpu", "vram"])
    raw = clean_redis.get(redis_keys.metrics_cats_key(reg["instance_id"]))
    assert json.loads(raw) == ["cpu", "vram"]


def test_first_connected_instance_owns_machine(
    client: TestClient, session: Session, assign_calls, clean_redis
) -> None:
    _make_stack(session, "mm-one")
    reg_a = _register(client, "mm-one", "a", ["cpu"])
    reg_b = _register(client, "mm-one", "b", ["cpu"])

    with _connect(client, reg_a) as ws:
        hello = Frame.model_validate_json(ws.receive_text())
        assert hello.type == "provider.hello"

        _wait_for(lambda: len(assign_calls) >= 1)
        assigned_id, payload = assign_calls[0]
        assert assigned_id == reg_a["instance_id"]
        assert payload["machine_uid"] == "mm-one"
        assert payload["categories"] == ["cpu"]
        assert _owner(clean_redis, "mm-one") == reg_a["instance_id"]

        # B connects on the SAME machine: must NOT be assigned.
        with _connect(client, reg_b) as ws_b:
            ws_b.receive_text()  # hello
            time.sleep(0.2)
            assert [c[0] for c in assign_calls] == [reg_a["instance_id"]]

    # A disconnected: ownership released.
    _wait_for(lambda: _owner(clean_redis, "mm-one") is None)


def test_owner_snapshot_stored_nonowner_dropped(
    client: TestClient, session: Session, assign_calls, clean_redis
) -> None:
    _make_stack(session, "mm-snap")
    reg_a = _register(client, "mm-snap", "a", ["cpu"])
    reg_b = _register(client, "mm-snap", "b", ["cpu"])
    key = redis_keys.metrics_machine_key("mm-snap")

    with _connect(client, reg_a) as ws:
        hello = Frame.model_validate_json(ws.receive_text())
        _wait_for(lambda: len(assign_calls) >= 1)

        ws.send_text(
            Frame(
                type="metrics.machine",
                id="m-1",
                epoch=hello.epoch,
                payload={"cpu": {"cores": 8}, "backend_status": "running"},
            ).to_json()
        )
        _wait_for(lambda: clean_redis.get(key) is not None)
        assert json.loads(clean_redis.get(key))["cpu"]["cores"] == 8

    # Now B is connected while the lease still names A: B's snapshot drops.
    # Wait for A's disconnect cleanup to release the lease first, then
    # re-arm it as A so the ordering is deterministic.
    _wait_for(lambda: _owner(clean_redis, "mm-snap") is None)
    clean_redis.set(
        redis_keys.metrics_owner_key("mm-snap"), reg_a["instance_id"], ex=60
    )
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
        assert _owner(clean_redis, "mm-snap") == reg_a["instance_id"]


def test_takeover_after_owner_disconnects(
    client: TestClient, session: Session, assign_calls, clean_redis
) -> None:
    _make_stack(session, "mm-take")
    reg_a = _register(client, "mm-take", "a", ["cpu"])
    reg_b = _register(client, "mm-take", "b", ["vram"])

    with _connect(client, reg_a) as ws:
        ws.receive_text()
        _wait_for(lambda: len(assign_calls) >= 1)

    _wait_for(lambda: _owner(clean_redis, "mm-take") is None)

    with _connect(client, reg_b) as ws_b:
        ws_b.receive_text()
        _wait_for(lambda: len(assign_calls) >= 2)
        assert assign_calls[-1][0] == reg_b["instance_id"]
        assert assign_calls[-1][1]["categories"] == ["vram"]
        assert _owner(clean_redis, "mm-take") == reg_b["instance_id"]


def test_no_categories_means_no_assignment(
    client: TestClient,
    session: Session,
    assign_calls,
    clean_redis,  # noqa: ARG001
) -> None:
    _make_stack(session, "mm-none")
    reg = _register(client, "mm-none", "a", [])

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
        registration_token="tok-rb-a",
    )
    session.add_all([machine, definition])
    session.commit()
    instance = ProviderInstance(
        machine_id=machine.id, provider_definition_id=definition.id
    )
    session.add(instance)
    session.commit()
    instance_id = str(instance.id)

    aredis = aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    await aredis.set(redis_keys.metrics_cats_key(instance_id), '["cpu"]')

    async def boom(*_args, **_kwargs):  # noqa: ARG001
        raise ConnectionError("no live connection")

    monkeypatch.setattr(metrics_service.manager, "send_command", boom)
    assigned = await metrics_service.assign_ownership(aredis, instance_id)
    assert assigned is False
    assert await aredis.get(redis_keys.metrics_owner_key("mm-rb")) is None

    # And with a working command, ownership is acquired.
    async def ok(*_args, **_kwargs):  # noqa: ARG001
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(metrics_service.manager, "send_command", ok)
    assert await metrics_service.assign_ownership(aredis, instance_id) is True
    assert await aredis.get(redis_keys.metrics_owner_key("mm-rb")) == instance_id
    await aredis.aclose()

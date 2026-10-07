"""Provider agent WebSocket endpoint: auth, hello/epoch, ping/pong, status
persistence, epoch fencing, and disconnect handling (Phase 16)."""

import asyncio
import os
import time
import uuid
from types import SimpleNamespace

import pytest
import redis.asyncio as aioredis
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.models import Machine, ProviderAgent, ProviderDefinition, ProviderInstance
from app.services import redis_keys
from app.services.connection_manager import (
    ConnectionState,
    manager,
)
from app.services.presence_sweep import sweep_once
from app.services.wire import Frame, FrameKind, InstanceStatusValue


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def _register(client: TestClient, session: Session) -> dict:
    machine = Machine(
        uid="ws-mach",
        name="ws-machine",
        host="10.0.0.9",
        registration_secret="ws-secret",
    )
    definition = ProviderDefinition(
        alias="ws-model",
        provider_type="mock",
        backend_config={"model": {"file": "x.gguf"}},
    )
    session.add(machine)
    session.add(definition)
    session.commit()
    resp = client.post(
        "/admin/api/providers/register",
        json={
            "machine_uid": "ws-mach",
            "machine_secret": "ws-secret",
            "agent_id": "ws-agent",
            "provider_type": "mock",
            "schema": {"type": "object"},
            "version": settings.VERSION,
            "base_port": 8081,
            "hardware": {"gpus": [], "total_vram_bytes": 1000},
            "metrics_categories": [],
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _instance_id(reg: dict) -> str:
    return reg["backends"][0]["instance_id"]


def _ws_headers(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


def _recv_frame(ws) -> Frame:
    return Frame.model_validate_json(ws.receive_text())


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("condition not met in time")


def test_connect_hello_epoch_and_db_state(client: TestClient, session: Session) -> None:
    reg = _register(client, session)
    agent_id = reg["agent_id"]

    with client.websocket_connect(
        f"/provider/ws?agent_id={agent_id}",
        headers=_ws_headers(reg["agent_secret"]),
    ) as ws:
        hello = _recv_frame(ws)
        assert hello.type == FrameKind.PROVIDER_HELLO
        assert hello.epoch == 1
        assert hello.payload["epoch"] == 1
        assert "server_time" in hello.payload

        def connected() -> bool:
            session.expire_all()
            row = session.exec(
                select(ProviderAgent).where(ProviderAgent.id == uuid.UUID(agent_id))
            ).one()
            return row.websocket_connected is True and row.epoch == 1

        _wait_for(connected)

    # After disconnect the agent is marked disconnected.
    def disconnected() -> bool:
        session.expire_all()
        row = session.exec(
            select(ProviderAgent).where(ProviderAgent.id == uuid.UUID(agent_id))
        ).one()
        return row.websocket_connected is False and (
            row.agent_status == InstanceStatusValue.DISCONNECTED
        )

    _wait_for(disconnected)


def test_ping_pong(client: TestClient, session: Session) -> None:
    reg = _register(client, session)
    with client.websocket_connect(
        f"/provider/ws?agent_id={reg['agent_id']}",
        headers=_ws_headers(reg["agent_secret"]),
    ) as ws:
        hello = _recv_frame(ws)
        epoch = hello.epoch

        ping = Frame(type="ping", id="ping-1", epoch=epoch)
        ws.send_text(ping.to_json())
        pong = _recv_frame(ws)
        assert pong.type == "pong"
        assert pong.reply_to == "ping-1"
        assert pong.epoch == epoch


def test_status_events_persist(client: TestClient, session: Session) -> None:
    reg = _register(client, session)
    instance_id = _instance_id(reg)
    with client.websocket_connect(
        f"/provider/ws?agent_id={reg['agent_id']}",
        headers=_ws_headers(reg["agent_secret"]),
    ) as ws:
        hello = _recv_frame(ws)
        epoch = hello.epoch

        # provider.status -> agent_status; backend.status -> instance.
        ws.send_text(
            Frame(
                type="provider.status",
                id="ev-1",
                epoch=epoch,
                payload={"agent_status": "running"},
            ).to_json()
        )
        ws.send_text(
            Frame(
                type="backend.status",
                id="ev-2",
                epoch=epoch,
                payload={"instance_id": instance_id, "backend_status": "running"},
            ).to_json()
        )

        def running() -> bool:
            session.expire_all()
            agent = session.exec(
                select(ProviderAgent).where(
                    ProviderAgent.id == uuid.UUID(reg["agent_id"])
                )
            ).one()
            row = session.get(ProviderInstance, uuid.UUID(instance_id))
            return agent.agent_status == "running" and row.backend_status == "running"

        _wait_for(running)

        def loaded_clock_stamped() -> bool:
            session.expire_all()
            row = session.get(ProviderInstance, uuid.UUID(instance_id))
            return row.backend_loaded_at is not None

        # Entering the loaded set arms the idle-reaper clock at boot time.
        _wait_for(loaded_clock_stamped)
        loaded_once = session.get(
            ProviderInstance, uuid.UUID(instance_id)
        ).backend_loaded_at

        # in_use -> running (idle heartbeat after a slot release) must NOT
        # re-arm the clock: the backend has been loaded the whole time.
        ws.send_text(
            Frame(
                type="backend.status",
                id="ev-3",
                epoch=epoch,
                payload={"instance_id": instance_id, "backend_status": "in_use"},
            ).to_json()
        )
        ws.send_text(
            Frame(
                type="backend.status",
                id="ev-4",
                epoch=epoch,
                payload={"instance_id": instance_id, "backend_status": "running"},
            ).to_json()
        )

        def back_to_running() -> bool:
            session.expire_all()
            row = session.get(ProviderInstance, uuid.UUID(instance_id))
            return row.backend_status == "running"

        _wait_for(back_to_running)
        row = session.get(ProviderInstance, uuid.UUID(instance_id))
        assert row.backend_loaded_at == loaded_once

        # Leaving the loaded set clears the clock; the next boot re-arms it.
        ws.send_text(
            Frame(
                type="backend.status",
                id="ev-5",
                epoch=epoch,
                payload={"instance_id": instance_id, "backend_status": "stopped"},
            ).to_json()
        )

        def stopped() -> bool:
            session.expire_all()
            row = session.get(ProviderInstance, uuid.UUID(instance_id))
            return row.backend_status == "stopped" and row.backend_loaded_at is None

        _wait_for(stopped)


def test_cross_agent_backend_status_ignored(client: TestClient, session: Session) -> None:
    """LOW (round 2): a backend.status/metadata frame naming an instance owned
    by a DIFFERENT agent is ignored — an agent may only mutate its own
    backends."""
    reg = _register(client, session)
    # Build a second agent + backend on the same machine, owned by another
    # agent id (not the one we connect as).
    machine = session.exec(select(Machine).where(Machine.uid == "ws-mach")).one()
    definition = session.exec(
        select(ProviderDefinition).where(ProviderDefinition.alias == "ws-model")
    ).one()
    other_agent = ProviderAgent(
        machine_id=machine.id,
        provider_type="mock",
        agent_id="ws-agent-other",
        base_port=8091,
    )
    session.add(other_agent)
    session.commit()
    other_inst = ProviderInstance(
        agent_id=other_agent.id,
        provider_definition_id=definition.id,
        port=8091,
        backend_status="stopped",
    )
    session.add(other_inst)
    session.commit()
    other_id = str(other_inst.id)

    with client.websocket_connect(
        f"/provider/ws?agent_id={reg['agent_id']}",
        headers=_ws_headers(reg["agent_secret"]),
    ) as ws:
        epoch = _recv_frame(ws).epoch
        ws.send_text(
            Frame(
                type="backend.status",
                id="x-1",
                epoch=epoch,
                payload={"instance_id": other_id, "backend_status": "running"},
            ).to_json()
        )
        ws.send_text(
            Frame(
                type="backend.metadata",
                id="x-2",
                epoch=epoch,
                payload={"instance_id": other_id, "models": [{"id": "m", "object": "model"}]},
            ).to_json()
        )
        # Sync on a frame we own so the server has processed the prior two.
        ws.send_text(Frame(type="ping", id="sync", epoch=epoch).to_json())
        _recv_frame(ws)

    session.expire_all()
    row = session.get(ProviderInstance, uuid.UUID(other_id))
    assert row.backend_status == "stopped"  # never mutated by the other agent


def test_epoch_fencing_drops_stale_frames(client: TestClient, session: Session) -> None:
    reg = _register(client, session)
    agent_id = reg["agent_id"]
    with client.websocket_connect(
        f"/provider/ws?agent_id={agent_id}",
        headers=_ws_headers(reg["agent_secret"]),
    ) as ws:
        hello = _recv_frame(ws)
        epoch = hello.epoch

        # Stale epoch (from a previous connection) must be dropped.
        ws.send_text(
            Frame(
                type="provider.status",
                id="stale-1",
                epoch=epoch - 1,
                payload={"agent_status": "error"},
            ).to_json()
        )
        # Future epoch must also be dropped.
        ws.send_text(
            Frame(
                type="provider.status",
                id="future-1",
                epoch=epoch + 7,
                payload={"agent_status": "error"},
            ).to_json()
        )
        # Valid epoch goes through.
        ws.send_text(
            Frame(
                type="provider.status",
                id="good-1",
                epoch=epoch,
                payload={"agent_status": "running"},
            ).to_json()
        )
        # Keep the socket busy so the pongs/processing order is deterministic,
        # then check the DB never went to 'error'.
        ws.send_text(Frame(type="ping", id="sync-1", epoch=epoch).to_json())
        pong = _recv_frame(ws)
        assert pong.type == "pong"

        def running() -> bool:
            session.expire_all()
            row = session.exec(
                select(ProviderAgent).where(ProviderAgent.id == uuid.UUID(agent_id))
            ).one()
            return row.agent_status == "running"

        _wait_for(running)
        session.expire_all()
        row = session.exec(
            select(ProviderAgent).where(ProviderAgent.id == uuid.UUID(agent_id))
        ).one()
        assert row.agent_status != "error"


def test_wrong_secret_rejected(client: TestClient, session: Session) -> None:
    from starlette.websockets import WebSocketDisconnect

    reg = _register(client, session)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
            f"/provider/ws?agent_id={reg['agent_id']}",
            headers=_ws_headers("not-the-secret"),
        ) as ws:
            ws.receive_text()
    assert exc.value.code == 4401


def test_missing_agent_id_rejected(client: TestClient, session: Session) -> None:
    from starlette.websockets import WebSocketDisconnect

    reg = _register(client, session)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
            "/provider/ws", headers=_ws_headers(reg["agent_secret"])
        ) as ws:
            ws.receive_text()
    assert exc.value.code == 4401


def test_unknown_agent_id_rejected(client: TestClient) -> None:
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
            f"/provider/ws?agent_id={uuid.uuid4()}",
            headers=_ws_headers("whatever"),
        ) as ws:
            ws.receive_text()
    assert exc.value.code == 4401


def test_reconnect_bumps_epoch(client: TestClient, session: Session) -> None:
    reg = _register(client, session)
    agent_id = reg["agent_id"]

    with client.websocket_connect(
        f"/provider/ws?agent_id={agent_id}",
        headers=_ws_headers(reg["agent_secret"]),
    ) as ws:
        first = _recv_frame(ws)
        assert first.epoch == 1

    with client.websocket_connect(
        f"/provider/ws?agent_id={agent_id}",
        headers=_ws_headers(reg["agent_secret"]),
    ) as ws:
        second = _recv_frame(ws)
        assert second.epoch == 2

    def final_state() -> bool:
        session.expire_all()
        row = session.exec(
            select(ProviderAgent).where(ProviderAgent.id == uuid.UUID(agent_id))
        ).one()
        return row.epoch == 2 and row.websocket_connected is False

    _wait_for(final_state)


def test_disconnect_clears_and_reconnect_rearms_load_clock(
    client: TestClient, session: Session
) -> None:
    """A dead socket means "backend state unknown": the load clock must be
    cleared so a container restart that re-reports `running` over the new
    socket re-arms the idle-reaper window at the real (re)boot time —
    never against the previous boot's stamp."""
    reg = _register(client, session)
    agent_id = reg["agent_id"]
    instance_id = _instance_id(reg)

    def row() -> ProviderInstance:
        session.expire_all()
        return session.get(ProviderInstance, uuid.UUID(instance_id))

    with client.websocket_connect(
        f"/provider/ws?agent_id={agent_id}",
        headers=_ws_headers(reg["agent_secret"]),
    ) as ws:
        hello = _recv_frame(ws)
        ws.send_text(
            Frame(
                type="backend.status",
                id="clock-1",
                epoch=hello.epoch,
                payload={"instance_id": instance_id, "backend_status": "running"},
            ).to_json()
        )
        _wait_for(lambda: row().backend_loaded_at is not None)
    loaded_first = row().backend_loaded_at

    # Socket closed -> disconnect handling clears the clock.
    _wait_for(lambda: row().backend_loaded_at is None)

    with client.websocket_connect(
        f"/provider/ws?agent_id={agent_id}",
        headers=_ws_headers(reg["agent_secret"]),
    ) as ws:
        hello2 = _recv_frame(ws)
        # The provider re-reports the CURRENT (still-running-in-the-new-
        # container) backend: DB still says running, clock is NULL ->
        # must re-arm.
        ws.send_text(
            Frame(
                type="backend.status",
                id="clock-2",
                epoch=hello2.epoch,
                payload={"instance_id": instance_id, "backend_status": "running"},
            ).to_json()
        )
        _wait_for(
            lambda: (
                row().backend_loaded_at is not None
                and row().backend_loaded_at != loaded_first
            )
        )


# ---------------------------------------------------------------------------
# Connection manager unit tests (fake socket, no HTTP layer)
# ---------------------------------------------------------------------------


class FakeWebSocket:
    def __init__(self, app_stub) -> None:  # type: ignore[type-arg]
        self.app = app_stub
        self.sent: list[str] = []
        self.closed_code: int | None = None

    async def send_text(self, data: str) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000) -> None:
        self.closed_code = code


async def test_send_command_awaits_ack() -> None:
    aredis = aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    app_stub = SimpleNamespace(state=SimpleNamespace(redis=aredis))
    ws = FakeWebSocket(app_stub)
    agent_id = str(uuid.uuid4())
    state = ConnectionState(
        agent_id=agent_id,
        websocket=ws,
        epoch=5,
        connection_token="tok-1",
    )
    manager._connections[agent_id] = state
    try:
        task = asyncio.create_task(
            manager.send_command(
                agent_id, "backend.start", {"instance_id": "i1", "x": 1}, timeout=2.0
            )
        )
        while not ws.sent:
            await asyncio.sleep(0.01)
        sent = Frame.model_validate_json(ws.sent[0])
        assert sent.type == "backend.start"
        assert sent.epoch == 5

        ack = Frame(
            type="ack",
            id="ack-1",
            reply_to=sent.id,
            epoch=5,
            payload={"ok": True, "error": None, "detail": {}},
        )
        await manager.handle_inbound(app_stub, state, ack)
        reply = await task
        assert reply.type == "ack"
        assert reply.payload["ok"] is True
        assert state.pending == {}
    finally:
        manager._connections.pop(agent_id, None)
        await aredis.aclose()


async def test_send_command_timeout_without_ack() -> None:
    aredis = aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    app_stub = SimpleNamespace(state=SimpleNamespace(redis=aredis))
    ws = FakeWebSocket(app_stub)
    agent_id = str(uuid.uuid4())
    state = ConnectionState(
        agent_id=agent_id,
        websocket=ws,
        epoch=1,
        connection_token="tok-2",
    )
    manager._connections[agent_id] = state
    try:
        with pytest.raises(asyncio.TimeoutError):
            await manager.send_command(
                agent_id, "backend.stop", {"instance_id": "i1"}, timeout=0.1
            )
        assert state.pending == {}
    finally:
        manager._connections.pop(agent_id, None)
        await aredis.aclose()


async def test_sweep_marks_presence_expired(session: Session) -> None:
    aredis = aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    try:
        machine = Machine(uid="sweep-m", name="sweep-machine")
        definition = ProviderDefinition(
            alias="sweep-model",
            provider_type="mock",
            backend_config={"model": {"file": "m.gguf"}},
        )
        session.add_all([machine, definition])
        session.commit()

        live = ProviderAgent(
            machine_id=machine.id,
            provider_type="mock",
            agent_id="live-agent",
            websocket_connected=True,
            agent_status="running",
        )
        stale = ProviderAgent(
            machine_id=machine.id,
            provider_type="mock",
            agent_id="stale-agent",
            websocket_connected=True,
            agent_status="running",
        )
        session.add_all([live, stale])
        session.commit()
        # one backend under the stale agent, to confirm its clock clears
        stale_inst = ProviderInstance(
            agent_id=stale.id,
            provider_definition_id=definition.id,
            backend_status="running",
        )
        stale_inst.backend_loaded_at = __import__("datetime").datetime.now(
            __import__("datetime").UTC
        )
        session.add(stale_inst)
        session.commit()

        # `live` has a presence key; `stale` does not.
        await aredis.set(redis_keys.presence_key(str(live.id)), "x", ex=60)

        marked = await sweep_once(aredis)
        assert marked == [str(stale.id)]

        session.expire_all()
        live_row = session.get(ProviderAgent, live.id)
        stale_row = session.get(ProviderAgent, stale.id)
        assert live_row.websocket_connected is True
        assert stale_row.websocket_connected is False
        assert stale_row.agent_status == InstanceStatusValue.DISCONNECTED
        inst_row = session.get(ProviderInstance, stale_inst.id)
        assert inst_row.backend_loaded_at is None
    finally:
        await aredis.aclose()

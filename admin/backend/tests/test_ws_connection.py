"""Provider WebSocket endpoint: auth, hello/epoch, ping/pong, status
persistence, epoch fencing, and disconnect handling."""

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
from app.models import Machine, ProviderDefinition, ProviderInstance
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
    machine = Machine(uid="ws-mach", name="ws-machine", host="10.0.0.9")
    definition = ProviderDefinition(
        alias="ws-model",
        provider_type="mock",
        registration_token="ws-token",
        backend_config={"model": {"file": "x.gguf"}},
    )
    session.add(machine)
    session.add(definition)
    session.commit()
    resp = client.post(
        "/admin/api/providers/register",
        json={
            "machine_uid": "ws-mach",
            "registration_token": "ws-token",
            "provider_type": "mock",
            "schema": {"type": "object"},
            "version": settings.VERSION,
            "port": 8081,
            "hardware": {"gpus": [], "total_vram_bytes": 1000},
            "metrics_categories": [],
        },
    )
    assert resp.status_code == 200
    return resp.json()


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
    instance_id = reg["instance_id"]

    with client.websocket_connect(
        f"/provider/ws?instance_id={instance_id}",
        headers=_ws_headers(reg["instance_secret"]),
    ) as ws:
        hello = _recv_frame(ws)
        assert hello.type == FrameKind.PROVIDER_HELLO
        assert hello.epoch == 1
        assert hello.payload["epoch"] == 1
        assert "server_time" in hello.payload

        def connected() -> bool:
            session.expire_all()
            row = session.exec(
                select(ProviderInstance).where(
                    ProviderInstance.id == uuid.UUID(instance_id)
                )
            ).one()
            return row.websocket_connected is True and row.epoch == 1

        _wait_for(connected)

    # After disconnect the row is marked disconnected.
    def disconnected() -> bool:
        session.expire_all()
        row = session.exec(
            select(ProviderInstance).where(
                ProviderInstance.id == uuid.UUID(instance_id)
            )
        ).one()
        return row.websocket_connected is False and (
            row.instance_status == InstanceStatusValue.DISCONNECTED
        )

    _wait_for(disconnected)


def test_ping_pong(client: TestClient, session: Session) -> None:
    reg = _register(client, session)
    with client.websocket_connect(
        f"/provider/ws?instance_id={reg['instance_id']}",
        headers=_ws_headers(reg["instance_secret"]),
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
    instance_id = reg["instance_id"]
    with client.websocket_connect(
        f"/provider/ws?instance_id={instance_id}",
        headers=_ws_headers(reg["instance_secret"]),
    ) as ws:
        hello = _recv_frame(ws)
        epoch = hello.epoch

        ws.send_text(
            Frame(
                type="provider.status",
                id="ev-1",
                epoch=epoch,
                payload={
                    "instance_status": "running",
                    "backend_status": "starting",
                },
            ).to_json()
        )
        ws.send_text(
            Frame(
                type="backend.status",
                id="ev-2",
                epoch=epoch,
                payload={"backend_status": "running"},
            ).to_json()
        )

        def running() -> bool:
            session.expire_all()
            row = session.exec(
                select(ProviderInstance).where(
                    ProviderInstance.id == uuid.UUID(instance_id)
                )
            ).one()
            return row.instance_status == "running" and row.backend_status == "running"

        _wait_for(running)


def test_epoch_fencing_drops_stale_frames(client: TestClient, session: Session) -> None:
    reg = _register(client, session)
    instance_id = reg["instance_id"]
    with client.websocket_connect(
        f"/provider/ws?instance_id={instance_id}",
        headers=_ws_headers(reg["instance_secret"]),
    ) as ws:
        hello = _recv_frame(ws)
        epoch = hello.epoch

        # Stale epoch (from a previous connection) must be dropped.
        ws.send_text(
            Frame(
                type="provider.status",
                id="stale-1",
                epoch=epoch - 1,
                payload={"instance_status": "error"},
            ).to_json()
        )
        # Future epoch must also be dropped.
        ws.send_text(
            Frame(
                type="provider.status",
                id="future-1",
                epoch=epoch + 7,
                payload={"instance_status": "error"},
            ).to_json()
        )
        # Valid epoch goes through.
        ws.send_text(
            Frame(
                type="provider.status",
                id="good-1",
                epoch=epoch,
                payload={"instance_status": "running"},
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
                select(ProviderInstance).where(
                    ProviderInstance.id == uuid.UUID(instance_id)
                )
            ).one()
            return row.instance_status == "running"

        _wait_for(running)
        session.expire_all()
        row = session.exec(
            select(ProviderInstance).where(
                ProviderInstance.id == uuid.UUID(instance_id)
            )
        ).one()
        assert row.instance_status != "error"


def test_wrong_secret_rejected(client: TestClient, session: Session) -> None:
    from starlette.websockets import WebSocketDisconnect

    reg = _register(client, session)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
            f"/provider/ws?instance_id={reg['instance_id']}",
            headers=_ws_headers("not-the-secret"),
        ) as ws:
            ws.receive_text()
    assert exc.value.code == 4401


def test_missing_instance_id_rejected(client: TestClient, session: Session) -> None:
    from starlette.websockets import WebSocketDisconnect

    reg = _register(client, session)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
            "/provider/ws", headers=_ws_headers(reg["instance_secret"])
        ) as ws:
            ws.receive_text()
    assert exc.value.code == 4401


def test_unknown_instance_id_rejected(client: TestClient) -> None:
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
            f"/provider/ws?instance_id={uuid.uuid4()}",
            headers=_ws_headers("whatever"),
        ) as ws:
            ws.receive_text()
    assert exc.value.code == 4401


def test_reconnect_bumps_epoch(client: TestClient, session: Session) -> None:
    reg = _register(client, session)
    instance_id = reg["instance_id"]

    with client.websocket_connect(
        f"/provider/ws?instance_id={instance_id}",
        headers=_ws_headers(reg["instance_secret"]),
    ) as ws:
        first = _recv_frame(ws)
        assert first.epoch == 1

    with client.websocket_connect(
        f"/provider/ws?instance_id={instance_id}",
        headers=_ws_headers(reg["instance_secret"]),
    ) as ws:
        second = _recv_frame(ws)
        assert second.epoch == 2

    def final_state() -> bool:
        session.expire_all()
        row = session.exec(
            select(ProviderInstance).where(
                ProviderInstance.id == uuid.UUID(instance_id)
            )
        ).one()
        return row.epoch == 2 and row.websocket_connected is False

    _wait_for(final_state)


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
    instance_id = str(uuid.uuid4())
    state = ConnectionState(
        instance_id=instance_id,
        websocket=ws,
        epoch=5,
        connection_token="tok-1",
    )
    manager._connections[instance_id] = state
    try:
        task = asyncio.create_task(
            manager.send_command(instance_id, "backend.start", {"x": 1}, timeout=2.0)
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
        manager._connections.pop(instance_id, None)
        await aredis.aclose()


async def test_send_command_timeout_without_ack() -> None:
    aredis = aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    app_stub = SimpleNamespace(state=SimpleNamespace(redis=aredis))
    ws = FakeWebSocket(app_stub)
    instance_id = str(uuid.uuid4())
    state = ConnectionState(
        instance_id=instance_id,
        websocket=ws,
        epoch=1,
        connection_token="tok-2",
    )
    manager._connections[instance_id] = state
    try:
        with pytest.raises(asyncio.TimeoutError):
            await manager.send_command(instance_id, "backend.stop", {}, timeout=0.1)
        assert state.pending == {}
    finally:
        manager._connections.pop(instance_id, None)
        await aredis.aclose()


async def test_sweep_marks_presence_expired(session: Session) -> None:
    aredis = aioredis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    try:
        machine = Machine(uid="sweep-m", name="sweep-machine")
        definition = ProviderDefinition(
            alias="sweep-model",
            provider_type="mock",
            registration_token="sweep-token",
            backend_config={"model": {"file": "m.gguf"}},
        )
        definition2 = ProviderDefinition(
            alias="sweep-model-2",
            provider_type="mock",
            registration_token="sweep-token-2",
            backend_config={"model": {"file": "m.gguf"}},
        )
        session.add_all([machine, definition, definition2])
        session.commit()

        live = ProviderInstance(
            machine_id=machine.id,
            provider_definition_id=definition.id,
            websocket_connected=True,
            instance_status="running",
        )
        stale = ProviderInstance(
            machine_id=machine.id,
            provider_definition_id=definition2.id,
            websocket_connected=True,
            instance_status="running",
        )
        session.add_all([live, stale])
        session.commit()

        # `live` has a presence key; `stale` does not.
        await aredis.set(redis_keys.presence_key(str(live.id)), "x", ex=60)

        marked = await sweep_once(aredis)
        assert marked == [str(stale.id)]

        session.expire_all()
        live_row = session.get(ProviderInstance, live.id)
        stale_row = session.get(ProviderInstance, stale.id)
        assert live_row.websocket_connected is True
        assert stale_row.websocket_connected is False
        assert stale_row.instance_status == InstanceStatusValue.DISCONNECTED
    finally:
        await aredis.aclose()


# ============================================================================
# Phase 14 — awaiting_config
# ============================================================================


def _register_shell(client: TestClient, session: Session) -> dict:
    machine = Machine(uid="sh-ws-mach", name="sh-ws-machine", host="10.0.0.9")
    definition = ProviderDefinition(
        alias="sh-ws-model",
        provider_type=None,  # shell
        registration_token="sh-ws-token",
        backend_config=None,
    )
    session.add(machine)
    session.add(definition)
    session.commit()
    resp = client.post(
        "/admin/api/providers/register",
        json={
            "machine_uid": "sh-ws-mach",
            "registration_token": "sh-ws-token",
            "provider_type": "mock",
            "schema": {"type": "object"},
            "version": settings.VERSION,
            "port": 8081,
            "hardware": {},
            "metrics_categories": [],
        },
    )
    assert resp.status_code == 200
    return resp.json()


def test_shell_instance_reports_awaiting_config(
    client: TestClient, session: Session
) -> None:
    """After connect, an unconfigured definition's instance is
    `awaiting_config` (admin-owned), and a provider running-status frame
    does NOT clobber it."""
    reg = _register_shell(client, session)
    instance_id = reg["instance_id"]

    with client.websocket_connect(
        f"/provider/ws?instance_id={instance_id}",
        headers=_ws_headers(reg["instance_secret"]),
    ) as ws:
        _recv_frame(ws)  # hello
        epoch = 1

        # Provider (ignorant of the pre-state) claims running.
        ws.send_text(
            Frame(
                type=FrameKind.PROVIDER_STATUS,
                id="ev-sh-1",
                epoch=epoch,
                payload={
                    "instance_status": InstanceStatusValue.RUNNING,
                    "backend_status": "running",
                },
            ).to_json()
        )

        def stayed_awaiting() -> bool:
            session.expire_all()
            row = session.get(ProviderInstance, uuid.UUID(instance_id))
            return (
                row.instance_status == InstanceStatusValue.AWAITING_CONFIG
                and row.backend_status == "running"
            )

        _wait_for(stayed_awaiting)

        # An error payload stays honest (not coerced).
        ws.send_text(
            Frame(
                type=FrameKind.PROVIDER_STATUS,
                id="ev-sh-2",
                epoch=epoch,
                payload={"instance_status": InstanceStatusValue.ERROR},
            ).to_json()
        )

        def honest_error() -> bool:
            session.expire_all()
            row = session.get(ProviderInstance, uuid.UUID(instance_id))
            return row.instance_status == InstanceStatusValue.ERROR

        _wait_for(honest_error)

    session.expunge_all()


def test_configured_instance_connect_does_not_show_awaiting_config(
    client: TestClient, session: Session
) -> None:
    """Today's behavior unchanged: a configured definition's instance goes
    registering → running via provider.status with no coercion."""
    reg = _register(client, session)
    instance_id = reg["instance_id"]

    with client.websocket_connect(
        f"/provider/ws?instance_id={instance_id}",
        headers=_ws_headers(reg["instance_secret"]),
    ) as ws:
        hello = _recv_frame(ws)
        epoch = hello.epoch

        def connecting() -> bool:
            session.expire_all()
            row = session.get(ProviderInstance, uuid.UUID(instance_id))
            return row.websocket_connected is True and (
                row.instance_status != InstanceStatusValue.AWAITING_CONFIG
            )

        _wait_for(connecting)

        ws.send_text(
            Frame(
                type=FrameKind.PROVIDER_STATUS,
                id="ev-cfg-1",
                epoch=epoch,
                payload={"instance_status": InstanceStatusValue.RUNNING},
            ).to_json()
        )

        def running() -> bool:
            session.expire_all()
            row = session.get(ProviderInstance, uuid.UUID(instance_id))
            return row.instance_status == InstanceStatusValue.RUNNING

        _wait_for(running)

"""Provider WebSocket dial-in endpoint.

Mounted at the app root: ``/provider/ws`` (NOT under /admin/api).

Handshake (see docs/ws-protocol.md), Phase 16 machine-scoped agents:
  1. Agent connects with ``Authorization: Bearer <agent_secret>`` (minted at
     registration) and ``?agent_id=<uuid>`` query param.
  2. Admin verifies the secret against Redis (``im:ws:secret:{agent_id}``);
     mismatch/missing closes with code 4401 before accept.
  3. Admin INCRs ``im:ws:epoch:{agent_id}`` and sends the hello frame
     ``Frame(type="provider.hello", payload={epoch, server_time})``.
  4. Admin marks the ProviderAgent row connected (websocket_connected=True,
     epoch=<new>) and claims Redis ownership/presence.
  5. Recv loop: epoch-fence inbound frames; answer ping with pong; persist
     provider.status (agent) / backend.status (instance); refresh presence TTL.
  6. On disconnect: if still the current connection, mark the agent and its
     backends disconnected. Epoch is kept (monotonic).
"""

import asyncio
import contextlib
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import anyio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select

from app.core.db import engine
from app.core.redis import RedisNotInitialized
from app.models import ProviderAgent, ProviderInstance
from app.services import metrics_service
from app.services.connection_manager import manager
from app.services.wire import (
    WS_CLOSE_AUTH_FAILED,
    Frame,
    FrameKind,
    InstanceStatusValue,
    now_iso,
)

logger = logging.getLogger("admin.ws")

WS_CLOSE_SERVER_ERROR = 1011
WS_CLOSE_TRY_LATER = 1013

router = APIRouter(tags=["provider-ws"])


def _mark_connected(agent_id: str, epoch: int) -> None:
    with Session(engine) as session:
        agent = session.exec(
            select(ProviderAgent).where(ProviderAgent.id == uuid.UUID(agent_id))
        ).first()
        if agent is None:
            logger.warning("ws auth for unknown agent %s", agent_id)
            return
        now = datetime.now(UTC)
        agent.websocket_connected = True
        agent.epoch = epoch
        agent.last_seen = now
        agent.updated_at = now
        session.add(agent)
        session.commit()


def _mark_disconnected(agent_id: str) -> None:
    with Session(engine) as session:
        agent = session.exec(
            select(ProviderAgent).where(ProviderAgent.id == uuid.UUID(agent_id))
        ).first()
        if agent is None:
            return
        now = datetime.now(UTC)
        agent.websocket_connected = False
        agent.agent_status = InstanceStatusValue.DISCONNECTED
        agent.last_seen = now
        agent.updated_at = now
        session.add(agent)
        # Backend state is unknown once the socket is gone (a container
        # restart may leave the DB mirror saying "running" for a process
        # that died). Drop each backend's idle-reaper load clock so the next
        # loaded status report re-arms it at the real boot time.
        backends = session.exec(
            select(ProviderInstance).where(ProviderInstance.agent_id == agent.id)
        ).all()
        for inst in backends:
            inst.backend_loaded_at = None
            session.add(inst)
        session.commit()


async def _assign_metrics_owner(app: Any, agent_id: str) -> None:
    """Background metrics-ownership assignment (never breaks the WS flow)."""
    with contextlib.suppress(Exception):
        await metrics_service.assign_ownership(app, agent_id)


async def _heal_config_fingerprint(app: Any, agent_id: str) -> None:  # noqa: ARG001
    """Background stale-fingerprint self-heal (Phase 9).

    If any of the agent's backends has a stored ``config_fingerprint`` that
    lags its definition's current ``backend_config`` (admin PATCHed while the
    agent was disconnected), push ``provider.config.update`` now. Never
    breaks the WS flow; mismatches are also caught by the presence sweep.
    """
    with contextlib.suppress(Exception):
        from app.services.config_update import heal_agent_stale_fingerprints

        await heal_agent_stale_fingerprints(agent_id)


@router.websocket("/provider/ws")
async def provider_ws(websocket: WebSocket) -> None:
    agent_id = websocket.query_params.get("agent_id", "")

    try:
        verified = await manager.authenticate(websocket, agent_id)
    except RedisNotInitialized:
        logger.error("ws auth failed: redis not initialized")
        await websocket.close(code=WS_CLOSE_TRY_LATER)
        return
    if verified is None:
        await websocket.close(code=WS_CLOSE_AUTH_FAILED)
        return

    await websocket.accept()

    epoch = await manager.bump_epoch(websocket.app, agent_id)
    state = await manager.connect(websocket, agent_id, epoch)

    hello = Frame(
        type=FrameKind.PROVIDER_HELLO,
        epoch=epoch,
        payload={"epoch": epoch, "server_time": now_iso()},
    )
    await manager.send_frame(state, hello)
    await manager.claim_ownership(websocket.app, state)
    _mark_connected(agent_id, epoch)
    # Try to make this agent the machine-level metrics reporter. Run as a
    # background task so a slow provider ack never blocks the handshake.
    asyncio.create_task(_assign_metrics_owner(websocket.app, agent_id))
    # Phase 9 self-heal: a reconnect with a stale config_fingerprint
    # (admin PATCHed while disconnected) gets a fresh provider.config.update.
    asyncio.create_task(_heal_config_fingerprint(websocket.app, agent_id))
    logger.info("provider agent %s connected (epoch %s)", agent_id, epoch)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                frame = Frame.model_validate_json(raw)
            except Exception:  # noqa: BLE001
                logger.warning("malformed frame from %s: %r", agent_id, raw)
                continue
            await manager.handle_inbound(websocket.app, state, frame)
    except WebSocketDisconnect:
        logger.info("provider agent %s disconnected (epoch %s)", agent_id, epoch)
    finally:
        # Shield cleanup: the surrounding task may be cancelled (client close
        # / admin shutdown) and the disconnect bookkeeping must still land.
        with anyio.CancelScope(shield=True):
            was_current = await manager.disconnect(state)
            if was_current:
                _mark_disconnected(agent_id)
                with contextlib.suppress(Exception):
                    await metrics_service.release_ownership(websocket.app, agent_id)

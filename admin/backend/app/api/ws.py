"""Provider WebSocket dial-in endpoint.

Mounted at the app root: ``/provider/ws`` (NOT under /admin/api).

Handshake (see docs/ws-protocol.md):
  1. Provider connects with ``Authorization: Bearer <instance_secret>``
     (minted at registration) and ``?instance_id=<uuid>`` query param.
  2. Admin verifies the secret against Redis (``im:ws:secret:{id}``);
     mismatch/missing closes with code 4401 before accept.
  3. Admin INCRs ``im:ws:epoch:{id}`` and sends the hello frame
     ``Frame(type="provider.hello", payload={epoch, server_time})``.
  4. Admin marks the DB row connected (websocket_connected=True,
     epoch=<new>) and claims Redis ownership/presence.
  5. Recv loop: epoch-fence inbound frames; answer ping with pong; persist
     provider.status / backend.status; refresh presence TTL.
  6. On disconnect: if still the current connection, mark the DB row
     disconnected. Epoch is kept (monotonic).
"""

import logging
import uuid
from datetime import UTC, datetime

import anyio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlmodel import Session

from app.core.db import engine
from app.core.redis import RedisNotInitialized
from app.models import ProviderInstance
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


def _mark_connected(instance_id: str, epoch: int) -> None:
    with Session(engine) as session:
        inst = session.get(ProviderInstance, uuid.UUID(instance_id))
        if inst is None:
            logger.warning("ws auth for unknown instance %s", instance_id)
            return
        now = datetime.now(UTC)
        inst.websocket_connected = True
        inst.epoch = epoch
        inst.last_seen = now
        inst.updated_at = now
        session.add(inst)
        session.commit()


def _mark_disconnected(instance_id: str) -> None:
    with Session(engine) as session:
        inst = session.get(ProviderInstance, uuid.UUID(instance_id))
        if inst is None:
            return
        now = datetime.now(UTC)
        inst.websocket_connected = False
        inst.instance_status = InstanceStatusValue.DISCONNECTED
        inst.last_seen = now
        inst.updated_at = now
        session.add(inst)
        session.commit()


@router.websocket("/provider/ws")
async def provider_ws(websocket: WebSocket) -> None:
    instance_id = websocket.query_params.get("instance_id", "")

    try:
        verified = await manager.authenticate(websocket, instance_id)
    except RedisNotInitialized:
        logger.error("ws auth failed: redis not initialized")
        await websocket.close(code=WS_CLOSE_TRY_LATER)
        return
    if verified is None:
        await websocket.close(code=WS_CLOSE_AUTH_FAILED)
        return

    await websocket.accept()

    epoch = await manager.bump_epoch(websocket.app, instance_id)
    state = await manager.connect(websocket, instance_id, epoch)

    hello = Frame(
        type=FrameKind.PROVIDER_HELLO,
        epoch=epoch,
        payload={"epoch": epoch, "server_time": now_iso()},
    )
    await manager.send_frame(state, hello)
    await manager.claim_ownership(websocket.app, state)
    _mark_connected(instance_id, epoch)
    logger.info("provider instance %s connected (epoch %s)", instance_id, epoch)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                frame = Frame.model_validate_json(raw)
            except Exception:  # noqa: BLE001
                logger.warning("malformed frame from %s: %r", instance_id, raw)
                continue
            await manager.handle_inbound(websocket.app, state, frame)
    except WebSocketDisconnect:
        logger.info("provider instance %s disconnected (epoch %s)", instance_id, epoch)
    finally:
        # Shield cleanup: the surrounding task may be cancelled (client close
        # / admin shutdown) and the disconnect bookkeeping must still land.
        with anyio.CancelScope(shield=True):
            was_current = await manager.disconnect(state)
            if was_current:
                _mark_disconnected(instance_id)

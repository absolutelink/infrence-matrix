"""Admin-side provider WebSocket connection manager.

Tracks the live socket per provider instance so the admin (and later the
scheduler) can push commands and receive events. Liveness is mirrored in
Redis:

  - ``im:ws:presence:{instance_id}`` is refreshed (with TTL) on every
    accepted inbound frame and on every outbound command; the stale sweep
    task in the lifespan marks DB rows disconnected when it expires.
  - ``im:ws:epoch:{instance_id}`` is a monotonic INCR counter; the value
    assigned when a connection is accepted fences off frames from older,
    dead sockets.

This module owns the auth check (Bearer secret vs Redis), the hello frame,
frame dispatch/persistence, and the request/ack pattern for admin->provider
commands.
"""

import asyncio
import contextlib
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fastapi import WebSocket
from sqlmodel import Session

from app.core.db import engine
from app.core.redis import get_redis_from_app
from app.models import Machine, ProviderInstance
from app.services import redis_keys
from app.services.wire import Ack, BackendStatusValue, Frame, FrameKind, now_iso

logger = logging.getLogger("admin.connection_manager")

# Backend states that mean "definitely not loaded: holds no VRAM". A
# status frame in one of these states must prune the scheduler's
# per-booted-instance hold (out-of-band stop). "stopping" is NOT
# included: the weights are still resident until the stop completes, so
# pruning early would let a concurrent boot transiently oversubscribe
# the machine.
_NON_LOADED_BACKEND_STATUSES = frozenset(
    {
        BackendStatusValue.STOPPED,
        BackendStatusValue.ERROR,
    }
)

# TTL for the Redis presence key. The provider must send or receive traffic
# (pings keep it alive) more often than this. Must be an int (Redis ex=).
PRESENCE_TTL_SECONDS = 60

# Default timeout awaiting a provider ack for an admin command.
COMMAND_TIMEOUT_SECONDS = 30.0


@dataclass
class ConnectionState:
    """Live socket state for one accepted provider connection."""

    instance_id: str
    websocket: WebSocket
    epoch: int
    connection_token: str
    # frame id -> future resolved with the reply/ack frame
    pending: dict[str, asyncio.Future[Frame]] = field(default_factory=dict)
    closed: bool = False


class ConnectionManager:
    """In-process registry of live provider WebSocket connections."""

    def __init__(self) -> None:
        self._connections: dict[str, ConnectionState] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    async def authenticate(self, websocket: WebSocket, instance_id: str) -> str | None:
        """Verify the Bearer secret for ``instance_id`` against Redis.

        Returns the secret-compatible instance_id on success, or None if
        auth fails (missing header, unknown secret, mismatch).
        """
        auth = websocket.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return None
        presented = auth[len("bearer ") :].strip()
        if not presented or not instance_id:
            return None
        redis_client = get_redis_from_app(websocket.app)
        stored = await redis_client.get(redis_keys.secret_key(instance_id))
        if stored is None:
            return None
        if not secrets.compare_digest(str(stored), presented):
            return None
        return instance_id

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    async def connect(
        self, websocket: WebSocket, instance_id: str, epoch: int
    ) -> ConnectionState:
        state = ConnectionState(
            instance_id=instance_id,
            websocket=websocket,
            epoch=epoch,
            connection_token=uuid.uuid4().hex,
        )
        async with self._lock:
            previous = self._connections.get(instance_id)
            if previous is not None:
                previous.closed = True
                for fut in previous.pending.values():
                    if not fut.done():
                        fut.cancel()
                with contextlib.suppress(Exception):
                    await previous.websocket.close(code=4409)
            self._connections[instance_id] = state
        return state

    async def disconnect(self, state: ConnectionState) -> bool:
        """Remove the connection from the registry if it is still current.

        Returns True when ``state`` was the live connection for its instance
        (so the caller may mark the DB row disconnected); False when a newer
        connection already took over, in which case the caller must not
        clobber the newer state.
        """
        async with self._lock:
            current = self._connections.get(state.instance_id)
            was_current = current is state
            if was_current:
                self._connections.pop(state.instance_id, None)
                redis_client = get_redis_from_app(state.websocket.app)
                # Only clear ownership if we still own it (a newer connection
                # may already have taken over).
                owner = await redis_client.get(redis_keys.owner_key(state.instance_id))
                if owner == state.connection_token:
                    await redis_client.delete(redis_keys.owner_key(state.instance_id))
            state.closed = True
            for fut in state.pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("provider connection closed"))
            state.pending.clear()
        return was_current

    def get(self, instance_id: str) -> ConnectionState | None:
        state = self._connections.get(instance_id)
        if state is not None and not state.closed:
            return state
        return None

    # ------------------------------------------------------------------
    # Epoch / presence helpers
    # ------------------------------------------------------------------
    async def bump_epoch(self, app: Any, instance_id: str) -> int:
        redis_client = get_redis_from_app(app)
        epoch = int(await redis_client.incr(redis_keys.epoch_key(instance_id)))
        return epoch

    async def claim_ownership(self, app: Any, state: ConnectionState) -> None:
        redis_client = get_redis_from_app(app)
        await redis_client.set(
            redis_keys.owner_key(state.instance_id),
            state.connection_token,
        )
        await self.refresh_presence(app, state.instance_id)

    async def refresh_presence(self, app: Any, instance_id: str) -> None:
        redis_client = get_redis_from_app(app)
        await redis_client.set(
            redis_keys.presence_key(instance_id),
            now_iso(),
            ex=PRESENCE_TTL_SECONDS,
        )

    # ------------------------------------------------------------------
    # Inbound frame handling
    # ------------------------------------------------------------------
    async def handle_inbound(
        self, app: Any, state: ConnectionState, frame: Frame
    ) -> None:
        """Process one inbound provider frame with epoch fencing."""
        if frame.epoch < state.epoch:
            logger.warning(
                "dropping stale frame type=%s epoch=%s < %s (instance %s)",
                frame.type,
                frame.epoch,
                state.epoch,
                state.instance_id,
            )
            return
        if frame.epoch > state.epoch:
            logger.warning(
                "dropping frame from future epoch type=%s epoch=%s > %s (instance %s)",
                frame.type,
                frame.epoch,
                state.epoch,
                state.instance_id,
            )
            return

        await self.refresh_presence(app, state.instance_id)

        if frame.type == FrameKind.PING:
            await self.send_frame(
                state,
                Frame(
                    type=FrameKind.PONG,
                    id=str(uuid.uuid4()),
                    reply_to=frame.id or None,
                    epoch=state.epoch,
                ),
            )
            return

        if frame.type == FrameKind.PROVIDER_STATUS:
            machine_uid = self._persist_status(
                state.instance_id,
                instance_status=frame.payload.get("instance_status"),
                backend_status=frame.payload.get("backend_status"),
                error_message=frame.payload.get("error_message")
                or frame.payload.get("error"),
            )
            await self._prune_stopped_vram(
                app, state.instance_id, frame.payload.get("backend_status"), machine_uid
            )
            return

        if frame.type == FrameKind.METRICS_MACHINE:
            # Lazy import: metrics_service depends on this module's manager.
            from app.services import metrics_service

            await metrics_service.handle_machine_metrics(
                app, state.instance_id, frame.payload
            )
            return

        if frame.type == FrameKind.BACKEND_STATUS:
            reported = frame.payload.get("backend_status") or frame.payload.get(
                "status"
            )
            machine_uid = self._persist_status(
                state.instance_id,
                backend_status=reported,
                error_message=frame.payload.get("error_message")
                or frame.payload.get("error"),
            )
            await self._prune_stopped_vram(
                app, state.instance_id, reported, machine_uid
            )
            return

        if frame.type in (FrameKind.BACKEND_LOGS, FrameKind.PROVIDER_LOGS):
            # Phase 13: batched log tail ingest into Redis. Best-effort —
            # never let a malformed log frame break the WS read loop.
            from app.services import log_store

            kind = (
                log_store.KIND_BACKEND
                if frame.type == FrameKind.BACKEND_LOGS
                else log_store.KIND_PROVIDER
            )
            try:
                await log_store.ingest_log_batch(
                    app, state.instance_id, kind, frame.payload
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "log ingest failed for instance %s",
                    state.instance_id,
                    exc_info=True,
                )
            return

        # A reply to an outstanding admin command.
        if frame.reply_to:
            fut = state.pending.get(frame.reply_to)
            if fut is not None and not fut.done():
                fut.set_result(frame)
            return

        logger.debug(
            "ignoring unhandled inbound frame type=%s (instance %s)",
            frame.type,
            state.instance_id,
        )

    def _persist_status(
        self,
        instance_id: str,
        *,
        instance_status: str | None = None,
        backend_status: str | None = None,
        error_message: str | None = None,
    ) -> str | None:
        """Persist a status frame to the DB; returns the instance's
        machine uid (when the row exists) for VRAM-ledger pruning."""
        with Session(engine) as session:
            inst = session.get(ProviderInstance, _to_uuid(instance_id))
            if inst is None:
                logger.warning(
                    "status event for unknown instance %s; ignored", instance_id
                )
                return None
            if instance_status:
                inst.instance_status = instance_status
            if backend_status:
                inst.backend_status = backend_status
            if error_message is not None:
                inst.error_message = error_message
            inst.last_seen = datetime.now(UTC)
            inst.updated_at = datetime.now(UTC)
            session.add(inst)
            session.commit()
            machine = session.get(Machine, inst.machine_id)
            return machine.uid if machine is not None else None

    async def _prune_stopped_vram(
        self,
        app: Any,
        instance_id: str,
        reported_backend_status: str | None,
        machine_uid: str | None,
    ) -> None:
        """Drop the scheduler's per-booted-instance VRAM hold when a
        status frame says the backend is stopped/error (out-of-band stop:
        admin UI, provider-side stop, or crash-emitted status).

        Event-driven counterpart to the scheduler-initiated stop path;
        without it a ``_booted`` hold survives with the DB already
        ``stopped``, the reaper's TTL refresh keeps the stale mirror
        entry alive, and the machine stays permanently over-counted.
        Scheduler reached via ``app.state.scheduler`` (lazy import to
        avoid a scheduler <-> connection_manager import cycle); no-op
        when the scheduler isn't running or the instance isn't held.
        """
        if machine_uid is None or reported_backend_status is None:
            return
        if reported_backend_status not in _NON_LOADED_BACKEND_STATUSES:
            return
        scheduler = getattr(app.state, "scheduler", None)
        if scheduler is None:
            return
        with contextlib.suppress(Exception):
            await scheduler.note_backend_stopped(instance_id, machine_uid)

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------
    async def send_frame(self, state: ConnectionState, frame: Frame) -> None:
        await state.websocket.send_text(frame.to_json())

    async def send_event(
        self, instance_id: str, type_: str, payload: dict[str, Any]
    ) -> None:
        state = self.get(instance_id)
        if state is None:
            raise ConnectionError(f"no live connection for instance {instance_id}")
        await self.send_frame(
            state,
            Frame(
                type=type_,
                id=str(uuid.uuid4()),
                epoch=state.epoch,
                payload=payload,
            ),
        )

    async def send_command(
        self,
        instance_id: str,
        type_: str,
        payload: dict[str, Any],
        timeout: float = COMMAND_TIMEOUT_SECONDS,
    ) -> Frame:
        """Send an admin->provider command and await its ack frame."""
        state = self.get(instance_id)
        if state is None:
            raise ConnectionError(f"no live connection for instance {instance_id}")
        frame = Frame(
            type=type_,
            id=str(uuid.uuid4()),
            epoch=state.epoch,
            payload=payload,
        )
        fut: asyncio.Future[Frame] = asyncio.get_running_loop().create_future()
        state.pending[frame.id] = fut
        try:
            await self.send_frame(state, frame)
            return await asyncio.wait_for(fut, timeout)
        finally:
            state.pending.pop(frame.id, None)

    async def send_command_ack_ok(
        self,
        instance_id: str,
        type_: str,
        payload: dict[str, Any],
        timeout: float = COMMAND_TIMEOUT_SECONDS,
    ) -> Ack:
        reply = await self.send_command(instance_id, type_, payload, timeout)
        return Ack.model_validate(reply.payload)


def _to_uuid(value: str) -> uuid.UUID:
    return uuid.UUID(value)


manager = ConnectionManager()

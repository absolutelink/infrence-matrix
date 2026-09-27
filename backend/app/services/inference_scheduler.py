"""PostgreSQL-backed scheduling leases for inference requests."""

import asyncio
import hashlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import exists, func, or_, text, update
from sqlmodel import select

from app.db.session import AsyncSessionMaker
from app.db.session import engine as async_engine
from app.models import Agent, InferenceLease, Model, ServerInstance
from app.services.agent_manager import agent_manager
from app.services.scheduler_locks import BENCHMARK_ADVISORY_LOCK_KEY

logger = logging.getLogger(__name__)

LEASE_TIMEOUT_SECONDS = 30 * 60
ACTIVE_LEASE_TTL_SECONDS = 90
LEASE_RENEWAL_INTERVAL_SECONDS = 30
SCHEDULER_POLL_SECONDS = 1.0
TELEMETRY_TIMEOUT_SECONDS = 5.0
RECONCILIATION_INTERVAL_SECONDS = 30.0
DEFAULT_SINGLE_REQUEST_CAPACITY = 1
# Give llama-server a brief settling period before another request claims the
# slot. The lease stays active during this delay so other workers also wait.
UPSTREAM_COMPLETION_COOLDOWN_SECONDS = 0.5


def _vram_requirements_fit(
    total_vram: int, target_required: int, running_required: int
) -> bool:
    """Check configured VRAM requirements against physical capacity."""
    return running_required + target_required <= total_vram


def server_capacity(server: ServerInstance) -> int:
    """Return capacity confirmed by the agent for this server generation."""
    try:
        return max(int(getattr(server, "effective_capacity", 1)), 1)
    except TypeError, ValueError:
        return DEFAULT_SINGLE_REQUEST_CAPACITY


class InferenceLeaseLost(RuntimeError):
    """Raised when an active request no longer owns its server slot."""


@dataclass
class InferenceLeaseHandle:
    request_id: str
    server: ServerInstance
    lease_id: uuid.UUID
    slot_generation: int | None = None
    lost: asyncio.Event = field(default_factory=asyncio.Event)
    _upstream_started: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _renewal_task: asyncio.Task[None] | None = field(default=None, init=False)
    _disconnect_task: asyncio.Task[None] | None = field(default=None, init=False)

    def start_renewal(self) -> None:
        if self._renewal_task is None:
            self._renewal_task = asyncio.create_task(self._renew_loop())

    def mark_upstream_started(self) -> None:
        """Allow the renewal loop to renew a stream owned outside ``guard``."""
        self._upstream_started.set()

    def monitor_disconnect(
        self, is_cancelled: Callable[[], Awaitable[bool]] | None
    ) -> None:
        if is_cancelled is not None and self._disconnect_task is None:
            self._disconnect_task = asyncio.create_task(
                self._disconnect_loop(is_cancelled)
            )

    async def _disconnect_loop(
        self, is_cancelled: Callable[[], Awaitable[bool]]
    ) -> None:
        try:
            while True:
                if await is_cancelled():
                    self.lost.set()
                    await self._mark_cancelled()
                    return
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise

    async def _stop_disconnect_monitor(self) -> None:
        if self._disconnect_task is not None:
            if self._disconnect_task is not asyncio.current_task():
                self._disconnect_task.cancel()
                await asyncio.gather(self._disconnect_task, return_exceptions=True)
            self._disconnect_task = None

    async def guard(
        self,
        awaitable: Awaitable[Any],
        *,
        cancelled: asyncio.Event | None = None,
    ) -> Any:
        """Cancel one upstream operation when lease ownership or its client is lost."""
        self._upstream_started.set()
        operation = asyncio.ensure_future(awaitable)
        lease_lost = asyncio.create_task(self.lost.wait())
        client_lost = (
            asyncio.create_task(cancelled.wait()) if cancelled is not None else None
        )
        sentinels = {lease_lost}
        if client_lost is not None:
            sentinels.add(client_lost)
        try:
            done, _pending = await asyncio.wait(
                {operation, *sentinels}, return_when=asyncio.FIRST_COMPLETED
            )
            if operation in done and not any(
                task.result() for task in done & sentinels
            ):
                return await operation
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            if self.lost.is_set():
                raise InferenceLeaseLost("Inference lease ownership was lost")
            raise InferenceRequestCancelled
        finally:
            if not operation.done():
                operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)
            for task in sentinels:
                task.cancel()
            await asyncio.gather(*sentinels, return_exceptions=True)

    async def _renew_loop(self) -> None:
        expiry = asyncio.get_running_loop().time() + ACTIVE_LEASE_TTL_SECONDS
        try:
            while True:
                await asyncio.sleep(LEASE_RENEWAL_INTERVAL_SECONDS)
                try:
                    if not self._upstream_started.is_set():
                        if asyncio.get_running_loop().time() >= expiry:
                            self.lost.set()
                            return
                        continue
                    async with AsyncSessionMaker() as session:
                        healthy_server = exists(
                            select(ServerInstance.id).where(
                                ServerInstance.id == self.server.id,
                                ServerInstance.status == "running",
                                ServerInstance.health_status == "healthy",
                                ServerInstance.slot_generation == self.slot_generation,
                            )
                        )
                        result = await session.execute(
                            update(InferenceLease)
                            .where(
                                InferenceLease.id == self.lease_id,
                                InferenceLease.status == "active",
                                InferenceLease.slot_generation == self.slot_generation,
                                healthy_server,
                            )
                            .values(
                                lease_expires_at=datetime.now(UTC)
                                + timedelta(seconds=ACTIVE_LEASE_TTL_SECONDS)
                            )
                        )
                        await session.commit()
                    if result.rowcount == 0:
                        self.lost.set()
                        return
                    expiry = (
                        asyncio.get_running_loop().time() + ACTIVE_LEASE_TTL_SECONDS
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Unable to renew inference lease %s", self.lease_id
                    )
                    if asyncio.get_running_loop().time() >= expiry:
                        self.lost.set()
                        return
        except asyncio.CancelledError:
            pass

    async def release(self) -> None:
        """Release the lease in a short, independent transaction."""

        async def _release() -> None:
            await self._stop_disconnect_monitor()
            if self._renewal_task is not None:
                self._renewal_task.cancel()
                await asyncio.gather(self._renewal_task, return_exceptions=True)
                self._renewal_task = None
            await asyncio.sleep(UPSTREAM_COMPLETION_COOLDOWN_SECONDS)
            async with AsyncSessionMaker() as session:
                await session.execute(
                    update(InferenceLease)
                    .where(
                        InferenceLease.id == self.lease_id,
                        InferenceLease.status == "active",
                        InferenceLease.slot_generation == self.slot_generation,
                    )
                    .values(
                        status="released",
                        terminal_reason="completed",
                        released_at=datetime.now(UTC),
                    )
                )
                await session.commit()

        # ASGI cancels the request task when a client disconnects. Shield the
        # database update so cancellation cannot strand the slot as active.
        release_task = asyncio.create_task(_release())
        try:
            await asyncio.shield(release_task)
        except asyncio.CancelledError:
            await asyncio.shield(release_task)
            raise

    async def cancel(self) -> None:
        """Cancel an active lease when its client disconnects before dispatch."""

        async def _cancel() -> None:
            await self._stop_disconnect_monitor()
            if self._renewal_task is not None:
                self._renewal_task.cancel()
                await asyncio.gather(self._renewal_task, return_exceptions=True)
                self._renewal_task = None
            async with AsyncSessionMaker() as session:
                await session.execute(
                    update(InferenceLease)
                    .where(
                        InferenceLease.id == self.lease_id,
                        InferenceLease.status == "active",
                        InferenceLease.slot_generation == self.slot_generation,
                    )
                    .values(
                        status="cancelled",
                        terminal_reason="client_disconnected",
                        released_at=datetime.now(UTC),
                    )
                )
                await session.commit()

        cancel_task = asyncio.create_task(_cancel())
        try:
            await asyncio.shield(cancel_task)
        except asyncio.CancelledError:
            await asyncio.shield(cancel_task)
            raise

    async def _mark_cancelled(self) -> None:
        async with AsyncSessionMaker() as session:
            await session.execute(
                update(InferenceLease)
                .where(
                    InferenceLease.id == self.lease_id,
                    InferenceLease.status == "active",
                    InferenceLease.slot_generation == self.slot_generation,
                )
                .values(
                    status="cancelled",
                    terminal_reason="client_disconnected",
                    released_at=datetime.now(UTC),
                )
            )
            await session.commit()


class InferenceRequestCancelled(asyncio.CancelledError):
    """Raised when a queued request no longer has a client waiting for it."""


class InferenceScheduler:
    """Select healthy instances and atomically claim PostgreSQL leases."""

    def __init__(self) -> None:
        self._admission_locks: dict[uuid.UUID, asyncio.Lock] = {}
        self._reconciliation_task: asyncio.Task[None] | None = None

    def start_reconciliation(self) -> None:
        """Start cleanup for leases left behind by failed requests or restarts."""
        if self._reconciliation_task is None or self._reconciliation_task.done():
            self._reconciliation_task = asyncio.create_task(self._reconciliation_loop())

    async def stop_reconciliation(self) -> None:
        """Stop the lease cleanup task during application shutdown."""
        if self._reconciliation_task is None:
            return
        self._reconciliation_task.cancel()
        try:
            await self._reconciliation_task
        except asyncio.CancelledError:
            pass
        self._reconciliation_task = None

    async def _reconciliation_loop(self) -> None:
        while True:
            await asyncio.sleep(RECONCILIATION_INTERVAL_SECONDS)
            try:
                await self.reconcile_stale_leases()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Inference lease reconciliation failed")

    async def reconcile_persisted_leases(self) -> int:
        """Reconcile deadlines without disturbing work owned by other workers."""
        return await self.reconcile_stale_leases()

    async def reconcile_stale_leases(self) -> int:
        """Expire queued work and fence servers whose active ownership was lost."""
        now = datetime.now(UTC)
        stops: list[tuple[uuid.UUID, uuid.UUID, int]] = []
        async with AsyncSessionMaker() as session:
            expired_queued = await session.execute(
                update(InferenceLease)
                .where(
                    InferenceLease.status == "queued",
                    InferenceLease.lease_expires_at <= now,
                )
                .values(
                    status="expired",
                    terminal_reason="queue_timeout",
                    released_at=now,
                )
            )
            result = await session.execute(
                select(InferenceLease, ServerInstance)
                .outerjoin(
                    ServerInstance,
                    ServerInstance.id == InferenceLease.server_instance_id,
                )
                .where(
                    InferenceLease.status == "active",
                    or_(
                        InferenceLease.lease_expires_at <= now,
                        InferenceLease.server_instance_id.is_(None),
                        ServerInstance.status != "running",
                        ServerInstance.health_status != "healthy",
                    ),
                )
            )
            stale = list(result.all())
            server_ids = {server.id for _lease, server in stale if server is not None}
            for server_id in server_ids:
                locked = (
                    await session.execute(
                        select(ServerInstance)
                        .where(ServerInstance.id == server_id)
                        .with_for_update()
                    )
                ).scalar_one_or_none()
                if locked is None:
                    continue
                server_unavailable = (
                    locked.status != "running" or locked.health_status != "healthy"
                )
                generation = locked.slot_generation
                failure_update = update(InferenceLease).where(
                    InferenceLease.server_instance_id == locked.id,
                    InferenceLease.status == "active",
                    InferenceLease.slot_generation == generation,
                )
                if not server_unavailable:
                    failure_update = failure_update.where(
                        InferenceLease.lease_expires_at <= now
                    )
                failed = await session.execute(
                    failure_update.values(
                        status="failed",
                        terminal_reason="lease_expired_or_server_unavailable",
                        released_at=now,
                    )
                )
                if not failed.rowcount:
                    continue
                if server_unavailable and locked.status in {"running", "starting"}:
                    locked.status = "stopping"
                    session.add(locked)
                    stops.append((locked.id, locked.agent_id, generation))
            missing_ids = [lease.id for lease, server in stale if server is None]
            if missing_ids:
                await session.execute(
                    update(InferenceLease)
                    .where(
                        InferenceLease.id.in_(missing_ids),
                        InferenceLease.status == "active",
                    )
                    .values(
                        status="failed",
                        terminal_reason="server_missing",
                        released_at=now,
                    )
                )
            await session.commit()

        for server_id, agent_id, generation in stops:
            try:
                await agent_manager.send_to_agent(
                    str(agent_id),
                    "POST",
                    "/servers/stop",
                    {"server_id": str(server_id), "slot_generation": generation},
                    timeout=60.0,
                )
            except Exception:
                logger.exception("Unable to stop fenced server %s", server_id)
        count = int(expired_queued.rowcount or 0) + len(stale)
        if count:
            logger.warning("Reconciled %d stale inference lease(s)", count)
        return count

    def _agent_lock(self, agent_id: uuid.UUID) -> asyncio.Lock:
        return self._admission_locks.setdefault(agent_id, asyncio.Lock())

    async def _admission_open(self) -> bool:
        async with AsyncSessionMaker() as session:
            return bool(
                (
                    await session.execute(
                        text("SELECT pg_try_advisory_xact_lock_shared(:key)"),
                        {"key": BENCHMARK_ADVISORY_LOCK_KEY},
                    )
                ).scalar_one()
            )

    async def _live_vram(self, agent_id: uuid.UUID) -> int | None:
        """Read maximum VRAM from the target agent, ignoring transient usage."""
        try:
            payload = await agent_manager.send_to_agent(
                str(agent_id), "GET", "/gpu", timeout=TELEMETRY_TIMEOUT_SECONDS
            )
            total = int(payload.get("vram_total", 0))
            return max(total, 0)
        except Exception as exc:
            logger.debug("VRAM telemetry unavailable for %s: %s", agent_id, exc)
            return None

    async def _required_vram(self, server: ServerInstance) -> int:
        async with AsyncSessionMaker() as session:
            model = await session.get(Model, server.model_id)
            if model is None:
                return server.vram_required_bytes or 0
            if server.vram_required_bytes is not None:
                return server.vram_required_bytes
            required = model.size_bytes
            for model_id in (server.mmproj_model_id, server.dflash_model_id):
                if model_id:
                    extra = await session.get(Model, model_id)
                    if extra:
                        required += extra.size_bytes
            return required

    async def _active_leases(self, server_id: uuid.UUID) -> int:
        async with AsyncSessionMaker() as session:
            return int(
                (
                    await session.execute(
                        select(func.count(InferenceLease.id)).where(
                            InferenceLease.server_instance_id == server_id,
                            InferenceLease.status == "active",
                        )
                    )
                ).scalar_one()
            )

    async def _stop_idle_server(self, server: ServerInstance) -> None:
        """Stop an idle server and wait for its stopped event."""
        from app.services.server_lifecycle import (
            ServerBusyError,
            request_server_stop,
        )

        try:
            await request_server_stop(server.id, force=False, reason="vram_eviction")
        except ServerBusyError:
            return

    async def _make_room(self, target: ServerInstance, deadline: float) -> None:
        """Evict idle co-located servers until the target fits in VRAM."""
        required = await self._required_vram(target)
        while True:
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    "No VRAM became available for the queued inference request"
                )
            total_vram = await self._live_vram(target.agent_id)

            async with AsyncSessionMaker() as session:
                target_agent = await session.get(Agent, target.agent_id)
                target_host = target_agent.host if target_agent else None
                target_gpu_id = (
                    (target_agent.gpu_info or {}).get("id") if target_agent else None
                )
                rows = (
                    await session.execute(
                        select(ServerInstance, Agent)
                        .join(Agent, Agent.id == ServerInstance.agent_id)
                        .where(
                            ServerInstance.status == "running",
                            ServerInstance.id != target.id,
                            Agent.host == target_host,
                        )
                        .order_by(
                            ServerInstance.last_request_at,
                            ServerInstance.id,
                        )
                    )
                ).all()
                candidates = [
                    server
                    for server, agent in rows
                    if target_gpu_id is None
                    or (agent.gpu_info or {}).get("id") == target_gpu_id
                ]
                allocated = 0
                for candidate in candidates:
                    allocated += await self._required_vram(candidate)
            if total_vram is None:
                await asyncio.sleep(SCHEDULER_POLL_SECONDS)
                continue
            if _vram_requirements_fit(total_vram, required, allocated):
                return
            stopped = False
            for candidate in candidates:
                if await self._active_leases(candidate.id):
                    continue
                logger.info(
                    "Evicting idle server %s for queued target %s",
                    candidate.id,
                    target.id,
                )
                await self._stop_idle_server(candidate)
                stopped = True
                break
            if not stopped:
                await asyncio.sleep(SCHEDULER_POLL_SECONDS)
            else:
                await asyncio.sleep(SCHEDULER_POLL_SECONDS)

    async def _prepare_target(
        self, server: ServerInstance, deadline: float
    ) -> ServerInstance:
        """Make a stopped target fit in VRAM, then cold-start it."""
        from app.services.server_startup import ensure_server_ready

        async with AsyncSessionMaker() as session:
            agent = await session.get(Agent, server.agent_id)
        resource_name = (
            f"{agent.host}:{(agent.gpu_info or {}).get('id', server.agent_id)}"
            if agent is not None
            else str(server.agent_id)
        )
        resource_key = int.from_bytes(
            hashlib.blake2b(resource_name.encode(), digest_size=8).digest(),
            "big",
        ) & ((1 << 63) - 1)

        async with async_engine.connect() as connection:
            locked = False
            while not locked:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError(
                        "Inference queue deadline elapsed during startup"
                    )
                locked = bool(
                    (
                        await connection.execute(
                            text("SELECT pg_try_advisory_lock(:key)"),
                            {"key": resource_key},
                        )
                    ).scalar_one()
                )
                await connection.commit()
                if not locked:
                    await asyncio.sleep(min(SCHEDULER_POLL_SECONDS, remaining))
            try:
                async with AsyncSessionMaker() as session:
                    current = await session.get(ServerInstance, server.id)
                if current is None:
                    raise RuntimeError("Server instance no longer exists")
                if current.status == "stopped":
                    await self._make_room(current, deadline)
                if current.status != "running":
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError(
                            "Inference queue deadline elapsed during startup"
                        )
                    current = await ensure_server_ready(
                        current, start_timeout=remaining
                    )
                return current
            finally:
                await connection.execute(
                    text("SELECT pg_advisory_unlock(:key)"),
                    {"key": resource_key},
                )
                await connection.commit()

    async def _candidates(
        self,
        model_id: uuid.UUID,
        preferred_server_id: uuid.UUID | None = None,
        required_agent_id: uuid.UUID | None = None,
    ) -> list[ServerInstance]:
        async with AsyncSessionMaker() as session:
            statement = select(ServerInstance).where(
                ServerInstance.model_id == model_id,
                ServerInstance.status.in_(["stopped", "starting", "running"]),
                or_(
                    ServerInstance.status != "running",
                    ServerInstance.health_status == "healthy",
                ),
            )
            if preferred_server_id is not None:
                statement = statement.where(ServerInstance.id == preferred_server_id)
            if required_agent_id is not None:
                statement = statement.where(
                    ServerInstance.agent_id == required_agent_id
                )
            result = await session.execute(
                statement.order_by(
                    ServerInstance.last_request_at.nulls_last(), ServerInstance.id
                )
            )
            return list(result.scalars().all())

    async def candidates_for_model(
        self,
        model_id: uuid.UUID,
        *,
        required_agent_id: uuid.UUID | None = None,
    ) -> list[ServerInstance]:
        """Return compatible server candidates for endpoint target resolution."""
        return await self._candidates(
            model_id,
            required_agent_id=required_agent_id,
        )

    async def _claim(
        self,
        lease_id: uuid.UUID,
        server: ServerInstance,
        capacity: int,
    ) -> InferenceLeaseHandle | None:
        now = datetime.now(UTC)
        async with AsyncSessionMaker() as session:
            benchmark_lock = (
                await session.execute(
                    text("SELECT pg_try_advisory_xact_lock_shared(:key)"),
                    {"key": BENCHMARK_ADVISORY_LOCK_KEY},
                )
            ).scalar_one()
            if not benchmark_lock:
                return None
            # Lock the server row so two backend workers cannot claim its last
            # known capacity at the same time.
            locked = (
                await session.execute(
                    select(ServerInstance)
                    .where(ServerInstance.id == server.id)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if (
                locked is None
                or locked.status != "running"
                or locked.health_status != "healthy"
            ):
                return None
            oldest_compatible = (
                await session.execute(
                    select(InferenceLease)
                    .where(
                        InferenceLease.status == "queued",
                        InferenceLease.model_id == locked.model_id,
                        InferenceLease.lease_expires_at > now,
                        or_(
                            InferenceLease.preferred_server_id.is_(None),
                            InferenceLease.preferred_server_id == locked.id,
                        ),
                        or_(
                            InferenceLease.required_agent_id.is_(None),
                            InferenceLease.required_agent_id == locked.agent_id,
                        ),
                    )
                    .order_by(InferenceLease.queued_at, InferenceLease.id)
                    .with_for_update(skip_locked=True)
                    .limit(1)
                )
            ).scalar_one_or_none()
            if oldest_compatible is None or oldest_compatible.id != lease_id:
                return None

            active = (
                await session.execute(
                    select(func.count(InferenceLease.id)).where(
                        InferenceLease.server_instance_id == locked.id,
                        InferenceLease.status == "active",
                    )
                )
            ).scalar_one()
            if int(active) >= server_capacity(locked):
                return None

            lease = oldest_compatible
            lease.server_instance_id = locked.id
            lease.status = "active"
            lease.started_at = now
            lease.lease_expires_at = now + timedelta(seconds=ACTIVE_LEASE_TTL_SECONDS)
            lease.slot_generation = locked.slot_generation
            locked.last_request_at = now
            session.add(locked)
            session.add(lease)
            await session.commit()
            await session.refresh(lease)
            handle = InferenceLeaseHandle(
                lease.request_id, locked, lease.id, lease.slot_generation
            )
            handle.start_renewal()
            return handle

    async def _queue(
        self,
        request_id: str,
        model_id: uuid.UUID,
        deadline_at: datetime,
        preferred_server_id: uuid.UUID | None,
        required_agent_id: uuid.UUID | None,
    ) -> InferenceLease:
        async with AsyncSessionMaker() as session:
            lease = InferenceLease(
                request_id=request_id,
                model_id=model_id,
                lease_expires_at=deadline_at,
                preferred_server_id=preferred_server_id,
                required_agent_id=required_agent_id,
            )
            session.add(lease)
            await session.commit()
            await session.refresh(lease)
            return lease

    async def _finish_queued(
        self, lease_id: uuid.UUID, status: str, reason: str
    ) -> bool:
        async with AsyncSessionMaker() as session:
            result = await session.execute(
                update(InferenceLease)
                .where(
                    InferenceLease.id == lease_id,
                    InferenceLease.status == "queued",
                )
                .values(
                    status=status,
                    terminal_reason=reason,
                    released_at=datetime.now(UTC),
                )
            )
            await session.commit()
            return bool(result.rowcount)

    async def _expire(self, lease_id: uuid.UUID) -> None:
        await self._finish_queued(lease_id, "expired", "queue_timeout")

    async def _cancel(self, lease_id: uuid.UUID) -> None:
        await self._finish_queued(lease_id, "cancelled", "client_disconnected")

    async def _cancel_when_disconnected(
        self,
        lease_id: uuid.UUID,
        is_cancelled: Callable[[], Awaitable[bool]],
    ) -> None:
        """Remove a queued request even while the scheduler is starting a model."""
        try:
            while True:
                if await is_cancelled():
                    await self._cancel(lease_id)
                    return
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise

    async def clear_queue(self) -> int:
        """Cancel every queued request without interrupting active inference."""
        async with AsyncSessionMaker() as session:
            result = await session.execute(
                update(InferenceLease)
                .where(InferenceLease.status == "queued")
                .values(
                    status="cancelled",
                    terminal_reason="queue_cleared",
                    released_at=datetime.now(UTC),
                )
            )
            await session.commit()
            return int(result.rowcount or 0)

    async def _ensure_queued(self, lease_id: uuid.UUID) -> None:
        async with AsyncSessionMaker() as session:
            lease = await session.get(InferenceLease, lease_id)
            if lease is None or lease.status != "queued":
                raise InferenceRequestCancelled

    async def acquire(
        self,
        model_id: uuid.UUID,
        request_id: str,
        preferred_server_id: uuid.UUID | None = None,
        required_agent_id: uuid.UUID | None = None,
        timeout: float = LEASE_TIMEOUT_SECONDS,
        is_cancelled: Callable[[], Awaitable[bool]] | None = None,
    ) -> InferenceLeaseHandle:
        """Wait for and claim a compatible server slot.

        ``preferred_server_id`` preserves explicit aliases. Model-name
        requests can use any compatible healthy instance.
        """
        deadline = asyncio.get_running_loop().time() + timeout
        queued = await self._queue(
            request_id,
            model_id,
            datetime.now(UTC) + timedelta(seconds=timeout),
            preferred_server_id,
            required_agent_id,
        )
        claimed = False
        disconnect_task = (
            asyncio.create_task(self._cancel_when_disconnected(queued.id, is_cancelled))
            if is_cancelled is not None
            else None
        )
        try:
            while True:
                if is_cancelled is not None and await is_cancelled():
                    await self._cancel(queued.id)
                    raise InferenceRequestCancelled
                await self._ensure_queued(queued.id)
                if asyncio.get_running_loop().time() >= deadline:
                    await self._expire(queued.id)
                    raise TimeoutError(
                        f"No inference server slot became available within {timeout:g} seconds"
                    )
                if not await self._admission_open():
                    await asyncio.sleep(SCHEDULER_POLL_SECONDS)
                    continue
                candidates = await self._candidates(
                    model_id, preferred_server_id, required_agent_id
                )
                scored: list[tuple[ServerInstance, int, int]] = []
                for server in candidates:
                    if server.status != "running":
                        continue
                    capacity = server_capacity(server)
                    active = await self._active_leases(server.id)
                    scored.append((server, capacity, active))
                scored.sort(
                    key=lambda item: (
                        -(item[1] - item[2]),
                        item[2],
                    )
                )
                for server, capacity, _active in scored:
                    lease = await self._claim(queued.id, server, capacity)
                    if lease:
                        # A disconnect can race the atomic queue claim. Do not
                        # return an active lease that the response will never
                        # consume.
                        if is_cancelled is not None and await is_cancelled():
                            await lease.cancel()
                            raise InferenceRequestCancelled
                        lease.monitor_disconnect(is_cancelled)
                        claimed = True
                        return lease

                starting = next(
                    (server for server in candidates if server.status == "starting"),
                    None,
                )
                stopped = next(
                    (server for server in candidates if server.status == "stopped"),
                    None,
                )
                target = starting or stopped
                if target is not None:
                    async with self._agent_lock(target.agent_id):
                        prepared = await self._prepare_target(target, deadline)
                    if is_cancelled is not None and await is_cancelled():
                        await self._cancel(queued.id)
                        raise InferenceRequestCancelled
                    await self._ensure_queued(queued.id)
                    lease = await self._claim(
                        queued.id, prepared, server_capacity(prepared)
                    )
                    if lease:
                        # A disconnect can race the atomic queue claim. Do not
                        # return an active lease that the streaming response
                        # will never consume.
                        if is_cancelled is not None and await is_cancelled():
                            await lease.cancel()
                            raise InferenceRequestCancelled
                        lease.monitor_disconnect(is_cancelled)
                        claimed = True
                        return lease
                if asyncio.get_running_loop().time() >= deadline:
                    await self._expire(queued.id)
                    raise TimeoutError(
                        f"No inference server slot became available within {timeout:g} seconds"
                    )
                await asyncio.sleep(SCHEDULER_POLL_SECONDS)
        except InferenceRequestCancelled:
            await self._finish_queued(
                queued.id, "cancelled", "client_disconnected_or_queue_cleared"
            )
            raise
        except asyncio.CancelledError:
            # ASGI cancellation is the reliable signal for a disconnected
            # client while waiting in the FIFO queue.
            cancel_task = asyncio.create_task(self._cancel(queued.id))
            try:
                await asyncio.shield(cancel_task)
            except asyncio.CancelledError:
                await asyncio.shield(cancel_task)
            raise
        except TimeoutError:
            await self._finish_queued(queued.id, "expired", "queue_timeout")
            raise
        except BaseException:
            await self._finish_queued(queued.id, "failed", "admission_error")
            raise
        finally:
            if disconnect_task is not None:
                disconnect_task.cancel()
                await asyncio.gather(disconnect_task, return_exceptions=True)
            if not claimed:
                # This is intentionally conditional so a concurrent claim or
                # queue clear cannot be overwritten by late request cleanup.
                await self._finish_queued(queued.id, "failed", "request_abandoned")

    async def status_snapshot(self) -> dict[str, Any]:
        """Return queue and live slot state for the UI status bar."""
        async with AsyncSessionMaker() as session:
            servers = list(
                (
                    await session.execute(
                        select(ServerInstance).where(
                            ServerInstance.status.in_(["starting", "running"]),
                        )
                    )
                ).scalars()
            )
            queued = int(
                (
                    await session.execute(
                        select(func.count(InferenceLease.id)).where(
                            InferenceLease.status == "queued"
                        )
                    )
                ).scalar_one()
            )
            active_leases = list(
                (
                    await session.execute(
                        select(InferenceLease).where(
                            InferenceLease.status == "active",
                        )
                    )
                ).scalars()
            )

        lease_counts: dict[uuid.UUID, int] = {}
        for lease in active_leases:
            if lease.server_instance_id is not None:
                lease_counts[lease.server_instance_id] = (
                    lease_counts.get(lease.server_instance_id, 0) + 1
                )
        server_status: list[dict[str, Any]] = []
        total_capacity = 0
        total_active = 0
        total_available = 0
        for server in servers:
            booting = server.status != "running" or server.health_status != "healthy"
            active = lease_counts.get(server.id, 0) if not booting else 0
            capacity = server_capacity(server) if not booting else 0
            available = max(capacity - active, 0) if not booting else 0
            total_capacity += capacity
            total_active += active
            total_available += available
            server_status.append(
                {
                    "id": str(server.id),
                    "alias": server.alias,
                    "model_id": str(server.model_id),
                    "capacity": capacity,
                    "active": active,
                    "available": available,
                    "telemetry_known": not booting,
                    "state": "booting" if booting else "ready",
                }
            )

        return {
            "event": "queue.status",
            "data": {
                "queued": queued,
                "active": total_active,
                "available": total_available,
                "capacity": total_capacity,
                "servers": server_status,
            },
        }


inference_scheduler = InferenceScheduler()

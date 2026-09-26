"""PostgreSQL-backed scheduling leases for inference requests."""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, or_
from sqlmodel import select

from app.db.session import AsyncSessionMaker
from app.models import Agent, InferenceLease, Model, ServerInstance
from app.services.agent_manager import agent_manager

logger = logging.getLogger(__name__)

LEASE_TIMEOUT_SECONDS = 30 * 60
SCHEDULER_POLL_SECONDS = 1.0
TELEMETRY_TIMEOUT_SECONDS = 5.0
ORPHANED_LEASE_GRACE_SECONDS = 30.0
DEFAULT_PARALLEL_SLOTS = 4


@dataclass(frozen=True)
class SlotTelemetry:
    """Normalized llama.cpp slot state."""

    capacity: int
    active: int
    available: int
    known: bool = True


def normalize_slot_telemetry(
    payload: Any, capacity: int = DEFAULT_PARALLEL_SLOTS
) -> SlotTelemetry:
    """Normalize the several llama.cpp `/slots` response shapes.

    Older llama.cpp builds expose a list directly, while newer builds wrap it
    in ``slots``. The endpoint reports active slots, not configured capacity,
    so capacity comes from the server's configured parallel count.
    """
    capacity = max(int(capacity), 1)
    slots: Any = payload.get("slots") if isinstance(payload, dict) else payload
    if not isinstance(slots, list):
        return SlotTelemetry(capacity, 0, capacity, known=False)
    active = 0
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        state = slot.get("state")
        busy = slot.get("is_processing") is True or slot.get("processing") is True
        busy = busy or state in {1, "processing", "busy", "generating"}
        if busy:
            active += 1
    return SlotTelemetry(capacity, active, max(capacity - active, 0))


async def get_slot_telemetry(server: ServerInstance) -> SlotTelemetry:
    """Read a server's llama.cpp slots without making telemetry mandatory."""
    options = (
        server.engine_options or {}
        if server.engine == "halogen-flash"
        else server.server_options or {}
    )
    configured_parallel = (
        options.get("kv_slots")
        if server.engine == "halogen-flash"
        else options.get("parallel")
    )
    try:
        capacity = (
            int(configured_parallel)
            if configured_parallel is not None
            else DEFAULT_PARALLEL_SLOTS
        )
    except TypeError, ValueError:
        capacity = DEFAULT_PARALLEL_SLOTS
    capacity = max(capacity, 1)
    try:
        agent = await agent_manager.get_agent(str(server.agent_id))
        if agent is None or agent.status != "online":
            return SlotTelemetry(capacity, 0, capacity, known=False)
        telemetry_path = (
            f"/proxy/{server.id}/health"
            if server.engine in {"halogen", "halogen-flash"}
            else f"/proxy/{server.id}/slots"
        )
        payload = await agent_manager.send_to_agent(
            str(server.agent_id),
            "GET",
            telemetry_path,
            timeout=TELEMETRY_TIMEOUT_SECONDS,
        )
        if server.engine in {"halogen", "halogen-flash"}:
            capacity = max(int(payload.get("slots", configured_parallel or 1)), 1)
            active = max(int(payload.get("in_flight", 0)), 0)
            return SlotTelemetry(capacity, active, max(capacity - active, 0))
        return normalize_slot_telemetry(payload, capacity)
    except Exception as exc:
        logger.debug("Slot telemetry unavailable for %s: %s", server.id, exc)
        return SlotTelemetry(capacity, 0, capacity, known=False)


@dataclass
class InferenceLeaseHandle:
    request_id: str
    server: ServerInstance
    lease_id: uuid.UUID

    async def release(self) -> None:
        """Release the lease in a short, independent transaction."""
        async with AsyncSessionMaker() as session:
            lease = await session.get(InferenceLease, self.lease_id)
            if lease and lease.status == "active":
                lease.status = "released"
                lease.released_at = datetime.now(UTC)
                session.add(lease)
                await session.commit()


class InferenceRequestCancelled(asyncio.CancelledError):
    """Raised when a queued request no longer has a client waiting for it."""


class InferenceScheduler:
    """Select healthy instances and atomically claim PostgreSQL leases."""

    def __init__(self) -> None:
        self._admission_locks: dict[uuid.UUID, asyncio.Lock] = {}

    def _agent_lock(self, agent_id: uuid.UUID) -> asyncio.Lock:
        return self._admission_locks.setdefault(agent_id, asyncio.Lock())

    async def _queue_is_head(self, lease_id: uuid.UUID) -> bool:
        async with AsyncSessionMaker() as session:
            head = (
                await session.execute(
                    select(InferenceLease)
                    .where(InferenceLease.status == "queued")
                    .order_by(InferenceLease.queued_at, InferenceLease.id)
                    .limit(1)
                )
            ).scalar_one_or_none()
            return head is not None and head.id == lease_id

    async def _live_vram(self, agent_id: uuid.UUID) -> tuple[int, int] | None:
        """Read current total/free VRAM from the target agent."""
        try:
            payload = await agent_manager.send_to_agent(
                str(agent_id), "GET", "/gpu", timeout=TELEMETRY_TIMEOUT_SECONDS
            )
            total = int(payload.get("vram_total", 0))
            used = int(payload.get("vram_used", 0))
            free = int(payload.get("vram_free", total - used))
            return total, max(free, 0)
        except Exception as exc:
            logger.debug("VRAM telemetry unavailable for %s: %s", agent_id, exc)
            return None

    async def _required_vram(self, server: ServerInstance) -> int:
        async with AsyncSessionMaker() as session:
            model = await session.get(Model, server.model_id)
            if model is None:
                return server.vram_required_bytes or 0
            required = server.vram_required_bytes or model.size_bytes
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
                            InferenceLease.lease_expires_at > datetime.now(UTC),
                        )
                    )
                ).scalar_one()
            )

    async def _reconcile_active_leases(
        self, server_id: uuid.UUID, engine_active: int
    ) -> None:
        """Release old leases no longer represented by engine activity."""
        cutoff = datetime.now(UTC) - timedelta(seconds=ORPHANED_LEASE_GRACE_SECONDS)
        async with AsyncSessionMaker() as session:
            result = await session.execute(
                select(InferenceLease)
                .where(
                    InferenceLease.server_instance_id == server_id,
                    InferenceLease.status == "active",
                    InferenceLease.started_at < cutoff,
                )
                .order_by(InferenceLease.started_at, InferenceLease.id)
            )
            leases = list(result.scalars().all())
            stale = leases[max(engine_active, 0) :]
            if not stale:
                return
            now = datetime.now(UTC)
            for lease in stale:
                lease.status = "released"
                lease.released_at = now
                session.add(lease)
            await session.commit()
            logger.warning(
                "Released %d orphaned lease(s) for server %s; engine reports %d active",
                len(stale),
                server_id,
                engine_active,
            )

    async def _stop_idle_server(self, server: ServerInstance) -> None:
        """Stop an idle server and wait for its stopped event."""
        if await self._active_leases(server.id):
            return
        async with AsyncSessionMaker() as session:
            current = await session.get(ServerInstance, server.id)
            if current is None or current.status != "running":
                return
            current.status = "stopping"
            session.add(current)
            await session.commit()
        try:
            await agent_manager.send_to_agent(
                str(server.agent_id),
                "POST",
                "/servers/stop",
                {"server_id": str(server.id)},
                timeout=60.0,
            )
        except Exception:
            async with AsyncSessionMaker() as session:
                current = await session.get(ServerInstance, server.id)
                if current and current.status == "stopping":
                    current.status = "running"
                    session.add(current)
                    await session.commit()
            raise

    async def _make_room(self, target: ServerInstance, deadline: float) -> None:
        """Evict idle co-located servers until the target fits in VRAM."""
        required = await self._required_vram(target)
        while True:
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    "No VRAM became available for the queued inference request"
                )
            usage = await self._live_vram(target.agent_id)
            if usage is not None and usage[1] >= required:
                return

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

        if server.status != "running":
            await self._make_room(server, deadline)
            server = await ensure_server_ready(server)
        return server

    async def _candidates(self, model_id: uuid.UUID) -> list[ServerInstance]:
        async with AsyncSessionMaker() as session:
            result = await session.execute(
                select(ServerInstance)
                .where(
                    ServerInstance.model_id == model_id,
                    ServerInstance.status.in_(["stopped", "starting", "running"]),
                    or_(
                        ServerInstance.status != "running",
                        ServerInstance.health_status == "healthy",
                    ),
                )
                .order_by(ServerInstance.last_request_at, ServerInstance.id)
            )
            return list(result.scalars().all())

    async def _claim(
        self,
        lease_id: uuid.UUID,
        server: ServerInstance,
        telemetry: SlotTelemetry,
    ) -> InferenceLeaseHandle | None:
        now = datetime.now(UTC)
        async with AsyncSessionMaker() as session:
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
            active = (
                await session.execute(
                    select(func.count(InferenceLease.id)).where(
                        InferenceLease.server_instance_id == locked.id,
                        InferenceLease.status == "active",
                        InferenceLease.lease_expires_at > now,
                    )
                )
            ).scalar_one()
            capacity = max(telemetry.capacity, 1)
            if telemetry.known and telemetry.available <= 0:
                return None
            admitted_active = telemetry.active if telemetry.known else int(active)
            if admitted_active >= capacity:
                return None

            lease = await session.get(InferenceLease, lease_id)
            if lease is None or lease.status != "queued":
                return None
            lease.server_instance_id = locked.id
            lease.status = "active"
            lease.started_at = now
            lease.lease_expires_at = now + timedelta(seconds=LEASE_TIMEOUT_SECONDS)
            locked.last_request_at = now
            session.add(locked)
            session.add(lease)
            await session.commit()
            await session.refresh(lease)
            return InferenceLeaseHandle(lease.request_id, locked, lease.id)

    async def _queue(self, request_id: str, model_id: uuid.UUID) -> InferenceLease:
        async with AsyncSessionMaker() as session:
            lease = InferenceLease(request_id=request_id, model_id=model_id)
            session.add(lease)
            await session.commit()
            await session.refresh(lease)
            return lease

    async def _expire(self, lease_id: uuid.UUID) -> None:
        async with AsyncSessionMaker() as session:
            lease = await session.get(InferenceLease, lease_id)
            if lease and lease.status == "queued":
                lease.status = "expired"
                lease.released_at = datetime.now(UTC)
                session.add(lease)
                await session.commit()

    async def _cancel(self, lease_id: uuid.UUID) -> None:
        async with AsyncSessionMaker() as session:
            lease = await session.get(InferenceLease, lease_id)
            if lease and lease.status == "queued":
                lease.status = "cancelled"
                lease.released_at = datetime.now(UTC)
                session.add(lease)
                await session.commit()

    async def clear_queue(self) -> int:
        """Cancel every queued request without interrupting active inference."""
        async with AsyncSessionMaker() as session:
            result = await session.execute(
                select(InferenceLease).where(InferenceLease.status == "queued")
            )
            leases = list(result.scalars().all())
            now = datetime.now(UTC)
            for lease in leases:
                lease.status = "cancelled"
                lease.released_at = now
                session.add(lease)
            if leases:
                await session.commit()
            return len(leases)

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
        timeout: float = LEASE_TIMEOUT_SECONDS,
        is_cancelled: Callable[[], Awaitable[bool]] | None = None,
    ) -> InferenceLeaseHandle:
        """Wait for and claim a compatible server slot.

        ``preferred_server_id`` preserves explicit aliases. Model-name
        requests can use any compatible healthy instance.
        """
        deadline = asyncio.get_running_loop().time() + timeout
        queued = await self._queue(request_id, model_id)
        while True:
            if is_cancelled is not None and await is_cancelled():
                await self._cancel(queued.id)
                raise InferenceRequestCancelled
            await self._ensure_queued(queued.id)
            if not await self._queue_is_head(queued.id):
                if asyncio.get_running_loop().time() >= deadline:
                    await self._expire(queued.id)
                    raise TimeoutError(
                        "No inference server slot became available within 30 minutes"
                    )
                await asyncio.sleep(SCHEDULER_POLL_SECONDS)
                continue
            candidates = await self._candidates(model_id)
            if preferred_server_id is not None:
                async with AsyncSessionMaker() as session:
                    target = await session.get(ServerInstance, preferred_server_id)
                candidates = [target] if target is not None else []
            scored: list[tuple[ServerInstance, SlotTelemetry]] = []
            for server in candidates:
                if server.status != "running":
                    async with self._agent_lock(server.agent_id):
                        server = await self._prepare_target(server, deadline)
                telemetry = await get_slot_telemetry(server)
                if telemetry.known:
                    await self._reconcile_active_leases(server.id, telemetry.active)
                scored.append((server, telemetry))
            scored.sort(
                key=lambda item: (not item[1].known, -item[1].available, item[1].active)
            )
            for server, telemetry in scored:
                lease = await self._claim(queued.id, server, telemetry)
                if lease:
                    return lease
            if asyncio.get_running_loop().time() >= deadline:
                await self._expire(queued.id)
                raise TimeoutError(
                    "No inference server slot became available within 30 minutes"
                )
            await asyncio.sleep(SCHEDULER_POLL_SECONDS)

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
                            InferenceLease.lease_expires_at > datetime.now(UTC),
                        )
                    )
                ).scalars()
            )

        telemetry = await asyncio.gather(
            *(
                get_slot_telemetry(server)
                if server.status == "running" and server.health_status == "healthy"
                else asyncio.sleep(0, result=SlotTelemetry(0, 0, 0, known=False))
                for server in servers
            ),
            return_exceptions=True,
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
        for server, value in zip(servers, telemetry, strict=True):
            slots = (
                value
                if isinstance(value, SlotTelemetry)
                else SlotTelemetry(0, 0, 0, known=False)
            )
            booting = server.status != "running" or server.health_status != "healthy"
            active = lease_counts.get(server.id, 0) if not booting else 0
            if not booting:
                active = slots.active if slots.known else max(active, slots.active)
            capacity = slots.capacity if not booting else 0
            available = slots.available if not booting else 0
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
                    "telemetry_known": slots.known and not booting,
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

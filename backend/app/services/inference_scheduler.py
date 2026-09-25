"""PostgreSQL-backed scheduling leases for inference requests."""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func
from sqlmodel import col, select

from app.db.session import AsyncSessionMaker
from app.models import InferenceLease, ServerInstance
from app.services.agent_manager import agent_manager

logger = logging.getLogger(__name__)

LEASE_TIMEOUT_SECONDS = 30 * 60
SCHEDULER_POLL_SECONDS = 1.0
TELEMETRY_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class SlotTelemetry:
    """Normalized llama.cpp slot state."""

    capacity: int
    active: int
    available: int
    known: bool = True


def normalize_slot_telemetry(payload: Any) -> SlotTelemetry:
    """Normalize the several llama.cpp `/slots` response shapes.

    Older llama.cpp builds expose a list directly, while newer builds wrap it
    in ``slots``. Unknown or malformed data is deliberately represented as a
    single available slot so a healthy server remains usable.
    """
    slots: Any = payload.get("slots") if isinstance(payload, dict) else payload
    if not isinstance(slots, list):
        return SlotTelemetry(1, 0, 1, known=False)
    capacity = len(slots)
    active = 0
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        state = slot.get("state")
        busy = slot.get("is_processing") is True or slot.get("processing") is True
        busy = busy or state in {1, "processing", "busy", "generating"}
        if busy:
            active += 1
    if capacity == 0:
        return SlotTelemetry(1, 0, 1, known=False)
    return SlotTelemetry(capacity, active, max(capacity - active, 0))


async def get_slot_telemetry(server: ServerInstance) -> SlotTelemetry:
    """Read a server's llama.cpp slots without making telemetry mandatory."""
    try:
        agent = await agent_manager.get_agent(str(server.agent_id))
        if agent is None or agent.status != "online" or not agent.websocket_connected:
            return SlotTelemetry(0, 0, 0, known=False)
        payload = await agent_manager.send_to_agent(
            str(server.agent_id),
            "GET",
            f"/proxy/{server.id}/slots",
            timeout=TELEMETRY_TIMEOUT_SECONDS,
        )
        return normalize_slot_telemetry(payload)
    except Exception as exc:
        logger.debug("Slot telemetry unavailable for %s: %s", server.id, exc)
        return SlotTelemetry(1, 0, 1, known=False)


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


class InferenceScheduler:
    """Select healthy instances and atomically claim PostgreSQL leases."""

    async def _candidates(self, model_id: uuid.UUID) -> list[ServerInstance]:
        async with AsyncSessionMaker() as session:
            result = await session.execute(
                select(ServerInstance)
                .where(
                    ServerInstance.model_id == model_id,
                    ServerInstance.status == "running",
                    col(ServerInstance.health_status).in_(["healthy", "unknown"]),
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
            if locked is None or locked.status != "running":
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
            if int(active) >= capacity:
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

    async def acquire(
        self,
        model_id: uuid.UUID,
        request_id: str,
        preferred_server_id: uuid.UUID | None = None,
        timeout: float = LEASE_TIMEOUT_SECONDS,
    ) -> InferenceLeaseHandle:
        """Wait for and claim a compatible server slot.

        ``preferred_server_id`` preserves explicit aliases. Model-name
        requests can use any compatible healthy instance.
        """
        deadline = asyncio.get_running_loop().time() + timeout
        queued = await self._queue(request_id, model_id)
        while True:
            candidates = await self._candidates(model_id)
            if preferred_server_id is not None:
                candidates = [s for s in candidates if s.id == preferred_server_id]
            scored: list[tuple[ServerInstance, SlotTelemetry]] = []
            for server in candidates:
                telemetry = await get_slot_telemetry(server)
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
                        select(ServerInstance).where(ServerInstance.status == "running")
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

        telemetry = await asyncio.gather(
            *(get_slot_telemetry(server) for server in servers),
            return_exceptions=True,
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
            total_capacity += slots.capacity
            total_active += slots.active
            total_available += slots.available
            server_status.append(
                {
                    "id": str(server.id),
                    "alias": server.alias,
                    "model_id": str(server.model_id),
                    "capacity": slots.capacity,
                    "active": slots.active,
                    "available": slots.available,
                    "telemetry_known": slots.known,
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

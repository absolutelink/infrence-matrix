"""Transactional server start/stop fencing shared by scheduler integrations."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, update
from sqlmodel import select

from app.db.session import AsyncSessionMaker
from app.models import InferenceLease, ServerInstance
from app.services.agent_manager import agent_manager


class ServerBusyError(RuntimeError):
    """Raised when a graceful stop would interrupt active inference."""


async def reserve_server_start(server_id: uuid.UUID) -> ServerInstance:
    """Fence a new process generation and reserve its startup."""
    async with AsyncSessionMaker() as session:
        server = (
            await session.execute(
                select(ServerInstance)
                .where(ServerInstance.id == server_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if server is None:
            raise RuntimeError("Server instance no longer exists")
        if server.status in {"starting", "running", "stopping"}:
            raise ServerBusyError(f"Server is already {server.status}")
        server.slot_generation += 1
        server.status = "starting"
        server.health_status = "unknown"
        server.error_message = None
        session.add(server)
        await session.commit()
        await session.refresh(server)
        return server


async def request_server_stop(
    server_id: uuid.UUID,
    *,
    force: bool,
    reason: str,
) -> ServerInstance:
    """Fence admission, optionally fail active leases, and stop one process."""
    async with AsyncSessionMaker() as session:
        server = (
            await session.execute(
                select(ServerInstance)
                .where(ServerInstance.id == server_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if server is None:
            raise RuntimeError("Server instance no longer exists")
        generation = server.slot_generation
        active = int(
            (
                await session.execute(
                    select(func.count(InferenceLease.id)).where(
                        InferenceLease.server_instance_id == server.id,
                        InferenceLease.status == "active",
                    )
                )
            ).scalar_one()
        )
        if active and not force:
            raise ServerBusyError(f"Server has {active} active inference request(s)")
        server.status = "stopping"
        server.health_status = "unknown"
        session.add(server)
        if active:
            await session.execute(
                update(InferenceLease)
                .where(
                    InferenceLease.server_instance_id == server.id,
                    InferenceLease.status == "active",
                )
                .values(
                    status="failed",
                    terminal_reason=reason,
                    released_at=datetime.now(UTC),
                )
            )
        await session.commit()
        agent_id = server.agent_id

    try:
        await agent_manager.send_to_agent(
            str(agent_id),
            "POST",
            "/servers/stop",
            {"server_id": str(server_id), "slot_generation": generation},
            timeout=60.0,
        )
    except Exception as exc:
        async with AsyncSessionMaker() as session:
            current = await session.get(ServerInstance, server_id)
            if (
                current is not None
                and current.slot_generation == generation
                and current.status == "stopping"
            ):
                current.status = "error"
                current.health_status = "unknown"
                current.error_message = str(exc)
                session.add(current)
                await session.commit()
        raise

    async with AsyncSessionMaker() as session:
        current = (
            await session.execute(
                select(ServerInstance)
                .where(ServerInstance.id == server_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if current is None:
            raise RuntimeError("Server instance was deleted while stopping")
        if current.slot_generation == generation and current.status == "stopping":
            current.status = "stopped"
            current.health_status = "unknown"
            current.error_message = None
            session.add(current)
            await session.commit()
        return current

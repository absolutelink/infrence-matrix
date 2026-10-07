"""Background presence sweep (Phase 16: agent-keyed).

Redis ``im:ws:presence:{agent_id}`` is the liveness source of truth: it is
refreshed with a TTL on every accepted inbound frame and every outbound
command/pong. This loop periodically finds ProviderAgent rows the admin
believes are connected (websocket_connected=True) whose presence key has
expired and marks them (and their backends) disconnected. It also covers
sockets orphaned by an unclean shutdown of a previous admin process.

The live-socket registry in the connection manager is the fast path (an
explicit WebSocketDisconnect marks the agent disconnected immediately); the
sweep is the safety net for missed disconnects and admin restarts.
"""

import asyncio
import contextlib
import logging
import uuid
from datetime import UTC, datetime

import redis.asyncio as aioredis
from sqlmodel import Session, col, select

from app.core.config import settings
from app.core.db import engine
from app.models import ProviderAgent, ProviderInstance
from app.services import redis_keys
from app.services.wire import InstanceStatusValue

logger = logging.getLogger("admin.presence_sweep")


async def sweep_once(redis_client: aioredis.Redis) -> list[str]:
    """Mark connected-but-expired agents disconnected.

    Returns the ids of the agents that were marked stale.
    """
    with Session(engine) as session:
        candidates = session.exec(
            select(ProviderAgent).where(
                col(ProviderAgent.websocket_connected) == True  # noqa: E712
            )
        ).all()
        agent_ids = [str(agent.id) for agent in candidates]

    stale: list[str] = []
    for agent_id in agent_ids:
        alive = await redis_client.exists(redis_keys.presence_key(agent_id))
        if not alive:
            stale.append(agent_id)

    if stale:
        now = datetime.now(UTC)
        with Session(engine) as session:
            rows = session.exec(
                select(ProviderAgent).where(
                    col(ProviderAgent.id).in_([uuid.UUID(i) for i in stale])
                )
            ).all()
            for agent in rows:
                agent.websocket_connected = False
                agent.agent_status = InstanceStatusValue.DISCONNECTED
                agent.last_seen = now
                agent.updated_at = now
                session.add(agent)
                # Backend state is unknown on a dead socket: drop each
                # backend's idle-reaper load clock so a reconnect boot
                # re-arms it.
                backends = session.exec(
                    select(ProviderInstance).where(
                        col(ProviderInstance.agent_id) == agent.id
                    )
                ).all()
                for inst in backends:
                    inst.backend_loaded_at = None
                    session.add(inst)
            session.commit()

        # Drop ownership so a zombie socket cannot act on behalf of the row.
        for agent_id in stale:
            await redis_client.delete(redis_keys.owner_key(agent_id))
        logger.info(
            "presence sweep marked %d agent(s) disconnected: %s",
            len(stale),
            ", ".join(stale),
        )

    # Metrics-ownership maintenance: release leases held by now-stale
    # agents, then try to reassign machines that still have a connected
    # agent but no owner.
    from app.services import metrics_service

    for agent_id in stale:
        with contextlib.suppress(Exception):
            await metrics_service.release_ownership(redis_client, agent_id)
    await _reassign_ownerless_machines(redis_client)
    # Phase 9 self-heal: connected agents whose backends' stored
    # config_fingerprint lags the definition's current backend_config get a
    # provider.config.update push (catches misses from the connect path; the
    # heal is a no-op when fingerprints match).
    await _heal_stale_fingerprints()
    return stale


async def _heal_stale_fingerprints() -> None:
    from app.services.config_update import heal_agent_stale_fingerprints

    with Session(engine) as session:
        rows = session.exec(
            select(ProviderAgent).where(
                col(ProviderAgent.websocket_connected) == True  # noqa: E712
            )
        ).all()
        agent_ids = [str(agent.id) for agent in rows]
    with contextlib.suppress(Exception):
        await asyncio.gather(*(heal_agent_stale_fingerprints(i) for i in agent_ids))


async def _reassign_ownerless_machines(redis_client: aioredis.Redis) -> None:
    """Assign metrics ownership for machines with connected agents but no
    current owner (e.g. the previous owner's lease expired)."""
    from app.services import metrics_service

    with Session(engine) as session:
        rows = session.exec(
            select(ProviderAgent).where(
                col(ProviderAgent.websocket_connected) == True  # noqa: E712
            )
        ).all()
        by_machine: dict[str, list[str]] = {}
        for agent in rows:
            by_machine.setdefault(agent.machine.uid, []).append(str(agent.id))

    for machine_uid, agent_ids in by_machine.items():
        owner = await redis_client.get(redis_keys.metrics_owner_key(machine_uid))
        if owner is not None and owner in agent_ids:
            continue
        for agent_id in agent_ids:
            try:
                if await metrics_service.assign_ownership(redis_client, agent_id):
                    break
            except Exception:  # noqa: BLE001
                logger.debug(
                    "metrics reassignment failed for machine %s",
                    machine_uid,
                    exc_info=True,
                )


async def presence_sweep_loop(
    redis_client: aioredis.Redis, stop_event: asyncio.Event
) -> None:
    interval = settings.INSTANCE_SWEEP_INTERVAL_SECONDS
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return
        except TimeoutError:
            pass
        try:
            await sweep_once(redis_client)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("presence sweep failed; continuing")

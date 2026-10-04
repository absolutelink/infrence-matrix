"""Background presence sweep.

Redis ``im:ws:presence:{instance_id}`` is the liveness source of truth: it
is refreshed with a TTL on every accepted inbound frame and every outbound
command/pong. This loop periodically finds DB rows the admin believes are
connected (websocket_connected=True) whose presence key has expired and
marks them disconnected. It also covers sockets orphaned by an unclean
shutdown of a previous admin process.

The live-socket registry in the connection manager is the fast path (an
explicit WebSocketDisconnect marks the row disconnected immediately); the
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
from app.models import ProviderInstance
from app.services import redis_keys
from app.services.wire import InstanceStatusValue

logger = logging.getLogger("admin.presence_sweep")


async def sweep_once(redis_client: aioredis.Redis) -> list[str]:
    """Mark connected-but-expired instances disconnected.

    Returns the ids of the instances that were marked stale.
    """
    with Session(engine) as session:
        candidates = session.exec(
            select(ProviderInstance).where(
                col(ProviderInstance.websocket_connected) == True  # noqa: E712
            )
        ).all()
        instance_ids = [str(inst.id) for inst in candidates]

    stale: list[str] = []
    for instance_id in instance_ids:
        alive = await redis_client.exists(redis_keys.presence_key(instance_id))
        if not alive:
            stale.append(instance_id)

    if stale:
        now = datetime.now(UTC)
        with Session(engine) as session:
            rows = session.exec(
                select(ProviderInstance).where(
                    col(ProviderInstance.id).in_([uuid.UUID(i) for i in stale])
                )
            ).all()
            for inst in rows:
                inst.websocket_connected = False
                inst.instance_status = InstanceStatusValue.DISCONNECTED
                inst.updated_at = now
                session.add(inst)
            session.commit()

        # Drop ownership so a zombie socket cannot act on behalf of the row.
        for instance_id in stale:
            await redis_client.delete(redis_keys.owner_key(instance_id))
        logger.info(
            "presence sweep marked %d instance(s) disconnected: %s",
            len(stale),
            ", ".join(stale),
        )

    # Metrics-ownership maintenance: release leases held by now-stale
    # instances, then try to reassign machines that still have a
    # connected instance but no owner.
    from app.services import metrics_service

    for instance_id in stale:
        with contextlib.suppress(Exception):
            await metrics_service.release_ownership(redis_client, instance_id)
    await _reassign_ownerless_machines(redis_client)
    return stale


async def _reassign_ownerless_machines(redis_client: aioredis.Redis) -> None:
    """Assign metrics ownership for machines with connected instances
    but no current owner (e.g. the previous owner's lease expired)."""
    from app.services import metrics_service

    with Session(engine) as session:
        rows = session.exec(
            select(ProviderInstance).where(
                col(ProviderInstance.websocket_connected) == True  # noqa: E712
            )
        ).all()
        by_machine: dict[str, list[str]] = {}
        for inst in rows:
            by_machine.setdefault(inst.machine.uid, []).append(str(inst.id))

    for machine_uid, instance_ids in by_machine.items():
        owner = await redis_client.get(redis_keys.metrics_owner_key(machine_uid))
        if owner is not None and owner in instance_ids:
            continue
        for instance_id in instance_ids:
            try:
                if await metrics_service.assign_ownership(redis_client, instance_id):
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

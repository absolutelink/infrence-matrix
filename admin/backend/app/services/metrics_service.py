"""Machine-level metrics ownership (Phase 5, agent-keyed in Phase 16).

Exactly one provider **agent** per machine reports machine-level metrics
(GPU/RAM/CPU/storage). The admin assigns ownership when an agent's
WebSocket connects (and re-tries from the presence sweep) using a Redis
lease:

  im:metrics:owner:{machine_uid}  -> agent_id (SET NX, TTL 30s)

Assignment sends the `metrics.assign` WS command with the agent's declared
categories (persisted at registration under
`im:metrics:cats:{agent_id}`). The owner refreshes its lease by emitting
`metrics.machine` events; the snapshot is stored at
`im:metrics:machine:{machine_uid}`. Events from non-owners are dropped
(stale duplicate reporters). On disconnect (or stale sweep) the lease is
released so another agent on the same machine can take over.
"""

import json
import logging
import uuid
from typing import Any

from sqlmodel import Session

from app.core.db import engine
from app.core.redis import get_redis_from_app
from app.models import ProviderAgent
from app.services import redis_keys
from app.services.connection_manager import manager

logger = logging.getLogger("admin.metrics_service")

# Shorter than the lease TTL so a dead provider frees the machine quickly.
ASSIGN_COMMAND_TIMEOUT = 10.0


def _resolve_redis(app_or_redis: Any) -> Any:
    """Accept either a FastAPI app or a Redis client directly.

    ws.py / connection_manager hold the app; the presence sweep holds the
    raw client. Both paths need the same lease operations.
    """
    if hasattr(app_or_redis, "state"):
        return get_redis_from_app(app_or_redis)
    return app_or_redis


def _machine_uid_for_agent(agent_id: str) -> str | None:
    with Session(engine) as session:
        agent = session.get(ProviderAgent, uuid.UUID(agent_id))
        if agent is None:
            return None
        return agent.machine.uid


async def assign_ownership(app: Any, agent_id: str) -> bool:
    """Try to make ``agent_id`` the metrics owner of its machine.

    Returns True when the lease was newly acquired (and the
    ``metrics.assign`` command was sent). No-op when another agent
    already owns the machine or the agent declares no categories.
    """
    machine_uid = _machine_uid_for_agent(agent_id)
    if machine_uid is None:
        logger.debug("metrics assign: unknown agent %s", agent_id)
        return False

    redis_client = _resolve_redis(app)
    categories_json = await redis_client.get(redis_keys.metrics_cats_key(agent_id))
    categories: list[str] = json.loads(categories_json) if categories_json else []
    if not categories:
        # Nothing to report; don't claim the machine.
        return False

    owner_key = redis_keys.metrics_owner_key(machine_uid)
    acquired = await redis_client.set(
        owner_key, agent_id, nx=True, ex=redis_keys.METRICS_OWNER_TTL_SECONDS
    )
    if not acquired:
        return False

    try:
        await manager.send_command(
            agent_id,
            "metrics.assign",
            {"machine_uid": machine_uid, "categories": categories},
            timeout=ASSIGN_COMMAND_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "metrics.assign to agent %s failed (%s); releasing ownership",
            agent_id,
            exc,
        )
        await release_ownership(app, agent_id)
        return False

    logger.info(
        "machine %s metrics ownership assigned to agent %s (%s)",
        machine_uid,
        agent_id,
        ", ".join(categories),
    )
    return True


async def release_ownership(app: Any, agent_id: str) -> None:
    """Drop the metrics-owner lease if ``agent_id`` currently holds it."""
    machine_uid = _machine_uid_for_agent(agent_id)
    if machine_uid is None:
        return
    redis_client = _resolve_redis(app)
    owner_key = redis_keys.metrics_owner_key(machine_uid)
    owner = await redis_client.get(owner_key)
    if owner == agent_id:
        await redis_client.delete(owner_key)
        logger.info(
            "machine %s metrics ownership released from %s", machine_uid, agent_id
        )


async def handle_machine_metrics(
    app: Any, agent_id: str, payload: dict[str, Any]
) -> bool:
    """Store a metrics.machine snapshot from the current owner.

    Returns True when stored; False (drop) when the sender is not the
    machine's owner. Refreshes the owner lease on receipt.
    """
    machine_uid = _machine_uid_for_agent(agent_id)
    if machine_uid is None:
        return False
    redis_client = _resolve_redis(app)
    owner_key = redis_keys.metrics_owner_key(machine_uid)
    owner = await redis_client.get(owner_key)
    if owner != agent_id:
        logger.debug(
            "dropping metrics.machine from non-owner agent %s (owner=%s)",
            agent_id,
            owner,
        )
        return False
    await redis_client.set(owner_key, agent_id, ex=redis_keys.METRICS_OWNER_TTL_SECONDS)
    await redis_client.set(
        redis_keys.metrics_machine_key(machine_uid),
        json.dumps(payload),
        ex=redis_keys.METRICS_OWNER_TTL_SECONDS,
    )
    return True

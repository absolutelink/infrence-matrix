"""Machine-level metrics: owner-gated machine-wide + per-agent GPU merge
(Phase 5, agent-keyed Phase 16, per-GPU split Phase 17).

Two ownership classes of metrics share the ``metrics.machine`` frame:

* **Machine-wide categories** (``os_ram`` / ``cpu`` / ``storage``) are visible
  from any container and stay single-owner per machine. The admin assigns
  ownership when an agent's WebSocket connects (and re-tries from the presence
  sweep) using a Redis lease ``im:metrics:owner:{machine_uid}`` (SET NX, TTL
  30s). Assignment sends the ``metrics.assign`` WS command with the agent's
  declared categories (persisted at registration under
  ``im:metrics:cats:{agent_id}``). The owner refreshes its lease by emitting
  those categories; the snapshot is stored at
  ``im:metrics:machine:{machine_uid}``. Machine-wide sections from non-owners
  are dropped (stale duplicate reporters). On disconnect (or stale sweep) the
  lease is released so another agent on the same machine can take over.

* **GPU categories** (``vram`` / ``gpu_usage``) are per-device and, on a
  device-isolated box, each agent sees only its own GPUs. They are therefore
  accepted from EVERY agent regardless of ownership and stored as a per-agent
  partial ``im:metrics:machine:{machine_uid}:agent:{agent_id}`` (TTL 30s,
  refreshed each frame). ``read_machine_metrics`` merges the live partials
  per-GPU-UUID on read and recomputes the aggregates; a dead agent's partial
  self-expires with its TTL.
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

# Phase 17 category split (mirrors provider_lib/metrics.py). GPU categories
# merge per-agent across every agent; machine-wide categories stay owner-gated.
GPU_CATEGORIES = frozenset({"vram", "gpu_usage"})
MACHINE_WIDE_CATEGORIES = frozenset({"os_ram", "cpu", "storage"})


def _as_int(value: Any) -> int:
    """Coerce a metrics field to ``int`` (non-numeric / null / bool → 0).

    Mirrors the provider collectors' failure tolerance so a malformed partial
    never raises on read.
    """
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):  # fmt: skip
        return 0


def _as_float(value: Any) -> float:
    """Coerce a metrics field to ``float`` (non-numeric / null / bool → 0.0)."""
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):  # fmt: skip
        return 0.0


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
    """Try to make ``agent_id`` the machine-wide metrics owner of its machine.

    Governs the single-owner lease for the MACHINE-WIDE categories
    (``os_ram``/``cpu``/``storage``) only — GPU categories are accepted from
    every agent regardless of ownership (Phase 17). An agent that declares no
    machine-wide category (e.g. ``METRICS_CATEGORIES="vram gpu_usage"``) must
    NOT claim the lease: it emits no machine-wide sections yet would keep the
    lease alive forever via GPU frames, starving a later machine-wide-capable
    agent (SET NX fails and its machine-wide frames get dropped as non-owner).

    Returns True when the lease was newly acquired (and the
    ``metrics.assign`` command was sent). No-op when another agent already
    owns the machine or the agent declares no machine-wide categories.
    """
    machine_uid = _machine_uid_for_agent(agent_id)
    if machine_uid is None:
        logger.debug("metrics assign: unknown agent %s", agent_id)
        return False

    redis_client = _resolve_redis(app)
    categories_json = await redis_client.get(redis_keys.metrics_cats_key(agent_id))
    categories: list[str] = json.loads(categories_json) if categories_json else []
    if not set(categories) & MACHINE_WIDE_CATEGORIES:
        # Nothing machine-wide to report; don't claim the lease.
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
    """Drop the machine-wide metrics-owner lease if ``agent_id`` holds it."""
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
    """Store a ``metrics.machine`` frame under the Phase 17 split semantics.

    GPU categories (``vram``/``gpu_usage``) are written to a per-agent partial
    regardless of ownership; machine-wide categories (``os_ram``/``cpu``/
    ``storage``) are stored only when the sender is the current owner, which
    also refreshes the owner lease. Returns True when anything was stored.
    """
    machine_uid = _machine_uid_for_agent(agent_id)
    if machine_uid is None:
        return False
    redis_client = _resolve_redis(app)

    stored = False

    # GPU portion: accepted from every agent, no owner gate. The partial stays
    # minimal (vram/gpu_usage only) — attribution on read comes from the
    # unioned ``vram.gpus`` uuids, so ``assigned_gpus`` is not stored.
    gpu_part: dict[str, Any] = {k: payload[k] for k in GPU_CATEGORIES if k in payload}
    if gpu_part:
        await redis_client.set(
            redis_keys.metrics_agent_partial_key(machine_uid, agent_id),
            json.dumps(gpu_part),
            ex=redis_keys.METRICS_OWNER_TTL_SECONDS,
        )
        stored = True

    # Machine-wide portion: owner-gated; refreshes the lease on receipt.
    owner_key = redis_keys.metrics_owner_key(machine_uid)
    owner = await redis_client.get(owner_key)
    if owner == agent_id:
        await redis_client.set(
            owner_key, agent_id, ex=redis_keys.METRICS_OWNER_TTL_SECONDS
        )
        mw_part: dict[str, Any] = {
            k: payload[k] for k in MACHINE_WIDE_CATEGORIES if k in payload
        }
        if mw_part:
            snapshot = {**mw_part, "owner_agent_id": agent_id}
            await redis_client.set(
                redis_keys.metrics_machine_key(machine_uid),
                json.dumps(snapshot),
                ex=redis_keys.METRICS_OWNER_TTL_SECONDS,
            )
        stored = True
    elif any(k in payload for k in MACHINE_WIDE_CATEGORIES):
        logger.debug(
            "dropping machine-wide metrics.machine sections from non-owner "
            "agent %s (owner=%s)",
            agent_id,
            owner,
        )

    return stored


async def read_machine_metrics(redis_client: Any, machine_uid: str) -> dict[str, Any]:
    """Merge live per-agent GPU partials + the owner machine-wide snapshot.

    Union GPUs by UUID across every agent partial (latest-writer-wins per
    UUID), recompute the ``vram``/``gpu_usage`` aggregates from that union, and
    overlay the owner-gated machine-wide sections. Returns a minimal dict
    (just ``machine_uid``) when nothing is present.
    """
    prefix = redis_keys.metrics_agent_partial_prefix(machine_uid)
    union: dict[str, dict[str, Any]] = {}
    reporting: set[str] = set()

    async for key in redis_client.scan_iter(match=f"{prefix}*"):
        raw = await redis_client.get(key)
        if not raw:
            continue
        partial_agent_id = key[len(prefix) :]
        try:
            part = json.loads(raw)
        except (TypeError, ValueError):  # fmt: skip
            logger.warning("skipping malformed GPU partial %s", key)
            continue
        contributed = False
        for gpu in (part.get("vram") or {}).get("gpus", []) or []:
            # Key on the real ``uuid`` and skip entries without one, matching
            # ``hardware.merge_hardware_union`` so the live ``gpu_count`` agrees
            # with the Postgres inventory (L1).
            if not isinstance(gpu, dict) or not gpu.get("uuid"):
                continue
            entry = dict(gpu)
            entry["agent_id"] = partial_agent_id
            union[str(gpu["uuid"])] = entry
            contributed = True
        # Only count an agent that actually yielded a GPU into the union, so
        # ``reporting_agents`` never disagrees with ``gpu_count`` (M2).
        if contributed:
            reporting.add(partial_agent_id)

    result: dict[str, Any] = {"machine_uid": machine_uid}

    if union:
        gpus = list(union.values())
        total = sum(_as_int(g.get("vram_total")) for g in gpus)
        used = sum(_as_int(g.get("vram_used")) for g in gpus)
        free = sum(_as_int(g.get("vram_free")) for g in gpus)
        result["vram"] = {
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "gpu_count": len(gpus),
            "gpus": gpus,
        }
        utils = [_as_float(g.get("utilization")) for g in gpus]
        result["gpu_usage"] = {
            "utilization_percent": round(sum(utils) / len(utils), 2) if utils else 0.0,
            "gpus": [
                {
                    "uuid": uid,
                    "id": g.get("id"),
                    "utilization": g.get("utilization"),
                    "agent_id": g["agent_id"],
                }
                for uid, g in union.items()
            ],
        }

    mw_raw = await redis_client.get(redis_keys.metrics_machine_key(machine_uid))
    if mw_raw:
        try:
            mw = json.loads(mw_raw)
        except (TypeError, ValueError):  # fmt: skip
            mw = {}
        for category in MACHINE_WIDE_CATEGORIES:
            if category in mw:
                result[category] = mw[category]
        if "owner_agent_id" in mw:
            result["owner_agent_id"] = mw["owner_agent_id"]

    if reporting:
        result["reporting_agents"] = sorted(reporting)

    return result

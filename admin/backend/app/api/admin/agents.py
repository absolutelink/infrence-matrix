"""Admin provider-agent reads + delete (Phase 16).

``/admin/api/agents`` — listing of ``ProviderAgent`` rows (the hardware-local
containers bound to a (machine, provider_type, agent_id) triple). Backends
(``ProviderInstance``) nest under their agent so the UI can show what each
container hosts. Placement cherry-picking on definitions references these
agent ids.

Agents are created (and re-resolved) by ``POST /admin/api/providers/register``;
this endpoint reads whatever rows exist and deletes decommissioned/renamed
ones (see ``DELETE /admin/api/agents/{agent_id}``).
"""

import logging
import uuid
from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, col, delete, select

from app.api.admin.serializers import iso_utc
from app.core.db import get_session
from app.core.redis import get_redis
from app.models import DefinitionAgent, Machine, ProviderAgent, ProviderInstance
from app.services import redis_keys
from app.services.hardware import normalize_gpu_uuids, sum_gpu_vram

logger = logging.getLogger("admin.agents")

router = APIRouter(prefix="/admin/api/agents", tags=["admin"])


def _agent_dict(
    agent: ProviderAgent,
    *,
    machine_uid: str | None = None,
    backends: list[dict[str, Any]] | None = None,
    definition_count: int | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": str(agent.id),
        "machine_id": str(agent.machine_id),
        "machine_uid": machine_uid,
        "provider_type": agent.provider_type,
        "agent_id": agent.agent_id,
        "base_port": agent.base_port,
        "version": agent.version,
        "agent_status": agent.agent_status,
        "websocket_connected": agent.websocket_connected,
        "epoch": agent.epoch,
        "reported_schema_fingerprint": agent.reported_schema_fingerprint,
        "assigned_gpus": agent.assigned_gpus,
        "error_message": agent.error_message,
        "last_seen": iso_utc(agent.last_seen),
        "created_at": iso_utc(agent.created_at),
        "updated_at": iso_utc(agent.updated_at),
    }
    if backends is not None:
        d["backends"] = backends
    if definition_count is not None:
        d["definition_count"] = definition_count
    return d


def _backend_dicts(session: Session, agent_id: uuid.UUID) -> list[dict[str, Any]]:
    rows = session.exec(
        select(ProviderInstance).where(ProviderInstance.agent_id == agent_id)
    ).all()
    return [
        {
            "id": str(i.id),
            "provider_definition_id": str(i.provider_definition_id),
            "backend_status": i.backend_status,
            "config_fingerprint": i.config_fingerprint,
        }
        for i in rows
    ]


def _definition_count(session: Session, agent_id: uuid.UUID) -> int:
    return len(
        session.exec(
            select(DefinitionAgent).where(DefinitionAgent.agent_id == agent_id)
        ).all()
    )


def _recompute_machine_union(
    session: Session,
    machine: Machine,
    deleted_uuids: set[str],
    surviving_uuids: set[str],
) -> None:
    """Drop a deleted agent's GPUs from the machine union (Phase 17 B).

    A GPU is removed only when no *remaining* agent on the machine still claims
    it (``deleted_uuids - surviving_uuids``), then ``total_vram_bytes`` is
    recomputed as the auto-sum over the surviving list. No-op when the machine
    carries no ``gpus`` list (nothing to prune, and a directly-set budget is
    preserved). Redis metrics partials are out of scope (Slice 3).
    """
    hardware = dict(machine.hardware or {})
    gpus = hardware.get("gpus")
    if not isinstance(gpus, list):
        return
    to_remove = deleted_uuids - surviving_uuids
    survivors = [
        g for g in gpus if not (isinstance(g, dict) and g.get("uuid") in to_remove)
    ]
    total = sum_gpu_vram(survivors)
    hardware["gpus"] = survivors
    hardware["total_vram_bytes"] = total
    machine.hardware = hardware
    machine.total_vram_bytes = total
    session.add(machine)


@router.get("")
def list_agents(session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    agents = session.exec(
        select(ProviderAgent).order_by(ProviderAgent.machine_id, ProviderAgent.agent_id)
    ).all()
    out: list[dict[str, Any]] = []
    for agent in agents:
        out.append(
            _agent_dict(
                agent,
                machine_uid=agent.machine.uid if agent.machine else None,
                backends=_backend_dicts(session, agent.id),
                definition_count=_definition_count(session, agent.id),
            )
        )
    return out


@router.get("/{agent_id}")
def get_agent(agent_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    try:
        agent = session.get(ProviderAgent, uuid.UUID(agent_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="agent not found") from None
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return _agent_dict(
        agent,
        machine_uid=agent.machine.uid if agent.machine else None,
        backends=_backend_dicts(session, agent.id),
        definition_count=_definition_count(session, agent.id),
    )


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str,
    request: Request,
    session: Session = Depends(get_session),
    redis_client: aioredis.Redis = Depends(get_redis),
) -> dict[str, Any]:
    """Delete a decommissioned/renamed provider agent (Phase 16).

    Safety gate: a **connected** agent is never deleted (409) — stop/redeploy
    the container first. A disconnected agent's backends are unschedulable
    ghosts regardless of their DB ``backend_status`` (the real-world case: a
    renamed container left ``running`` ghost rows), so they are removed with it.

    NOTE: agents are Phase-12 schema-consensus voters — deleting one drops its
    vote from any pending schema for its type. That is intended here: this is
    for renamed/decommissioned containers whose rows linger as ghosts and which
    operators previously had to clear via raw SQL + Redis.
    """
    try:
        agent_uuid_obj = uuid.UUID(agent_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="agent not found") from None

    # Existence check first so a missing row is a 404 (not conflated with the
    # 409 connected-gate below).
    agent = session.get(ProviderAgent, agent_uuid_obj)
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")

    # Race-proof connected-gate: lock the row FOR UPDATE and re-check the
    # disconnected flag inside this same transaction. A plain read would let a
    # container that restarts + reconnects between the check and the commit flip
    # ``websocket_connected`` True and get deleted out from under us (leaving an
    # orphaned live socket). The FOR UPDATE lock serializes that flip until our
    # commit, so a reconnected agent is refused here (409) rather than deleted.
    deletable = session.exec(
        select(ProviderAgent)
        .where(
            ProviderAgent.id == agent_uuid_obj,
            col(ProviderAgent.websocket_connected) == False,  # noqa: E712
        )
        .with_for_update()
    ).first()
    if deletable is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"agent '{agent.agent_id}' is connected; "
                "stop/redeploy it before deleting"
            ),
        )

    agent_uuid = str(agent.id)
    machine = agent.machine
    machine_uid = machine.uid if machine else None
    # Phase 17 B: capture the deleted agent's GPU attribution and the union of
    # every *other* agent on the machine so the machine inventory can be
    # recomputed after the delete (a GPU is dropped only if no survivor claims it).
    deleted_uuids = set(normalize_gpu_uuids(agent.assigned_gpus))
    surviving_uuids: set[str] = set()
    if machine is not None:
        for other in session.exec(
            select(ProviderAgent).where(
                ProviderAgent.machine_id == machine.id,
                ProviderAgent.id != agent.id,
            )
        ).all():
            surviving_uuids.update(normalize_gpu_uuids(other.assigned_gpus))
    instance_ids = [
        str(i.id)
        for i in session.exec(
            select(ProviderInstance).where(ProviderInstance.agent_id == agent.id)
        ).all()
    ]

    # Redis cleanup BEFORE the DB commit: the DB delete is the step that can
    # realistically fail (FK constraints), whereas a Redis blip here would
    # otherwise leave the ghost keys (long-TTL secret, never-expiring epoch)
    # behind with no row to retry against. Clearing first means a 200 always
    # implies both Postgres and Redis are clean.
    #
    # Failure path (do NOT reorder Redis after the commit): if the DB commit
    # fails AFTER Redis was cleared, the surviving disconnected agent's cached
    # secret (its ``provider_config.json``) no longer matches Redis, so a WS
    # reconnect is rejected at auth until the container re-registers (restart)
    # — self-healing, and the DELETE is idempotent (a retry re-clears + deletes).
    # The reverse ordering is strictly worse: a Redis failure after commit would
    # leak the never-expiring ``im:ws:epoch`` key with no row left to clean it.
    #
    # ``im:metrics:owner:{machine_uid}`` / ``im:metrics:machine:{machine_uid}``
    # are deliberately NOT cleared: they are 30s-TTL leases that self-expire.
    await redis_client.delete(
        redis_keys.secret_key(agent_uuid),
        redis_keys.epoch_key(agent_uuid),
        redis_keys.owner_key(agent_uuid),
        redis_keys.presence_key(agent_uuid),
        redis_keys.metrics_cats_key(agent_uuid),
    )

    # The ORM does NOT cascade ProviderAgent -> children: the relationships
    # carry no delete-orphan cascade and no passive_deletes, so session.delete
    # tries to NULL the non-nullable child FKs and raises before the DB
    # ON DELETE CASCADE can fire. Delete the children explicitly in the same
    # transaction.
    session.exec(delete(DefinitionAgent).where(DefinitionAgent.agent_id == agent.id))
    session.exec(delete(ProviderInstance).where(ProviderInstance.agent_id == agent.id))
    session.delete(agent)
    # Recompute the machine GPU union in the SAME transaction as the delete so a
    # committed 200 always implies the inventory no longer credits the removed
    # agent's GPUs (unless a survivor still claims them).
    if machine is not None:
        _recompute_machine_union(session, machine, deleted_uuids, surviving_uuids)
    session.commit()

    # Scheduler hygiene: free any in-process VRAM/_booted hold the removed
    # ghost backends still carry so a deleted agent cannot linger in the
    # ledger. ``note_backend_stopped`` is the existing external-stop hook and
    # is idempotent (no-op when this process never held the instance).
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is not None and machine_uid is not None:
        for instance_id in instance_ids:
            await scheduler.note_backend_stopped(instance_id, machine_uid)

    logger.info(
        "deleted agent %s (machine %s) with %d backend(s)",
        agent_id,
        machine_uid,
        len(instance_ids),
    )
    return {"ok": True, "agent_id": agent_id, "deleted_instances": len(instance_ids)}

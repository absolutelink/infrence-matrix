"""Admin response-log + usage reads (Phase 10 UI).

Read-only views over the stored inference records for the dashboard,
Responses/Usage page, and settings overview:

- ``GET /admin/api/responses`` — recent ``ResponseRecord`` turns
  (paginated, newest first).
- ``GET /admin/api/stats/usage`` — recent ``TokenUsageSample`` rows +
  aggregate totals (for the usage chart + dashboard cards). Samples carry
  the definition ``alias`` for per-model rollups (Phase 19).
- ``GET /admin/api/stats/overview`` — dashboard aggregates: machines,
  definitions, instances, live scheduler queue/active counts (Redis
  mirror), and token totals.
- ``GET /admin/api/stats/scheduler`` — per-definition queue/active counts
  from the authoritative in-process scheduler state (Phase 19).
- ``DELETE /admin/api/stats/scheduler/queue/{alias}`` /
  ``DELETE /admin/api/stats/scheduler/queue`` — operator drain of queued
  waiters for one alias or all (Phase 19).
- ``GET /admin/api/stats/metrics`` — fleet VRAM/GPU rollup across every
  machine's merged metrics (Phase 19).

These join the Phase 9 CRUD surfaces under ``/admin/api`` (trusted LAN,
unauthenticated).
"""

import logging
from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy import func
from sqlmodel import Session, select

from app.api.admin.serializers import iso_utc
from app.api.v1.responses import get_scheduler
from app.core.db import get_session
from app.core.redis import get_redis
from app.models import (
    Machine,
    ProviderAgent,
    ProviderDefinition,
    ProviderInstance,
    ResponseRecord,
    TokenUsageSample,
)
from app.services import metrics_service, redis_keys
from app.services.scheduler import InferenceScheduler

logger = logging.getLogger("admin.responses")

router = APIRouter(prefix="/admin/api", tags=["admin"])


def response_dict(record: ResponseRecord) -> dict[str, Any]:
    api_format = (record.parameters or {}).get("api_format", "responses")
    return {
        "id": str(record.id),
        "response_id": record.response_id,
        "previous_response_id": record.previous_response_id,
        "model_alias": (
            record.provider_definition.alias if record.provider_definition else None
        ),
        "provider_type": (
            record.provider_definition.provider_type
            if record.provider_definition
            else None
        ),
        "provider_instance_id": (
            str(record.provider_instance_id) if record.provider_instance_id else None
        ),
        "api_format": api_format,
        "status": record.status,
        "error_code": record.error_code,
        "error_message": record.error_message,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "total_tokens": record.total_tokens,
        "store": record.store,
        "created_at": iso_utc(record.created_at),
        "completed_at": iso_utc(record.completed_at),
    }


def usage_sample_dict(
    sample: TokenUsageSample, alias: str | None = None
) -> dict[str, Any]:
    return {
        "id": str(sample.id),
        "provider_instance_id": (
            str(sample.provider_instance_id) if sample.provider_instance_id else None
        ),
        "provider_definition_id": (
            str(sample.provider_definition_id)
            if sample.provider_definition_id
            else None
        ),
        "alias": alias,
        "prompt_tokens": sample.prompt_tokens,
        "cached_tokens": sample.cached_tokens,
        "completion_tokens": sample.completion_tokens,
        "prompt_ms": sample.prompt_ms,
        "predicted_ms": sample.predicted_ms,
        "prompt_per_second": sample.prompt_per_second,
        "predicted_per_second": sample.predicted_per_second,
        "created_at": iso_utc(sample.created_at),
    }


@router.get("/responses")
def list_responses(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    total = session.exec(select(func.count()).select_from(ResponseRecord)).one()
    records = session.exec(
        select(ResponseRecord)
        .order_by(ResponseRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    return {
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "responses": [response_dict(r) for r in records],
    }


@router.get("/stats/usage")
def usage_stats(
    limit: int = Query(default=100, ge=1, le=1000),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    totals = session.exec(
        select(
            func.coalesce(func.sum(TokenUsageSample.prompt_tokens), 0),
            func.coalesce(func.sum(TokenUsageSample.cached_tokens), 0),
            func.coalesce(func.sum(TokenUsageSample.completion_tokens), 0),
        )
    ).one()
    samples = session.exec(
        select(TokenUsageSample, ProviderDefinition.alias)
        .outerjoin(
            ProviderDefinition,
            TokenUsageSample.provider_definition_id == ProviderDefinition.id,
        )
        .order_by(TokenUsageSample.created_at.desc())
        .limit(limit)
    ).all()
    return {
        "totals": {
            "prompt_tokens": int(totals[0]),
            "cached_tokens": int(totals[1]),
            "completion_tokens": int(totals[2]),
        },
        "samples": [usage_sample_dict(sample, alias) for sample, alias in samples],
    }


@router.get("/stats/overview")
async def overview_stats(
    session: Session = Depends(get_session),
    redis_client: aioredis.Redis = Depends(get_redis),
) -> dict[str, Any]:
    machine_count = session.exec(select(func.count()).select_from(Machine)).one()
    definition_count = session.exec(
        select(func.count()).select_from(ProviderDefinition)
    ).one()
    enabled_count = session.exec(
        select(func.count())
        .select_from(ProviderDefinition)
        .where(ProviderDefinition.enabled == True)  # noqa: E712
    ).one()
    instance_total = session.exec(
        select(func.count()).select_from(ProviderInstance)
    ).one()
    connected_count = session.exec(
        select(func.count())
        .select_from(ProviderInstance)
        .join(ProviderAgent, ProviderInstance.agent_id == ProviderAgent.id)
        .where(ProviderAgent.websocket_connected == True)  # noqa: E712
    ).one()
    backend_running = session.exec(
        select(func.count())
        .select_from(ProviderInstance)
        .where(ProviderInstance.backend_status.in_(("running", "in_use")))  # type: ignore[attr-defined]
    ).one()

    # Live inference counts from the scheduler's Redis mirror (best
    # effort: the in-process deque is authoritative; the mirror lags by
    # at most one transition).
    queued = 0
    active = 0
    aliases = session.exec(
        select(ProviderDefinition.alias).where(
            ProviderDefinition.enabled == True  # noqa: E712
        )
    ).all()
    for alias in aliases:
        try:
            queued += int(await redis_client.llen(redis_keys.sched_queue_key(alias)))
            active += int(await redis_client.scard(redis_keys.sched_active_key(alias)))
        except Exception:  # noqa: BLE001 - mirror is observability only
            logger.debug("overview: redis mirror read failed for alias %s", alias)

    token_totals = session.exec(
        select(
            func.coalesce(func.sum(ResponseRecord.input_tokens), 0),
            func.coalesce(func.sum(ResponseRecord.output_tokens), 0),
            func.coalesce(func.sum(ResponseRecord.total_tokens), 0),
        )
    ).one()

    return {
        "machines": int(machine_count),
        "definitions": int(definition_count),
        "definitions_enabled": int(enabled_count),
        "instances": int(instance_total),
        "instances_connected": int(connected_count),
        "backends_running": int(backend_running),
        "queued_requests": queued,
        "active_requests": active,
        "input_tokens": int(token_totals[0]),
        "output_tokens": int(token_totals[1]),
        "total_tokens": int(token_totals[2]),
    }


# ---------------------------------------------------------------------------
# Phase 19 — live stats bar: scheduler queue observability + operator clear
# ---------------------------------------------------------------------------


@router.get("/stats/scheduler")
async def scheduler_stats(
    session: Session = Depends(get_session),
    scheduler: InferenceScheduler = Depends(get_scheduler),
) -> dict[str, Any]:
    """Per-definition queue/active counts from the authoritative in-process
    scheduler state (never the Redis mirror). Every ``ProviderDefinition`` is
    listed — enabled or not — so the operator sees all queues; a definition
    with no scheduler state reports ``{queued: 0, active: 0}``.
    """
    snapshot = await scheduler.queue_snapshot()
    by_alias = {str(row["alias"]): row for row in snapshot}
    definitions = session.exec(
        select(ProviderDefinition).order_by(ProviderDefinition.alias)
    ).all()

    out_definitions: list[dict[str, Any]] = []
    total_queued = 0
    total_active = 0
    for definition in definitions:
        row = by_alias.get(definition.alias)
        queued = int(row["queued"]) if row is not None else 0
        active = int(row["active"]) if row is not None else 0
        out_definitions.append(
            {"alias": definition.alias, "queued": queued, "active": active}
        )
        total_queued += queued
        total_active += active

    return {
        "definitions": out_definitions,
        "totals": {"queued": total_queued, "active": total_active},
    }


@router.delete("/stats/scheduler/queue")
async def clear_all_scheduler_queues(
    scheduler: InferenceScheduler = Depends(get_scheduler),
) -> dict[str, Any]:
    """Operator drain: cancel every queued waiter across all aliases.

    Admitted/active requests, boots, and VRAM holds are untouched (see
    ``InferenceScheduler.clear_queue``).
    """
    cleared = await scheduler.clear_queue(None)
    return {"ok": True, "cleared": cleared}


@router.delete("/stats/scheduler/queue/{alias}")
async def clear_scheduler_queue(
    alias: str = Path(..., max_length=200),
    session: Session = Depends(get_session),
    scheduler: InferenceScheduler = Depends(get_scheduler),
) -> dict[str, Any]:
    """Operator drain for a single alias's queue. 404 when no definition has
    that alias; otherwise clears its waiters (admitted slots untouched).
    """
    exists = session.exec(
        select(ProviderDefinition).where(ProviderDefinition.alias == alias)
    ).first()
    if exists is None:
        raise HTTPException(status_code=404, detail="definition not found")
    cleared = await scheduler.clear_queue(alias)
    return {"ok": True, "alias": alias, "cleared": cleared}


# ---------------------------------------------------------------------------
# Phase 19 — fleet metrics rollup (VRAM + GPU utilization)
# ---------------------------------------------------------------------------


@router.get("/stats/metrics")
async def fleet_metrics(
    session: Session = Depends(get_session),
    redis_client: aioredis.Redis = Depends(get_redis),
) -> dict[str, Any]:
    """Fleet VRAM/GPU rollup for the UI stats bar.

    For every machine, read its merged live metrics via the same path as
    ``GET /admin/api/machines/{id}/metrics`` (``read_machine_metrics``), then
    sum VRAM and average GPU utilization fleet-wide. Machines with no fresh
    metrics degrade to zeros (VRAM total falls back to the DB
    ``Machine.total_vram_bytes``); a metrics read failure never crashes the
    route.

    ``gpu_utilization_percent`` is the **unweighted mean of the per-machine
    means** (each machine's own value is already the mean across its GPUs),
    deliberately — every machine counts equally regardless of GPU count, which
    is what the stats bar wants. Machines that report no ``gpu_usage`` are
    excluded from the mean (0.0 when none report).
    """
    machines = session.exec(select(Machine).order_by(Machine.uid)).all()

    fleet_used = 0
    fleet_total = 0
    gpu_utils: list[float] = []
    per_machine: list[dict[str, Any]] = []

    for machine in machines:
        try:
            merged = await metrics_service.read_machine_metrics(
                redis_client, machine.uid
            )
        except Exception:  # noqa: BLE001 - observability read, degrade to zeros
            logger.debug("stats/metrics: read failed for machine %s", machine.uid)
            merged = {}

        vram = merged.get("vram") or {}
        used = int(vram.get("used_bytes", 0) or 0)
        total = int(vram.get("total_bytes", 0) or 0)
        if total == 0:
            total = int(machine.total_vram_bytes or 0)

        gpu_usage = merged.get("gpu_usage")
        util: float | None = None
        if isinstance(gpu_usage, dict) and "utilization_percent" in gpu_usage:
            util = float(gpu_usage["utilization_percent"])
            gpu_utils.append(util)

        fleet_used += used
        fleet_total += total
        per_machine.append(
            {
                "uid": machine.uid,
                "name": machine.name,
                "vram_used_bytes": used,
                "vram_total_bytes": total,
                "gpu_utilization_percent": util if util is not None else 0.0,
            }
        )

    fleet_gpu = round(sum(gpu_utils) / len(gpu_utils), 2) if gpu_utils else 0.0
    return {
        "vram": {"used_bytes": fleet_used, "total_bytes": fleet_total},
        "gpu_utilization_percent": fleet_gpu,
        "machines": per_machine,
    }

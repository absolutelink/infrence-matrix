"""Admin response-log + usage reads (Phase 10 UI).

Read-only views over the stored inference records for the dashboard,
Responses/Usage page, and settings overview:

- ``GET /admin/api/responses`` — recent ``ResponseRecord`` turns
  (paginated, newest first).
- ``GET /admin/api/stats/usage`` — recent ``TokenUsageSample`` rows +
  aggregate totals (for the usage chart + dashboard cards).
- ``GET /admin/api/stats/overview`` — dashboard aggregates: machines,
  definitions, instances, live scheduler queue/active counts (Redis
  mirror), and token totals.

These join the Phase 9 CRUD surfaces under ``/admin/api`` (trusted LAN,
unauthenticated).
"""

import logging
from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlmodel import Session, select

from app.api.admin.serializers import iso_utc
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
from app.services import redis_keys

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


def usage_sample_dict(sample: TokenUsageSample) -> dict[str, Any]:
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
        select(TokenUsageSample)
        .order_by(TokenUsageSample.created_at.desc())
        .limit(limit)
    ).all()
    return {
        "totals": {
            "prompt_tokens": int(totals[0]),
            "cached_tokens": int(totals[1]),
            "completion_tokens": int(totals[2]),
        },
        "samples": [usage_sample_dict(s) for s in samples],
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

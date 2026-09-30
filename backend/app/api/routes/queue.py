"""Inference queue administration endpoints."""

from fastapi import APIRouter, HTTPException
from sqlmodel import select

from app.db.session import AsyncSessionMaker
from app.models import InferenceLease
from app.services.inference_scheduler import inference_scheduler

router = APIRouter(prefix="/queue", tags=["queue"])


@router.get("/{request_id}")
async def inference_request_status(request_id: str) -> dict[str, str | None]:
    """Report one request's queue/admission status without exposing its prompt."""
    async with AsyncSessionMaker() as session:
        result = await session.exec(
            select(InferenceLease).where(InferenceLease.request_id == request_id)
        )
        lease = result.one_or_none()
    if lease is None:
        raise HTTPException(404, "Inference request not found")
    return {
        "request_id": lease.request_id,
        "status": lease.status,
        "terminal_reason": lease.terminal_reason,
    }


@router.post("/clear")
async def clear_inference_queue() -> dict[str, int | str]:
    """Cancel queued requests while allowing active requests to finish."""
    cancelled = await inference_scheduler.clear_queue()
    return {"status": "cleared", "cancelled": cancelled}

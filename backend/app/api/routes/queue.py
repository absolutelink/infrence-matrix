"""Inference queue administration endpoints."""

from fastapi import APIRouter

from app.services.inference_scheduler import inference_scheduler

router = APIRouter(prefix="/queue", tags=["queue"])


@router.post("/clear")
async def clear_inference_queue() -> dict[str, int | str]:
    """Cancel queued requests while allowing active requests to finish."""
    cancelled = await inference_scheduler.clear_queue()
    return {"status": "cleared", "cancelled": cancelled}

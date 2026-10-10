"""Public voice catalog: ``GET /v1/audio/voices`` (Phase 24; boot-on-demand
added with Phase 25 multi-model).

Proxies the **live** voice catalog of a ``tts`` backend (the source of truth —
the admin's ``TTSVoice`` rows are only the UI catalog). Like every other /v1
endpoint this **boots the backend on demand** via the scheduler: a catalog read
on a stopped/idle-reaped backend must not dead-end clients (e.g. the admin
playground's voice dropdown) into an empty list. The slot is released in a
cancellation-safe ``finally``, mirroring the speech routes. An unknown /
non-tts / disabled alias is a clean 404; no connected provider answers 503 and
a queue timeout 504.

Auth model: trusted LAN — this public route is unauthenticated.
"""

import asyncio
import contextlib
import logging
import uuid

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from sqlmodel import Session

from app.api.v1.responses import get_scheduler
from app.core.config import settings
from app.core.db import engine
from app.services.scheduler import NoProviderAvailable, QueueCleared, QueueTimeout
from app.services.served_models import resolve_served

logger = logging.getLogger("admin.v1.audio_voices")

router = APIRouter(tags=["audio"])


@router.get("/v1/audio/voices")
async def list_voices(
    request: Request, model: str = Query(..., description="tts alias")
) -> Response:
    alias = model
    with Session(engine) as session:
        resolved = resolve_served(session, alias)
        if resolved is None:
            raise HTTPException(
                status_code=404, detail=f"unknown model alias '{alias}'"
            )
        definition, spec = resolved
        if not definition.enabled:
            raise HTTPException(
                status_code=404, detail=f"model alias '{alias}' is disabled"
            )
        if spec.modality != "tts":
            raise HTTPException(
                status_code=404,
                detail=f"model alias '{alias}' is not a speech (tts) model",
            )

    request_id = f"voices-{uuid.uuid4().hex}"
    scheduler = get_scheduler(request)
    try:
        admission = await scheduler.acquire(alias, request_id)
    except (NoProviderAvailable, QueueCleared) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QueueTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc

    try:
        async with httpx.AsyncClient(
            timeout=settings.AUDIO_REQUEST_TIMEOUT_SECONDS
        ) as client:
            upstream = await client.get(
                f"{admission.base_url}/v1/audio/voices", params={"model": alias}
            )
    except httpx.HTTPError as exc:
        logger.warning("voices proxy failed for %s: %s", alias, exc)
        raise HTTPException(
            status_code=502, detail=f"voices upstream unreachable: {exc}"
        ) from exc
    finally:
        # Cancellation-safe release: the slot must not leak even if the
        # client disconnects mid-boot (shield mirrors speech/embeddings).
        with contextlib.suppress(Exception):
            await asyncio.shield(scheduler.release(alias, request_id))

    # Pass the agent's response through untouched (200 catalog or 503 not-ready).
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )

"""Public voice catalog: ``GET /v1/audio/voices`` (Phase 24).

Proxies the **live** voice catalog of a ``tts`` backend (the source of truth —
the admin's ``TTSVoice`` rows are only the UI catalog). No scheduler slot is
taken (the agent's voices route is serving-gated but slot-free), but the target
backend must be up: an agent that reports the backend not-ready answers 503,
which is passed through. An unknown / non-tts alias is a clean 404.

Auth model: trusted LAN — this public route is unauthenticated.
"""

import logging

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from sqlmodel import Session, select

from app.core.config import settings
from app.core.db import engine
from app.models import ProviderDefinition
from app.services import audio as audio_svc

logger = logging.getLogger("admin.v1.audio_voices")

router = APIRouter(tags=["audio"])


@router.get("/v1/audio/voices")
async def list_voices(model: str = Query(..., description="tts alias")) -> Response:
    alias = model
    with Session(engine) as session:
        definition = session.exec(
            select(ProviderDefinition).where(ProviderDefinition.alias == alias)
        ).first()
        if definition is None:
            raise HTTPException(
                status_code=404, detail=f"unknown model alias '{alias}'"
            )
        if not definition.enabled:
            raise HTTPException(
                status_code=404, detail=f"model alias '{alias}' is disabled"
            )
        if definition.modality != "tts":
            raise HTTPException(
                status_code=404,
                detail=f"model alias '{alias}' is not a speech (tts) model",
            )
        base_url = audio_svc.agent_base_url_for_definition(session, definition)

    if base_url is None:
        raise HTTPException(
            status_code=503,
            detail=f"no connected provider agent for alias '{alias}'",
        )

    try:
        async with httpx.AsyncClient(
            timeout=settings.AUDIO_REQUEST_TIMEOUT_SECONDS
        ) as client:
            upstream = await client.get(
                f"{base_url}/v1/audio/voices", params={"model": alias}
            )
    except httpx.HTTPError as exc:
        logger.warning("voices proxy failed for %s: %s", alias, exc)
        raise HTTPException(
            status_code=502, detail=f"voices upstream unreachable: {exc}"
        ) from exc

    # Pass the agent's response through untouched (200 catalog or 503 not-ready).
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )

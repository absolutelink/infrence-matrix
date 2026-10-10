"""Public voice catalog: ``GET /v1/audio/voices`` (Phase 24; boot-on-demand
added with Phase 25 multi-model; cache-first with this change).

The **live** agent catalog is the source of truth (the admin's ``TTSVoice``
rows are only the UI catalog). To keep a catalog read off the boot path when
possible, the route resolves in this order, **falling through** on any failure
of an earlier path so a client never dead-ends when a cache exists:

1. **A loaded backend exists** (some connected agent's instance for the
   definition is running/in_use): proxy **slot-free** with a direct httpx GET to
   that backend (no scheduler admission — nothing to release). A 200 refreshes
   the cache and is passed through. A transport failure OR a non-2xx (e.g. the
   backend raced to stopped and 503s) is treated as a cache miss and falls
   through — the cache / boot paths get a chance before any error surfaces.
2. **No live 200 but a cache exists**: serve the cached voice list from
   ``ProviderDefinition.model_metadata`` (populated by the talkies driver's
   ``backend.metadata`` scrape) with ``"cached": true`` — no boot at all.
3. **No live 200 and no cache**: fall back to boot-on-demand through the
   scheduler (a catalog read on a stopped/idle-reaped backend must not
   dead-end clients, e.g. the playground voice dropdown), releasing the slot in
   a cancellation-safe ``finally`` like the speech routes; a 200 refreshes the
   cache. This is the last resort: a transport failure here surfaces as 502 and
   a non-2xx is passed through untouched.

An unknown / disabled / non-tts alias is a clean 404; no connected provider
answers 503 and a queue timeout 504.

Auth model: trusted LAN — this public route is unauthenticated.
"""

import asyncio
import contextlib
import logging
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from sqlmodel import Session

from app.api.v1.responses import get_scheduler
from app.core.config import settings
from app.core.db import engine
from app.models import ProviderDefinition
from app.services import audio as audio_svc
from app.services.scheduler import NoProviderAvailable, QueueCleared, QueueTimeout
from app.services.served_models import resolve_served

logger = logging.getLogger("admin.v1.audio_voices")

router = APIRouter(tags=["audio"])


async def _proxy_voices(base_url: str, alias: str) -> httpx.Response | None:
    """Direct GET to a backend's ``/v1/audio/voices`` (fully read).

    Returns the response, or ``None`` on a transport failure (connection error /
    timeout) — the caller decides whether that is a fall-through (loaded path) or
    a 502 (final boot-on-demand path). A non-2xx HTTP response is returned as-is
    (it is a valid answer from a reachable backend, not a transport failure).
    """
    try:
        async with httpx.AsyncClient(
            timeout=settings.AUDIO_REQUEST_TIMEOUT_SECONDS
        ) as client:
            return await client.get(
                f"{base_url}/v1/audio/voices", params={"model": alias}
            )
    except httpx.HTTPError as exc:
        logger.warning("voices proxy failed for %s: %s", alias, exc)
        return None


def _passthrough(upstream: httpx.Response) -> Response:
    """Relay an agent response untouched (200 catalog or 503 not-ready)."""
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type"),
    )


def _extract_voices(upstream: httpx.Response) -> list[Any] | None:
    """The ``voices`` list from a 200 agent catalog, or ``None`` when the body
    is not JSON / carries no ``voices`` list (nothing to cache)."""
    try:
        data = upstream.json()
    except ValueError:
        return None
    if isinstance(data, dict):
        voices = data.get("voices")
        if isinstance(voices, list):
            return voices
    return None


def _refresh_cache(definition_id: uuid.UUID, alias: str, voices: list[Any]) -> None:
    """Persist a fresh live catalog onto the definition's cache."""
    with Session(engine) as session:
        definition = session.get(ProviderDefinition, definition_id)
        if definition is not None:
            audio_svc.cache_voices_for_definition(session, definition, alias, voices)


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
        definition_id = definition.id
        loaded_url = audio_svc.loaded_base_url_for_definition(session, definition)
        cached_voices = audio_svc.cached_voices_for_definition(definition, alias)

    # 1. A loaded backend: proxy slot-free (no scheduler slot taken). A 200 is
    #    the answer; a transport failure or non-2xx falls through (cache miss).
    if loaded_url is not None:
        upstream = await _proxy_voices(loaded_url, alias)
        if upstream is not None and upstream.status_code == 200:
            voices = _extract_voices(upstream)
            if voices is not None:
                _refresh_cache(definition_id, alias, voices)
            return _passthrough(upstream)

    # 2. No live 200 but a cache exists: serve it without booting.
    if cached_voices is not None:
        return JSONResponse({"voices": cached_voices, "cached": True}, status_code=200)

    # 3. No live 200 and no cache: boot on demand (last resort).
    request_id = f"voices-{uuid.uuid4().hex}"
    scheduler = get_scheduler(request)
    try:
        admission = await scheduler.acquire(alias, request_id)
    except (NoProviderAvailable, QueueCleared) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QueueTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc

    try:
        upstream = await _proxy_voices(admission.base_url, alias)
    finally:
        # Cancellation-safe release: the slot must not leak even if the
        # client disconnects mid-boot (shield mirrors speech/embeddings).
        with contextlib.suppress(Exception):
            await asyncio.shield(scheduler.release(alias, request_id))

    if upstream is None:
        raise HTTPException(
            status_code=502,
            detail=f"voices upstream unreachable for '{alias}' after boot",
        )
    if upstream.status_code == 200:
        voices = _extract_voices(upstream)
        if voices is not None:
            _refresh_cache(definition_id, alias, voices)
    return _passthrough(upstream)

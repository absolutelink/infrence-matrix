"""Public OpenAI TTS endpoint: ``POST /v1/audio/speech`` (Phase 24).

Mirrors the Phase 18 ``/v1/embeddings`` admission + release discipline
(``app/api/v1/embeddings.py``) but **bypasses litellm**: speech responses are
binary audio (optionally incremental PCM), which litellm's call surface cannot
carry, so the admin does a direct ``httpx`` passthrough to the agent's
``/v1/audio/speech`` (ARCHITECTURE.md §7 Audio specifics).

Differences from embeddings:

- The alias must resolve to a ``modality == "tts"`` definition; any other kind
  is a clean 404 (the reverse gate lives in the chat/embeddings routes).
- ``input`` must be a non-empty string (OpenAI TTS contract) → 400 otherwise.
- A 2xx upstream is **streamed through** as a ``StreamingResponse`` preserving
  ``Content-Type`` and ``X-Sample-Rate`` (incremental, no full buffering). The
  scheduler slot is held for the whole stream and released — shielded — when
  the upstream closes (normal end, error, or client disconnect), exactly like
  the Phase 7 chat streaming path.
- A non-2xx upstream relays its status + JSON error body (the slot releases
  immediately — nothing is streamed).
- Persistence is an ``AudioUsageSample`` only (``chars`` = ``len(input)``;
  ``audio_seconds`` sniffed from a WAV header when cheap, else null) — no
  ``ResponseRecord`` (synthesis is not a conversation turn).

Auth model: trusted LAN — this public route is unauthenticated.
"""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlmodel import Session, select

from app.api.v1.responses import get_scheduler
from app.core.config import settings
from app.core.db import engine
from app.models import AudioUsageSample, ProviderDefinition
from app.services import audio as audio_svc
from app.services.scheduler import (
    Admission,
    InferenceScheduler,
    NoProviderAvailable,
    QueueCleared,
    QueueTimeout,
)

logger = logging.getLogger("admin.v1.audio_speech")

router = APIRouter(tags=["audio"])


def _persist_audio_usage(**kwargs: Any) -> None:
    """Write ONLY an AudioUsageSample for a speech call (no ResponseRecord).

    Audio is telemetry, not a conversation turn — see ARCHITECTURE.md §7. A DB
    failure must never break the response but must not vanish silently either
    (same discipline as the embeddings persist helper).
    """
    try:
        with Session(engine) as session:
            session.add(AudioUsageSample(**kwargs))
            session.commit()
    except Exception:
        logger.exception(
            "failed to persist audio usage for instance %s",
            kwargs.get("provider_instance_id"),
        )


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": "server_error",
                "code": "upstream_failed",
                "message": message,
            }
        },
    )


@router.post("/v1/audio/speech")
async def create_speech(request: Request) -> Any:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400, detail="request body must be a JSON object"
        )

    alias = body.get("model")
    if not alias:
        raise HTTPException(status_code=400, detail="'model' is required")

    input_ = body.get("input")
    if not isinstance(input_, str) or not input_:
        raise HTTPException(
            status_code=400, detail="'input' must be a non-empty string"
        )

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
        definition_id = definition.id

    request_id = f"tts-{uuid.uuid4().hex}"
    scheduler = get_scheduler(request)
    try:
        admission = await scheduler.acquire(alias, request_id)
    except (NoProviderAvailable, QueueCleared) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QueueTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc

    client = httpx.AsyncClient(timeout=settings.AUDIO_REQUEST_TIMEOUT_SECONDS)
    upstream: httpx.Response | None = None
    handed_off = False
    try:
        upstream = await _open_speech(client, admission, body)
        if upstream.status_code >= 400:
            # Non-2xx: relay status + JSON error body (the slot is freed by the
            # ``finally`` below — nothing streams).
            raw = await upstream.aread()
            media_type = upstream.headers.get("content-type", "application/json")
            response: Response = Response(
                content=raw, status_code=upstream.status_code, media_type=media_type
            )
        else:
            # 2xx: stream through. The relay generator takes ownership of the
            # upstream response, the client, and the scheduler slot — it closes
            # and releases them (shielded) when the stream ends or is abandoned.
            headers: dict[str, str] = {}
            sample_rate = upstream.headers.get("x-sample-rate")
            if sample_rate is not None:
                headers["X-Sample-Rate"] = sample_rate
            media_type = upstream.headers.get(
                "content-type", "application/octet-stream"
            )
            response = StreamingResponse(
                _relay_speech(
                    scheduler=scheduler,
                    alias=alias,
                    request_id=request_id,
                    upstream=upstream,
                    client=client,
                    definition_id=definition_id,
                    instance_id=uuid.UUID(admission.instance_id),
                    chars=len(input_),
                ),
                media_type=media_type,
                headers=headers,
            )
            handed_off = True
        return response
    except httpx.HTTPError as exc:
        logger.warning("speech upstream connect failed for %s: %s", request_id, exc)
        return _error_response(502, f"speech upstream unreachable: {exc}")
    finally:
        # Close the leak window between ``acquire`` and the stream hand-off: any
        # exit that did NOT transfer ownership to the relay generator (connect
        # error, a raise while building the response / parsing the admission
        # UUID, or a client disconnect mid-``aread``) frees the client + slot
        # here. ``handed_off`` guards the 2xx path so the generator's own
        # shielded cleanup is the single owner. Release is idempotent.
        if not handed_off:
            if upstream is not None:
                with contextlib.suppress(Exception):
                    await upstream.aclose()
            with contextlib.suppress(Exception):
                await client.aclose()
            with contextlib.suppress(Exception):
                await asyncio.shield(scheduler.release(alias, request_id))


async def _open_speech(
    client: httpx.AsyncClient, admission: Admission, body: dict[str, Any]
) -> httpx.Response:
    """Open the upstream speech stream (status + headers available, body not
    yet read). Raises ``httpx.HTTPError`` on connect/timeout failures."""
    req = client.build_request(
        "POST", f"{admission.base_url}/v1/audio/speech", json=body
    )
    return await client.send(req, stream=True)


async def _relay_speech(
    *,
    scheduler: InferenceScheduler,
    alias: str,
    request_id: str,
    upstream: httpx.Response,
    client: httpx.AsyncClient,
    definition_id: uuid.UUID | None,
    instance_id: uuid.UUID,
    chars: int,
) -> AsyncIterator[bytes]:
    """Relay upstream audio bytes incrementally, sniffing a WAV header for the
    usage sample, and release the slot (shielded) on every exit path."""
    sniff = bytearray()
    limit = audio_svc.sniff_limit()
    completed = False
    try:
        async for chunk in upstream.aiter_raw():
            if len(sniff) < limit:
                sniff.extend(chunk[: limit - len(sniff)])
            yield chunk
        # Reached only on a clean upstream EOF: a client abort throws
        # GeneratorExit into the ``yield`` and a mid-stream upstream error
        # raises out of the ``async for`` — both skip this line, so the usage
        # sample is persisted only for a fully-delivered synthesis.
        completed = True
    finally:

        async def _cleanup() -> None:
            with contextlib.suppress(Exception):
                await upstream.aclose()
            with contextlib.suppress(Exception):
                await client.aclose()
            with contextlib.suppress(Exception):
                await scheduler.release(alias, request_id)

        with contextlib.suppress(Exception):
            await asyncio.shield(_cleanup())
        if completed:
            _persist_audio_usage(
                **audio_svc.audio_usage_kwargs(
                    definition_id=definition_id,
                    instance_id=instance_id,
                    modality="tts",
                    chars=chars,
                    audio_seconds=audio_svc.wav_duration_seconds(bytes(sniff)),
                )
            )

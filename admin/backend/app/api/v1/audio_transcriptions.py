"""Public OpenAI ASR endpoint: ``POST /v1/audio/transcriptions`` (Phase 24).

Mirrors the Phase 18 ``/v1/embeddings`` admission + release discipline but
**bypasses litellm**: the request is a multipart upload and the response may be
``json`` / ``text`` / ``verbose_json`` / ``srt`` / ``vtt``, so the admin relays
the multipart straight to the agent's ``/v1/audio/transcriptions`` with
``httpx`` and passes the upstream response through untouched (ARCHITECTURE.md
§7 Audio specifics).

Differences from embeddings:

- The alias must resolve to a ``modality == "asr"`` definition; any other kind
  is a clean 404.
- The request is ``multipart/form-data`` (``file`` upload and/or ``file_path``,
  plus ``model`` and optional ``language`` / ``response_format`` / ``prompt`` /
  ``diarize`` ...). The uploaded bytes are size-capped
  (``settings.AUDIO_MAX_UPLOAD_BYTES``) before relaying.
- Non-streaming: the full upstream response is read, relayed verbatim (status +
  body + content-type), and the slot released in a cancellation-safe ``finally``
  exactly like embeddings.
- Persistence is an ``AudioUsageSample`` only (``chars`` = transcript length;
  ``audio_seconds`` from a ``verbose_json`` ``duration`` field when present,
  else null) — no ``ResponseRecord``.

Auth model: trusted LAN — this public route is unauthenticated.
"""

import asyncio
import contextlib
import json
import logging
import os
import uuid
from typing import Any, BinaryIO

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from sqlmodel import Session
from starlette.datastructures import UploadFile

from app.api.v1.responses import get_scheduler
from app.core.config import settings
from app.core.db import engine
from app.models import AudioUsageSample
from app.services import audio as audio_svc
from app.services.scheduler import (
    NoProviderAvailable,
    QueueCleared,
    QueueTimeout,
)
from app.services.served_models import resolve_served

logger = logging.getLogger("admin.v1.audio_transcriptions")

router = APIRouter(tags=["audio"])


def _persist_audio_usage(**kwargs: Any) -> None:
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


def _transcript_metrics(status: int, body: bytes) -> tuple[int, float | None]:
    """(chars, audio_seconds) from an upstream transcription body.

    JSON / verbose_json: ``chars`` = length of the ``text`` field,
    ``audio_seconds`` = the ``duration`` field when present (verbose_json).
    Any other content (text / srt / vtt) or a non-JSON body: ``chars`` = the
    decoded body length, ``audio_seconds`` = null.
    """
    if status >= 400:
        return 0, None
    try:
        parsed = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError, UnicodeDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        text = parsed.get("text")
        chars = len(text) if isinstance(text, str) else 0
        duration = parsed.get("duration")
        audio_seconds = float(duration) if isinstance(duration, (int, float)) else None
        return chars, audio_seconds
    return len(body), None


@router.post("/v1/audio/transcriptions")
async def create_transcription(request: Request) -> Any:
    form = await request.form()

    fields: dict[str, str] = {}
    uploads: list[tuple[str, UploadFile]] = []
    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            uploads.append((key, value))
        else:
            fields[key] = str(value)

    alias = fields.get("model")
    if not alias:
        raise HTTPException(status_code=400, detail="'model' is required")
    if not uploads and not fields.get("file_path"):
        raise HTTPException(
            status_code=400, detail="a 'file' upload or 'file_path' is required"
        )

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
        if spec.modality != "asr":
            raise HTTPException(
                status_code=404,
                detail=f"model alias '{alias}' is not a transcription (asr) model",
            )
        definition_id = definition.id

    # Size-cap the uploads BEFORE relaying, then stream each spooled file
    # handle straight to httpx (no full-body buffering in admin memory).
    # Starlette spools an UploadFile to an in-memory buffer up to its spool
    # threshold and then to a temp file on disk; either way ``upload.file`` is a
    # seekable binary handle whose ``fstat`` size is the authoritative byte
    # count, so the cap is enforced from the size (not by reading the body).
    cap = settings.AUDIO_MAX_UPLOAD_BYTES
    files: list[tuple[str, tuple[str, BinaryIO, str]]] = []
    total = 0
    for key, upload in uploads:
        try:
            size = os.fstat(upload.file.fileno()).st_size
        except AttributeError, OSError:
            # Non-file-backed handle (e.g. an in-memory BytesIO in a test):
            # fall back to seeking to measure without consuming.
            pos = upload.file.tell()
            size = upload.file.seek(0, os.SEEK_END)
            upload.file.seek(pos)
        total += size
        if total > cap:
            raise HTTPException(status_code=413, detail="request body too large")
        upload.file.seek(0)
        files.append(
            (
                key,
                (
                    upload.filename or "",
                    upload.file,
                    upload.content_type or "application/octet-stream",
                ),
            )
        )

    request_id = f"asr-{uuid.uuid4().hex}"
    scheduler = get_scheduler(request)
    try:
        admission = await scheduler.acquire(alias, request_id)
    except (NoProviderAvailable, QueueCleared) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QueueTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc

    try:
        try:
            async with httpx.AsyncClient(
                timeout=settings.AUDIO_REQUEST_TIMEOUT_SECONDS
            ) as client:
                upstream = await client.post(
                    f"{admission.base_url}/v1/audio/transcriptions",
                    data=fields,
                    files=files or None,
                )
        except asyncio.CancelledError:
            raise
        except httpx.HTTPError as exc:
            logger.warning("transcription upstream failed for %s: %s", request_id, exc)
            return _error_response(502, f"transcription upstream unreachable: {exc}")

        chars, audio_seconds = _transcript_metrics(
            upstream.status_code, upstream.content
        )
        if upstream.status_code < 400:
            _persist_audio_usage(
                **audio_svc.audio_usage_kwargs(
                    definition_id=definition_id,
                    instance_id=uuid.UUID(admission.instance_id),
                    modality="asr",
                    chars=chars,
                    audio_seconds=audio_seconds,
                )
            )
        # Relay the upstream response untouched (json / text / verbose_json /
        # srt / vtt). content-length/content-encoding are recomputed by
        # starlette, so they are not forwarded.
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
        )
    finally:
        # Guaranteed slot release on EVERY exit path — including
        # asyncio.CancelledError (client disconnect during the await), which
        # `except Exception` does not catch. Release is idempotent.
        with contextlib.suppress(Exception):
            await asyncio.shield(scheduler.release(alias, request_id))

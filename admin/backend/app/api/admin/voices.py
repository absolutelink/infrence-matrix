"""Admin saved-voice enrollment (Phase 24).

``/admin/api/voices`` — the **UI catalog** of cloned voices for ``tts``
definitions (``TTSVoice`` rows). The clips themselves live on the agent's disk;
enrollment rides the agent's HTTP ``/v1/audio/voices`` data plane (no new WS
frame kinds — ARCHITECTURE.md §4 / §7):

- ``GET  /admin/api/voices?definition_id=<id>`` — list the definition's rows
  (readable while the backend is stopped).
- ``POST /admin/api/voices`` — multipart (``definition_id``, ``name``,
  ``ref_text?``, ``language?``, ``file`` wav clip). Proxies
  ``PUT {agent}/v1/audio/voices/{name}`` to a running backend of the
  definition and creates the row only after the agent acks 2xx.
- ``DELETE /admin/api/voices/{voice_id}`` — proxies
  ``DELETE {agent}/v1/audio/voices/{name}`` and drops the row.

**Delete semantics (documented choice):** the row is dropped when the agent
confirms deletion (2xx) **or** reports the voice already gone (404, idempotent);
on any other outcome (503 backend not-ready, transport error) the row is kept
and the error surfaced so the operator can retry — a transient agent failure
never silently loses the catalog entry while the clip may still exist on disk.

The agent address is resolved the same way the scheduler builds an admission
URL (definition → placed instances → connected agent → machine address),
preferring a loaded backend so the write reaches a serving process.

Auth model: trusted LAN — unauthenticated, like the rest of ``/admin/api``.
"""

import logging
import re
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.api.admin.serializers import iso_utc
from app.core.config import settings
from app.core.db import get_session
from app.models import ProviderDefinition, TTSVoice
from app.services import audio as audio_svc

logger = logging.getLogger("admin.voices")

router = APIRouter(prefix="/admin/api/voices", tags=["admin"])

# Mirrors the agent's voice-name guard (provider_lib.app_factory): a single
# filesystem-safe path segment, bounded, no traversal. The name is interpolated
# into the outbound URL path, so this is also the admin-side SSRF/traversal gate.
_VOICE_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")


def _validate_voice_name(name: str) -> str:
    if name in (".", "..") or not _VOICE_NAME_RE.fullmatch(name):
        raise HTTPException(status_code=422, detail="invalid voice name")
    return name


def _get_tts_definition(session: Session, definition_id: str) -> ProviderDefinition:
    try:
        definition = session.get(ProviderDefinition, uuid.UUID(definition_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="definition not found") from None
    if definition is None:
        raise HTTPException(status_code=404, detail="definition not found")
    if definition.modality != "tts":
        raise HTTPException(
            status_code=422,
            detail=f"definition '{definition.alias}' is not a tts definition",
        )
    return definition


def _voice_dict(voice: TTSVoice) -> dict[str, Any]:
    return {
        "id": str(voice.id),
        "definition_id": str(voice.definition_id),
        "name": voice.name,
        "ref_text": voice.ref_text,
        "language": voice.language,
        "created_at": iso_utc(voice.created_at),
    }


def _resolve_agent_base_url(session: Session, definition: ProviderDefinition) -> str:
    base_url = audio_svc.agent_base_url_for_definition(session, definition)
    if base_url is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"no connected provider agent hosts definition '{definition.alias}'"
            ),
        )
    return base_url


def _raise_agent_failure(upstream: httpx.Response) -> None:
    """Map an agent voice-surface failure to a distinct admin HTTP status.

    A 503 (backend not-ready) passes through unchanged; any other 5xx collapses
    to 502 (bad gateway); 4xx are relayed verbatim (the agent's own validation
    rejection). The caller has already decided not to mutate the DB row.
    """
    status = upstream.status_code
    if status == 503:
        mapped = 503
    elif status >= 500:
        mapped = 502
    else:
        mapped = status
    raise HTTPException(status_code=mapped, detail=upstream.text[:500])


@router.get("")
def list_voices(
    session: Session = Depends(get_session),
    definition_id: str = Query(..., description="tts definition id"),
) -> list[dict[str, Any]]:
    definition = _get_tts_definition(session, definition_id)
    rows = session.exec(
        select(TTSVoice)
        .where(TTSVoice.definition_id == definition.id)
        .order_by(TTSVoice.name)
    ).all()
    return [_voice_dict(v) for v in rows]


@router.post("")
async def enroll_voice(
    session: Session = Depends(get_session),
    definition_id: str = Form(...),
    name: str = Form(...),
    ref_text: str | None = Form(default=None),
    language: str | None = Form(default=None),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    definition = _get_tts_definition(session, definition_id)
    name = _validate_voice_name(name)

    wav = await file.read(settings.AUDIO_MAX_VOICE_BYTES + 1)
    if len(wav) > settings.AUDIO_MAX_VOICE_BYTES:
        raise HTTPException(status_code=413, detail="voice clip too large")

    base_url = _resolve_agent_base_url(session, definition)
    params: dict[str, str] = {"model": definition.alias}
    if ref_text is not None:
        params["ref_text"] = ref_text
    if language is not None:
        params["language"] = language

    try:
        async with httpx.AsyncClient(
            timeout=settings.AUDIO_REQUEST_TIMEOUT_SECONDS
        ) as client:
            upstream = await client.put(
                f"{base_url}/v1/audio/voices/{name}",
                params=params,
                content=wav,
            )
    except httpx.HTTPError as exc:
        logger.warning("voice enrollment proxy failed for %s: %s", name, exc)
        raise HTTPException(
            status_code=502, detail=f"voice enrollment upstream unreachable: {exc}"
        ) from exc

    if upstream.status_code >= 400:
        # Relay the agent's refusal with a distinct status; no row is created.
        _raise_agent_failure(upstream)

    try:
        existing = session.exec(
            select(TTSVoice).where(
                TTSVoice.definition_id == definition.id, TTSVoice.name == name
            )
        ).first()
        if existing is not None:
            existing.ref_text = ref_text
            existing.language = language
            voice = existing
        else:
            voice = TTSVoice(
                definition_id=definition.id,
                name=name,
                ref_text=ref_text,
                language=language,
            )
        session.add(voice)
        session.commit()
    except IntegrityError:
        # NIT: a concurrent enroll won the (definition_id, name) unique race
        # between the read and the INSERT — fall back to update semantics.
        session.rollback()
        voice = session.exec(
            select(TTSVoice).where(
                TTSVoice.definition_id == definition.id, TTSVoice.name == name
            )
        ).one()
        voice.ref_text = ref_text
        voice.language = language
        session.add(voice)
        session.commit()
    session.refresh(voice)
    logger.info("enrolled voice '%s' for definition %s", name, definition.alias)
    return {"ok": True, "voice": _voice_dict(voice)}


@router.delete("/{voice_id}")
async def delete_voice(
    voice_id: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    try:
        voice = session.get(TTSVoice, uuid.UUID(voice_id))
    except ValueError:
        voice = None
    if voice is None:
        raise HTTPException(status_code=404, detail="voice not found")

    definition = session.get(ProviderDefinition, voice.definition_id)
    if definition is None:  # pragma: no cover - FK cascade keeps this consistent
        session.delete(voice)
        session.commit()
        return {"ok": True, "id": voice_id, "orphan_row_dropped": True}

    base_url = _resolve_agent_base_url(session, definition)
    try:
        async with httpx.AsyncClient(
            timeout=settings.AUDIO_REQUEST_TIMEOUT_SECONDS
        ) as client:
            upstream = await client.delete(
                f"{base_url}/v1/audio/voices/{voice.name}",
                params={"model": definition.alias},
            )
    except httpx.HTTPError as exc:
        logger.warning("voice delete proxy failed for %s: %s", voice.name, exc)
        raise HTTPException(
            status_code=502, detail=f"voice delete upstream unreachable: {exc}"
        ) from exc

    # Drop the row on agent 2xx (confirmed) or 404 (already gone — idempotent).
    # Any other status keeps the row and relays a distinct error for retry.
    if upstream.status_code >= 400 and upstream.status_code != 404:
        _raise_agent_failure(upstream)

    session.delete(voice)
    session.commit()
    logger.info("deleted voice '%s' for definition %s", voice.name, definition.alias)
    return {"ok": True, "id": voice_id}

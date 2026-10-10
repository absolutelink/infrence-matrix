"""Live-ASR WebSocket relay: ``WS /v1/audio/transcriptions/stream`` (Phase 24 S4).

The admin **terminates the client WebSocket** and relays frames bidirectionally
to the agent's live-ASR WS surface
(``provider/lib/provider_lib/app_factory.py`` → ``transcribe_stream``), mirroring
the Phase 21 Responses-over-WebSocket transport discipline
(:mod:`app.api.v1.responses_ws`): the admin owns the client socket, admits a
scheduler slot for the whole connection lifetime, and releases it — shielded —
on every teardown path. Providers are unchanged; **litellm is bypassed**
(ARCHITECTURE.md §7 Audio specifics).

Unlike the Phase 21 transport (a multi-turn socket that stays open and reports
errors as JSON ``error`` envelopes), this is a single-session byte relay: there
is no "next turn" to keep the socket alive for, so every failure terminates the
connection with a WebSocket **close code + reason** (the idiomatic relay error
channel). The client's frames are the talkies protocol's business — the relay is
a straight text/binary passthrough and parses nothing (so no usage sample is
persisted here; the HTTP ``/v1/audio/transcriptions`` route owns ASR telemetry).

Close-code mapping (documented contract):

==========================================  ==========  =========================
condition                                   close code  HTTP analog
==========================================  ==========  =========================
missing / empty ``model`` query param       1008        400
unknown alias                               1008        404
disabled definition                         1008        404
wrong modality (not ``asr``)                1008        404
no provider (NoProviderAvailable/Cleared)   1013        503
queue timeout (QueueTimeout)                1013        504
upstream connect / handshake failure        1011        502
oversized inbound frame                     1009        413
upstream close (agent 1008/1013/1011/...)   relayed     —
max connection duration reached             1000        —
idle timeout reached                        1000        —
client disconnect                           (gone)      —
==========================================  ==========  =========================

Auth model: trusted LAN — unauthenticated, like the HTTP ``/v1`` routes and the
agent data plane (the client's bearer/identity is NOT forwarded upstream).
"""

import asyncio
import contextlib
import logging
import time
import uuid
from typing import Any
from urllib.parse import quote

import websockets
from fastapi import APIRouter, WebSocket
from sqlmodel import Session, select

from app.core.config import settings
from app.core.db import engine
from app.models import ProviderDefinition
from app.services.scheduler import (
    Admission,
    InferenceScheduler,
    NoProviderAvailable,
    QueueCleared,
    QueueTimeout,
)

logger = logging.getLogger("admin.v1.audio_transcriptions_ws")

router = APIRouter(tags=["audio"])

# Reserved WS close codes used by this relay (RFC 6455 §7.4).
_CLOSE_POLICY = 1008  # pre-flight rejection (missing/unknown/disabled/non-asr)
_CLOSE_TRY_AGAIN = 1013  # scheduler admission failure (503/504 analog)
_CLOSE_INTERNAL = 1011  # upstream unreachable (502 analog)
_CLOSE_TOO_BIG = 1009  # oversized inbound frame (413 analog)
_CLOSE_NORMAL = 1000  # graceful end (max duration / idle / normal upstream close)


def _now() -> float:
    """Monotonic clock indirection so tests can drive the duration/idle timers."""
    return time.monotonic()


def _ws_url(base_url: str, alias: str) -> str:
    """Turn an admission ``http(s)://host:port`` base into the agent's live-ASR
    WS URL with the alias in the ``model`` query param (the agent routes by it)."""
    if base_url.startswith("https://"):
        ws_base = "wss://" + base_url[len("https://") :]
    elif base_url.startswith("http://"):
        ws_base = "ws://" + base_url[len("http://") :]
    else:
        ws_base = base_url
    return f"{ws_base}/v1/audio/transcriptions/stream?model={quote(alias)}"


def _resolve_asr_alias(alias: str) -> tuple[uuid.UUID, str] | str:
    """Resolve an enabled ``asr`` alias to ``(definition_id, provider_type)`` or
    return a short human-readable rejection reason (mapped to close 1008)."""
    with Session(engine) as session:
        definition = session.exec(
            select(ProviderDefinition).where(ProviderDefinition.alias == alias)
        ).first()
        if definition is None:
            return f"unknown model alias '{alias}'"
        if not definition.enabled:
            return f"model alias '{alias}' is disabled"
        if definition.modality != "asr":
            return f"model alias '{alias}' is not a transcription (asr) model"
        return definition.id, definition.provider_type


async def _close(websocket: WebSocket, code: int, reason: str) -> None:
    """Close the client socket, swallowing an already-gone socket."""
    with contextlib.suppress(Exception):
        await websocket.close(code=code, reason=reason)


async def _pump_client_to_upstream(
    websocket: WebSocket, upstream: Any, state: dict[str, Any], max_frame: int
) -> None:
    """Forward client frames (text + binary) upstream verbatim.

    Ends on a client disconnect (returns) or records ``frame_too_big`` and
    returns when a frame exceeds ``max_frame`` bytes (the untrusted direction)."""
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return
        text = message.get("text")
        if text is not None:
            if len(text) > max_frame:
                state["frame_too_big"] = True
                return
            await upstream.send(text)
        else:
            raw = message.get("bytes") or b""
            if len(raw) > max_frame:
                state["frame_too_big"] = True
                return
            await upstream.send(raw)
        state["last_activity"] = _now()


async def _pump_upstream_to_client(
    websocket: WebSocket, upstream: Any, state: dict[str, Any]
) -> None:
    """Forward agent frames (text + binary) to the client verbatim.

    Ends when the upstream closes: a clean close returns; a non-normal close
    raises ``ConnectionClosed`` which is swallowed here so the caller can read
    ``upstream.close_code`` and relay it to the client."""
    try:
        async for message in upstream:
            if isinstance(message, str):
                await websocket.send_text(message)
            else:
                await websocket.send_bytes(message)
            state["last_activity"] = _now()
    except websockets.exceptions.ConnectionClosed:
        return


async def _watchdog(
    state: dict[str, Any], *, start: float, max_duration: float, idle_timeout: float
) -> None:
    """Return (recording the reason in ``state``) when the connection cap or the
    idle timeout elapses; the caller races this against the relay pumps."""
    while True:
        now = _now()
        if max_duration > 0 and now - start >= max_duration:
            state["watchdog"] = (_CLOSE_NORMAL, "connection limit reached")
            return
        if idle_timeout > 0 and now - state["last_activity"] >= idle_timeout:
            state["watchdog"] = (_CLOSE_NORMAL, "idle timeout")
            return
        await asyncio.sleep(0.05)


async def _relay(
    websocket: WebSocket,
    upstream: Any,
    *,
    max_frame: int,
    max_duration: float,
    idle_timeout: float,
) -> None:
    """Bidirectional relay: whichever side ends first tears down the others, then
    the client is closed with the appropriate code (relay / limit / oversize)."""
    start = _now()
    state: dict[str, Any] = {"last_activity": start}
    c2u = asyncio.create_task(
        _pump_client_to_upstream(websocket, upstream, state, max_frame)
    )
    u2c = asyncio.create_task(_pump_upstream_to_client(websocket, upstream, state))
    watchdog = asyncio.create_task(
        _watchdog(
            state, start=start, max_duration=max_duration, idle_timeout=idle_timeout
        )
    )
    pumps = {c2u, u2c, watchdog}
    try:
        await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in pumps:
            task.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(*pumps, return_exceptions=True)

    # Decide the client close code by precedence.
    if state.get("frame_too_big"):
        await _close(websocket, _CLOSE_TOO_BIG, "frame too large")
    elif "watchdog" in state:
        code, reason = state["watchdog"]
        await _close(websocket, code, reason)
    else:
        upstream_code = getattr(upstream, "close_code", None)
        if upstream_code not in (None, _CLOSE_NORMAL):
            reason = getattr(upstream, "close_reason", None) or "upstream closed"
            await _close(websocket, upstream_code, reason)
        else:
            # Normal end (clean upstream close) or a client disconnect (the
            # close is a no-op on a gone socket).
            await _close(websocket, _CLOSE_NORMAL, "closed")


@router.websocket("/v1/audio/transcriptions/stream")
async def transcriptions_websocket(websocket: WebSocket) -> None:
    """Live-ASR WS relay to a scheduler-admitted ``asr`` agent backend."""
    await websocket.accept()
    scheduler: InferenceScheduler | None = getattr(
        websocket.app.state, "scheduler", None
    )
    if scheduler is None:
        await _close(websocket, _CLOSE_INTERNAL, "scheduler not initialized")
        return

    alias = websocket.query_params.get("model")
    if not alias:
        await _close(websocket, _CLOSE_POLICY, "'model' query param is required")
        return

    resolved = _resolve_asr_alias(alias)
    if isinstance(resolved, str):
        await _close(websocket, _CLOSE_POLICY, resolved)
        return

    request_id = f"asrws-{uuid.uuid4().hex}"
    try:
        admission: Admission = await scheduler.acquire(alias, request_id)
    except (NoProviderAvailable, QueueCleared) as exc:
        await _close(websocket, _CLOSE_TRY_AGAIN, str(exc))
        return
    except QueueTimeout as exc:
        await _close(websocket, _CLOSE_TRY_AGAIN, str(exc))
        return

    # From here the slot is held: the shielded ``finally`` below releases it on
    # EVERY exit path (client disconnect, upstream close incl. the agent's
    # 1013/1008/1011, relay exception, cancellation). Release is idempotent.
    try:
        try:
            upstream = await websockets.connect(_ws_url(admission.base_url, alias))
        except Exception as exc:  # noqa: BLE001 - agent unreachable / rejected
            logger.warning(
                "live-ASR upstream connect failed for %s: %s", request_id, exc
            )
            await _close(
                websocket, _CLOSE_INTERNAL, "transcription upstream unreachable"
            )
            return
        try:
            await _relay(
                websocket,
                upstream,
                max_frame=settings.AUDIO_WS_MAX_FRAME_BYTES,
                max_duration=settings.AUDIO_WS_MAX_DURATION_SECONDS,
                idle_timeout=settings.AUDIO_WS_IDLE_TIMEOUT_SECONDS,
            )
        finally:
            with contextlib.suppress(Exception):
                await upstream.close()
    finally:
        # Shielded so a task cancellation cannot skip the release.
        with contextlib.suppress(Exception):
            await asyncio.shield(scheduler.release(alias, request_id))

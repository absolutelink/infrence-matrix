"""FastAPI app factory for a provider instance.

Every provider type gets a base app that serves:
  - GET  /health         liveness + provider type + version + backend/slot state
  - GET  /v1/models      model list from the backend driver
  - POST /v1/responses   OpenResponses SSE stream (slot-admitted)
  - POST /v1/chat/completions  OpenAI chat.completions (SSE or JSON,
    slot-admitted; non-stream aggregates the driver's chunk stream)
  - POST /v1/embeddings  OpenAI embeddings (single JSON response,
    slot-admitted; non-streaming only)
  - POST /v1/audio/speech  OpenAI TTS (binary audio out, slot-admitted
    streaming; Phase 24)
  - POST /v1/audio/transcriptions  OpenAI ASR multipart relay
    (slot-admitted; upstream status/headers/body pass through; Phase 24)
  - WS   /v1/audio/transcriptions/stream  live ASR WS bridge (slot held for
    the connection lifetime; Phase 24)
  - GET/PUT/DELETE /v1/audio/voices[/{name}]  voice catalog + enrollment
    (no slot; Phase 24)

Admin's litellm client targets ``http://<machine>:<PROVIDER_PORT>/v1``;
the provider normalizes its backend into a clean OpenAI-compatible
surface here. Per-type behavior is injected via `BackendOverrides` so the
admin carries zero provider-specific logic.

Slot discipline: the lifecycle slot is acquired *before* the streaming
response starts (so 429/503 are real HTTP statuses, not mid-stream
errors) and released when the UPSTREAM driver stream closes, never when
the downstream client disconnects. See `provider_lib.backend`.
"""

import contextlib
import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.datastructures import UploadFile

from provider_lib.backend import (
    BackendBusy,
    BackendLifecycle,
    BackendNotReady,
)
from provider_lib.config import ProviderSettings
from provider_lib.registry import BackendRegistry
from provider_lib.wire import BackendStatusValue

logger = logging.getLogger("provider.app_factory")

# Phase 24 (audio): response headers that must NEVER be relayed from an
# upstream transcription back to the client. These are hop-by-hop (RFC 9110
# §7.6.1) plus ``content-encoding`` / ``content-length``: httpx has already
# decoded the body and starlette recomputes the length, so forwarding the
# upstream's would corrupt the relayed response.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-encoding",
        "content-length",
    }
)

# Phase 24 (audio): upload ceilings (bytes) enforced by the route before any
# body is buffered into memory. These are the agent-side hard caps; an
# individual driver may lower them further inside its own hook, never raise
# them here.
MAX_TRANSCRIPTION_BYTES = 100 * 1024 * 1024  # 100 MiB
MAX_VOICE_BYTES = 32 * 1024 * 1024  # 32 MiB

# Phase 24 (audio): saved-voice names must be filesystem-safe and bounded so a
# name can never escape the driver's custom-voices directory. ``.`` / ``..``
# are additionally rejected as reserved traversal segments (the character class
# alone would otherwise admit them).
_VOICE_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")


def _validate_voice_name(name: str) -> None:
    """Reject unsafe / traversal voice names with HTTP 400 before dispatch."""
    if name in (".", "..") or not _VOICE_NAME_RE.fullmatch(name):
        raise HTTPException(status_code=400, detail="invalid voice name")


async def _read_body_capped(request: Request, limit: int) -> bytes:
    """Read a raw request body, refusing (413) once it exceeds ``limit``.

    Streams rather than calling ``request.body()`` so an oversized upload is
    rejected without ever fully buffering into memory.
    """
    buffer = bytearray()
    async for chunk in request.stream():
        buffer.extend(chunk)
        if len(buffer) > limit:
            raise HTTPException(status_code=413, detail="request body too large")
    return bytes(buffer)


class BackendOverrides:
    """Hook bundle a provider package supplies to customize the generic app.

    Subclass or instantiate with callables; unset hooks fall back to lib
    defaults.

    - ``provider_type`` / ``version``: identity reported on /health and
      registration.
    - ``calculate_usage``: per-type usage computation (kept from
      Phase 3; unused by the generic /v1 layer).
    - ``lifecycle``: a single `BackendLifecycle` (the pre-overhaul /
      single-backend path). When present (and ``registry`` is ``None``) the /v1
      surface mounts on it; when both are ``None`` the /v1 endpoints answer 503.
    - ``registry``: a :class:`~provider_lib.registry.BackendRegistry` of hosted
      backends (port model overhaul). When present the agent serves ONE ``/v1``
      surface and routes each request to the backend whose ``alias`` equals the
      request's ``model`` field (``registry.resolve_by_model``); an unknown
      model is a 404. Takes precedence over ``lifecycle``.
    """

    def __init__(
        self,
        *,
        provider_type: str,
        version: str,
        calculate_usage: Callable[[Any], dict[str, int]] | None = None,
        lifecycle: BackendLifecycle | None = None,
        registry: BackendRegistry | None = None,
    ) -> None:
        self.provider_type = provider_type
        self.version = version
        self.calculate_usage = calculate_usage
        self.lifecycle = lifecycle
        self.registry = registry


def _sse(event: dict[str, Any]) -> str:
    """Format one OpenResponses/OpenAI event dict as an SSE frame."""
    event_type = event.get("type", "message")
    return f"event: {event_type}\ndata: {json.dumps(event)}\n\n"


def create_provider_app(
    settings: ProviderSettings, overrides: BackendOverrides
) -> FastAPI:
    app = FastAPI(
        title=f"Inference Matrix provider ({overrides.provider_type})",
        version=overrides.version,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        body: dict[str, Any] = {
            "status": "ok",
            "provider_type": overrides.provider_type,
            "version": overrides.version,
            "machine_uid": settings.MACHINE_UID,
        }
        if overrides.registry is not None:
            # Agent-level health for the single-env-port multiplexing surface:
            # a per-backend summary (no single backend_status applies to all).
            body["backends"] = [
                {
                    "instance_id": h.instance_id,
                    "alias": h.alias,
                    "backend_status": h.lifecycle.backend_status,
                    "in_flight": h.lifecycle.in_flight,
                    "capacity": h.lifecycle.capacity,
                }
                for h in overrides.registry.handles()
            ]
            return body
        if overrides.lifecycle is not None:
            body["backend_status"] = overrides.lifecycle.backend_status
            body["in_flight"] = overrides.lifecycle.in_flight
            body["capacity"] = overrides.lifecycle.capacity
        return body

    def _require_lifecycle() -> BackendLifecycle:
        if overrides.lifecycle is None:
            raise HTTPException(status_code=503, detail="backend lifecycle not wired")
        return overrides.lifecycle

    def _resolve_lifecycle(body: dict[str, Any]) -> BackendLifecycle:
        """Pick the lifecycle that serves this request.

        Registry (single-env-port) path: route by the request's ``model`` (=
        definition alias); an unknown/absent model is a 404. A non-object body
        is a 400 (a JSON array/string would otherwise raise inside ``body.get``).
        Single-lifecycle path: the one wired lifecycle (unchanged behavior).
        """
        if overrides.registry is not None:
            if not isinstance(body, dict):
                raise HTTPException(
                    status_code=400, detail="request body must be a JSON object"
                )
            model = body.get("model")
            handle = overrides.registry.resolve_by_model(model)
            if handle is None:
                raise HTTPException(status_code=404, detail=f"unknown model '{model}'")
            return handle.lifecycle
        return _require_lifecycle()

    async def _stream_or_error(
        stream: AsyncIterator[dict[str, Any]], terminator: bool
    ) -> AsyncIterator[str]:
        try:
            async for event in stream:
                yield _sse(event)
            if terminator:
                yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("stream failure after response started")
            yield _sse(
                {
                    "type": "error",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
        finally:
            # Explicitly close the lifecycle stream on every exit path
            # (normal end, error, or client-disconnect cancellation) so the
            # slot release is driven deterministically by the upstream
            # generator's close rather than by CPython refcount GC. The
            # pump's own `finally` performs the actual release.
            with contextlib.suppress(Exception):
                await stream.aclose()

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        if overrides.registry is not None:
            # Aggregate the model list across every hosted backend that is
            # currently serving (running/in_use), unioned by model id. A
            # per-backend scrape failure is skipped (best-effort) so one bad
            # engine never blanks the whole agent's listing.
            data: list[dict[str, Any]] = []
            seen: set[Any] = set()
            for handle in overrides.registry.handles():
                lifecycle = handle.lifecycle
                if lifecycle.backend_status not in (
                    BackendStatusValue.RUNNING,
                    BackendStatusValue.IN_USE,
                ):
                    continue
                try:
                    models = await lifecycle.driver.list_models()
                except Exception:  # noqa: BLE001 - one backend must not fail all
                    logger.warning(
                        "list_models failed for backend %s",
                        handle.instance_id or "?",
                        exc_info=True,
                    )
                    continue
                for model in models:
                    key = model.get("id") if isinstance(model, dict) else model
                    if key in seen:
                        continue
                    seen.add(key)
                    data.append(model)
            return {"object": "list", "data": data}
        lifecycle = _require_lifecycle()
        # Models are only listable while the backend is serving; a stopped
        # backend has no cheap source of truth, so report 503 and let the
        # admin (Phase 6) start it on demand.
        if lifecycle.backend_status not in (
            BackendStatusValue.RUNNING,
            BackendStatusValue.IN_USE,
        ):
            raise HTTPException(
                status_code=503,
                detail=f"backend is {lifecycle.backend_status}; models unavailable",
            )
        try:
            data = await lifecycle.driver.list_models()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"object": "list", "data": data}

    @app.post("/v1/responses")
    async def responses(request: Request) -> Response:
        body = await request.json()
        lifecycle = _resolve_lifecycle(body)
        try:
            # Eager acquire inside: BackendBusy/BackendNotReady surface
            # here, before any bytes are committed to the client.
            stream = await lifecycle.stream_responses(body)
        except BackendBusy as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except BackendNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if body.get("stream"):
            return StreamingResponse(
                _stream_or_error(stream, terminator=False),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        # Non-streaming (spec default): drain the event stream and return
        # the terminal response object as JSON. The slot is still owned by
        # the pump task and released when the upstream closes.
        terminal: dict[str, Any] | None = None
        try:
            async for event in stream:
                if event.get("type") in (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                ):
                    terminal = event.get("response")
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()
        if terminal is None:
            raise HTTPException(
                status_code=502, detail="backend stream ended without a terminal event"
            )
        return JSONResponse(content=terminal)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body = await request.json()
        lifecycle = _resolve_lifecycle(body)
        if not body.get("stream"):
            # Non-stream (spec default): the upstream llama.cpp chat SSE
            # only carries a usage chunk when stream_options.include_usage
            # is set — force it so the aggregated chat.completion carries
            # usage per OpenAI spec (llama.cpp direct non-stream calls
            # report usage, but its SSE aggregation path does not without
            # this).
            options = dict(body.get("stream_options") or {})
            options["include_usage"] = True
            body["stream_options"] = options
        try:
            # Eager acquire inside: BackendBusy/BackendNotReady surface
            # here, before any bytes are committed to the client.
            stream = await lifecycle.stream_chat_completions(body)
        except BackendBusy as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except BackendNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        if body.get("stream"):
            return StreamingResponse(
                _stream_or_error(stream, terminator=True),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        # Non-streaming (spec default): drain the chunk stream and return
        # the aggregated chat.completion object as JSON. The slot is still
        # owned by the pump task and released when the upstream closes.
        completion: dict[str, Any] | None = None
        try:
            async for chunk in stream:
                completion = _aggregate_chat_chunk(completion, chunk)
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()
        if completion is None:
            raise HTTPException(
                status_code=502, detail="backend stream ended without a chunk"
            )
        return JSONResponse(content=completion)

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        body = await request.json()
        lifecycle = _resolve_lifecycle(body)
        try:
            result = await lifecycle.embeddings(body)
        except BackendBusy as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except BackendNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        return JSONResponse(content=result)

    # ------------------------------------------------------------------
    # Phase 24 (audio): speech / transcription / voice catalog.
    # Same resolution + slot discipline as /v1/embeddings: the target
    # lifecycle is chosen by the request's ``model`` (unknown -> 404), an
    # unset driver hook -> 501, busy -> 429, not-ready -> 503. Slots are
    # acquired before any bytes are committed and released on upstream close.
    # ------------------------------------------------------------------
    @app.post("/v1/audio/speech")
    async def audio_speech(request: Request) -> Response:
        body = await request.json()
        lifecycle = _resolve_lifecycle(body)
        try:
            # Eager acquire inside: busy/not-ready surface before the
            # StreamingResponse starts.
            stream = await lifecycle.speech(body)
        except BackendBusy as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except BackendNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        headers = dict(stream.headers)
        media_type = headers.pop("Content-Type", None) or "application/octet-stream"
        return StreamingResponse(stream.chunks, media_type=media_type, headers=headers)

    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(request: Request) -> Response:
        form = await request.form()
        # Collect text fields and file references first; resolve the target
        # backend from the text fields BEFORE buffering any upload body into
        # memory (an unknown model is a 404 without reading the files).
        fields: dict[str, str] = {}
        uploads: list[tuple[str, UploadFile]] = []
        for key, value in form.multi_items():
            if isinstance(value, UploadFile):
                uploads.append((key, value))
            else:
                fields[key] = str(value)
        lifecycle = _resolve_lifecycle(fields)
        files: list[tuple[str, str, bytes, str]] = []
        for key, upload in uploads:
            data = await upload.read(MAX_TRANSCRIPTION_BYTES + 1)
            if len(data) > MAX_TRANSCRIPTION_BYTES:
                raise HTTPException(
                    status_code=413, detail="transcription upload too large"
                )
            files.append(
                (
                    key,
                    upload.filename or "",
                    data,
                    upload.content_type or "application/octet-stream",
                )
            )
        try:
            status, headers, body = await lifecycle.transcribe(fields, files)
        except BackendBusy as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except BackendNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        # Upstream status/body pass through untouched (json / text /
        # verbose_json / srt / vtt); hop-by-hop + content-encoding/length are
        # stripped so the relayed response is not corrupted.
        relay_headers = {
            k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS
        }
        return Response(content=body, status_code=status, headers=relay_headers)

    @app.get("/v1/audio/voices")
    async def audio_voices(request: Request) -> JSONResponse:
        lifecycle = _resolve_lifecycle({"model": request.query_params.get("model")})
        try:
            result = await lifecycle.voices()
        except BackendNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        return JSONResponse(content=result)

    @app.put("/v1/audio/voices/{name:path}")
    async def audio_voice_put(name: str, request: Request) -> JSONResponse:
        _validate_voice_name(name)
        lifecycle = _resolve_lifecycle({"model": request.query_params.get("model")})
        wav = await _read_body_capped(request, MAX_VOICE_BYTES)
        ref_text = request.query_params.get("ref_text")
        language = request.query_params.get("language")
        try:
            result = await lifecycle.voice_put(name, wav, ref_text, language)
        except BackendNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        return JSONResponse(content=result)

    @app.delete("/v1/audio/voices/{name:path}")
    async def audio_voice_delete(name: str, request: Request) -> JSONResponse:
        _validate_voice_name(name)
        lifecycle = _resolve_lifecycle({"model": request.query_params.get("model")})
        try:
            result = await lifecycle.voice_delete(name)
        except BackendNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        return JSONResponse(content=result)

    @app.websocket("/v1/audio/transcriptions/stream")
    async def audio_transcriptions_ws(websocket: WebSocket) -> None:
        try:
            lifecycle = _resolve_lifecycle(
                {"model": websocket.query_params.get("model")}
            )
        except HTTPException:
            # Unknown model / unwired backend: close with a policy code (no
            # HTTP status exists on an accepted WS handshake).
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            # Slot held for the connection lifetime; released when the driver
            # bridge returns (any disconnect) via the lifecycle's finally.
            await lifecycle.transcribe_stream(websocket)
        # Parenthesized deliberately (py314's ruff format would strip it).
        except (BackendBusy, BackendNotReady):  # fmt: skip
            await websocket.close(code=1013)
        except NotImplementedError:
            await websocket.close(code=1011)

    return app


def _aggregate_chat_chunk(
    acc: dict[str, Any] | None, chunk: dict[str, Any]
) -> dict[str, Any]:
    """Fold one chat.completion.chunk into a chat.completion object.

    Merges per-choice ``delta`` content and tool-call fragments (keyed by
    the chunk's tool-call ``index``) into ``choices[].message``; carries
    the last-seen ``finish_reason`` and any ``usage`` object. Mirrors the
    OpenAI chat non-stream shape so admin's litellm client parses it.
    """
    if acc is None:
        acc = {
            "id": chunk.get("id"),
            "object": "chat.completion",
            "created": chunk.get("created"),
            "model": chunk.get("model"),
            "choices": [],
        }
    for key in ("id", "created", "model"):
        if acc.get(key) is None and chunk.get(key) is not None:
            acc[key] = chunk[key]
    choices: list[dict[str, Any]] = acc["choices"]
    for choice in chunk.get("choices") or []:
        index = int(choice.get("index", len(choices)))
        while len(choices) <= index:
            choices.append(
                {
                    "index": len(choices),
                    "message": {"role": "assistant"},
                    "finish_reason": None,
                }
            )
        slot = choices[index]
        message: dict[str, Any] = slot["message"]
        delta = choice.get("delta") or {}
        if delta.get("role"):
            message["role"] = delta["role"]
        if delta.get("content") is not None:
            message["content"] = (message.get("content") or "") + str(delta["content"])
        if delta.get("tool_calls") is not None:
            calls: list[dict[str, Any]] = message.setdefault("tool_calls", [])
            for tool_delta in delta["tool_calls"]:
                t_index = int(tool_delta.get("index", len(calls)))
                while len(calls) <= t_index:
                    calls.append(
                        {
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    )
                call = calls[t_index]
                if tool_delta.get("id"):
                    call["id"] = tool_delta["id"]
                if tool_delta.get("type"):
                    call["type"] = tool_delta["type"]
                fn = tool_delta.get("function") or {}
                if fn.get("name"):
                    call["function"]["name"] = (
                        call["function"].get("name") or ""
                    ) + str(fn["name"])
                if fn.get("arguments"):
                    call["function"]["arguments"] = (
                        call["function"].get("arguments") or ""
                    ) + str(fn["arguments"])
        if choice.get("finish_reason") is not None:
            slot["finish_reason"] = choice["finish_reason"]
    if chunk.get("usage") is not None:
        acc["usage"] = chunk["usage"]
    return acc

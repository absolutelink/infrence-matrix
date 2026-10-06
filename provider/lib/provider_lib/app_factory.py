"""FastAPI app factory for a provider instance.

Every provider type gets a base app that serves:
  - GET  /health         liveness + provider type + version + backend/slot state
  - GET  /v1/models      model list from the backend driver
  - POST /v1/responses   OpenResponses SSE stream (slot-admitted)
  - POST /v1/chat/completions  OpenAI chat.completions (SSE or JSON,
    slot-admitted; non-stream aggregates the driver's chunk stream)

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
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from provider_lib.backend import (
    BackendBusy,
    BackendLifecycle,
    BackendNotReady,
)
from provider_lib.config import ProviderSettings
from provider_lib.wire import BackendStatusValue

logger = logging.getLogger("provider.app_factory")


class BackendOverrides:
    """Hook bundle a provider package supplies to customize the generic app.

    Subclass or instantiate with callables; unset hooks fall back to lib
    defaults.

    - ``provider_type`` / ``version``: identity reported on /health and
      registration.
    - ``calculate_usage``: per-type usage computation (kept from
      Phase 3; unused by the generic /v1 layer).
    - ``lifecycle``: a `BackendLifecycle` wrapping the provider's
      `BackendDriver`. When present the /v1 surface is mounted; when
      ``None`` the /v1 endpoints answer 503 (backend machinery not
      wired). The provider package constructs the driver and the
      lifecycle once and shares the same instance between the admin WS
      command handlers (backend.start/stop) and this app.
    """

    def __init__(
        self,
        *,
        provider_type: str,
        version: str,
        calculate_usage: Callable[[Any], dict[str, int]] | None = None,
        lifecycle: BackendLifecycle | None = None,
    ) -> None:
        self.provider_type = provider_type
        self.version = version
        self.calculate_usage = calculate_usage
        self.lifecycle = lifecycle


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
        if overrides.lifecycle is not None:
            body["backend_status"] = overrides.lifecycle.backend_status
            body["in_flight"] = overrides.lifecycle.in_flight
            body["capacity"] = overrides.lifecycle.capacity
        return body

    def _require_lifecycle() -> BackendLifecycle:
        if overrides.lifecycle is None:
            raise HTTPException(status_code=503, detail="backend lifecycle not wired")
        return overrides.lifecycle

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
        lifecycle = _require_lifecycle()
        body = await request.json()
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
        lifecycle = _require_lifecycle()
        body = await request.json()
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

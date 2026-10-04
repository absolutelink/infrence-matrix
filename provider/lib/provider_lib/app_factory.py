"""FastAPI app factory for a provider instance.

Every provider type gets a base app that serves:
  - GET  /health         liveness + provider type + version + backend/slot state
  - GET  /v1/models      model list from the backend driver
  - POST /v1/responses   OpenResponses SSE stream (slot-admitted)
  - POST /v1/chat/completions  OpenAI chat.completions SSE (slot-admitted)

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
    async def chat_completions(request: Request) -> StreamingResponse:
        lifecycle = _require_lifecycle()
        body = await request.json()
        try:
            stream = await lifecycle.stream_chat_completions(body)
        except BackendBusy as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except BackendNotReady as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        return StreamingResponse(
            _stream_or_error(stream, terminator=True),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app

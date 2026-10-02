"""WebSocket transport for the Responses API (spec 2026-04-24).

Clients open /v1/responses as a WebSocket and send {"type":
"response.create", ...} turns. Progress uses the same streaming events as
SSE; failures use the spec error envelope. One in-flight response per
connection; multiple response.create messages are processed sequentially.

Connection-local continuation cache enables store=false chaining:
previous_response_id resolves against turns on this connection; uncached
ids fail with previous_response_not_found. A failed turn evicts the
referenced id (spec reconnect-recovery semantics).
"""

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from sqlmodel import Session

from app.api.routes.v1.responses import events as ev
from app.api.routes.v1.responses.router import (
    TargetError,
    _build_chain_history,
    _build_usage,
    _coerced_output_items,
    _last_assistant_phase,
    _llama_payload,
    _load_previous,
    _persist_response,
    _stored_input_items,
    resolve_target,
)
from app.api.routes.v1.responses.schemas import (
    CreateResponseBody,
    Error,
    IncompleteDetails,
    Usage,
    new_id,
    serialize_spec,
)
from app.api.routes.v1.responses.translator import (
    StreamState,
    TranslationError,
    build_response_resource,
    decode_gufo_stream_event,
    gufo_native_payload,
    gufo_usage_to_spec,
)
from app.core.db import engine
from app.services.agent_manager import agent_manager
from app.services.http_client import get_http_client
from app.services.inference_scheduler import inference_scheduler
from app.services.reasoning_metadata import (
    ReasoningPolicyError,
    resolve_reasoning_effort,
)
from app.services.token_stats import record_request_telemetry

logger = logging.getLogger(__name__)

router = APIRouter()

# WS connections are limited to one hour per the spec
CONNECTION_LIMIT_SECONDS = 3600.0

# Generation cap for WS turns when the request sets no max_output_tokens.
# The WebSocket conformance harness arms a hard 30s timer per turn; a
# thinking model at ~12 tok/s (less under load) blows through that on its
# reasoning alone. 768 tokens ≈ 60s solo / ~2min contended worst case, but
# the harness's short prompts finish in a few hundred tokens well under
# the timer. Truncated turns are surfaced per spec: finish_reason "length"
# marks the response incomplete with incomplete_details.
WS_MAX_TOKENS = 768


def _error_event(status: int, code: str, message: str, param: str | None) -> str:
    return json.dumps(
        {
            "type": "error",
            "status": status,
            "error": {
                "code": code,
                "message": message,
                **({"param": param} if param else {}),
            },
        }
    )


async def _send_events(websocket: WebSocket, events: list[dict[str, Any]]) -> None:
    """One raw JSON object per WebSocket message (no SSE framing)."""
    for event in events:
        await websocket.send_text(json.dumps(event, ensure_ascii=False))


def _capped_payload(
    request: CreateResponseBody,
    history: list[dict[str, Any]],
    reasoning_effort: str | None = None,
    is_gufo: bool = False,
) -> dict[str, Any]:
    """WS llama payload with the generation cap applied when the request
    set no max_output_tokens. Bounds turn duration for the harness's 30s
    terminal timer (and UI responsiveness): a thinking model at ~12 tok/s
    blows through that on reasoning alone."""
    if is_gufo:
        payload = gufo_native_payload(
            request, history, stream=True, reasoning_effort=reasoning_effort
        )
        if request.max_output_tokens is None:
            payload["max_output_tokens"] = WS_MAX_TOKENS
        return payload
    payload = _llama_payload(
        request, history, stream=True, reasoning_effort=reasoning_effort
    )
    if request.max_output_tokens is None:
        payload["max_tokens"] = WS_MAX_TOKENS
    return payload


def _validate_tool_outputs(
    request: CreateResponseBody, history: list[dict[str, Any]]
) -> None:
    """Reject tool results that do not correspond to a prior tool call."""
    call_ids: set[str] = set()
    items = [*history, *_stored_input_items(request)]
    for item in items:
        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id")
            if call_id:
                call_ids.add(call_id)
        elif (
            item_type == "function_call_output" and item.get("call_id") not in call_ids
        ):
            raise TranslationError(
                f"No matching tool call exists for call_id '{item.get('call_id')}'"
            )


async def _stream_to_ws(
    websocket: WebSocket,
    request: CreateResponseBody,
    history: list[dict[str, Any]] | None = None,
    disconnected: asyncio.Event | None = None,
) -> dict[str, Any] | None:
    """Run one response turn, pushing streaming event frames to the socket.

    Returns the serialized terminal ResponseResource, or None on failure
    (after sending response.failed).
    """
    lease = None
    server = None
    lease_request_id = "-"
    started = time.monotonic()
    upstream_status: int | None = None
    upstream_opened = False
    upstream_lines = 0
    emitted_events = 0
    done_marker_seen = False
    telemetry_recorded = False
    terminal_outcome = "completed"
    seq = ev.SSEmitter()
    try:
        server, model, preferred_server_id = await resolve_target(request.model)

        async def is_disconnected() -> bool:
            return disconnected is not None and disconnected.is_set()

        lease = await inference_scheduler.acquire(
            model.id if model else server.model_id,
            new_id("resp"),
            preferred_server_id=preferred_server_id,
            is_cancelled=is_disconnected,
        )
        lease_request_id = lease.request_id
        server = lease.server
        is_gufo = getattr(server, "engine", None) == "gufo"
        agent = await agent_manager.get_agent(str(server.agent_id))
        if not agent:
            raise TargetError(503, "no_agent_available", "Agent not found")

        history = list(history or [])
        if request.previous_response_id and request.store:
            # store=true: hydrate from DB like the HTTP path
            with Session(engine) as session:
                previous = _load_previous(session, request.previous_response_id)
                history = _build_chain_history(session, previous)
        _validate_tool_outputs(request, history)

        seq = ev.SSEmitter()
        response_id = new_id("resp")
        resource = build_response_resource(
            request,
            response_id,
            int(time.time()),
            previous_response_id=request.previous_response_id,
        )
        ev.response_created(seq, resource)
        ev.response_in_progress(seq, resource)
        await _send_events(websocket, seq.drain_events())

        state = StreamState(seq, request.model)
        state.assistant_phase = _last_assistant_phase(request)
        try:
            reasoning_effort = resolve_reasoning_effort(
                server, request.reasoning.effort if request.reasoning else None
            )
        except ReasoningPolicyError as e:
            await websocket.send_text(
                _error_event(400, "unsupported_capability", str(e), "reasoning.effort")
            )
            return None
        payload = _capped_payload(request, history, reasoning_effort, is_gufo=is_gufo)

        proxy_url = (
            f"http://{agent.host}:{agent.port}/proxy/{server.id}/v1/responses"
            if is_gufo
            else f"http://{agent.host}:{agent.port}/proxy/{server.id}/v1/chat/completions"
        )
        lease.mark_upstream_started()
        finish_reason: str | None = None
        final_usage: dict[str, Any] = {}
        fallback_chars = 0
        client = get_http_client()
        async with client.stream(
            "POST",
            proxy_url,
            json=payload,
            headers={
                **lease.dispatch_headers(),
            },
        ) as upstream:
            upstream_opened = True
            upstream_status = upstream.status_code
            logger.info(
                "responses_ws_upstream_open request_id=%s response_id=%s server_id=%s agent_id=%s upstream_status=%s",
                lease.request_id,
                response_id,
                server.id,
                server.agent_id,
                upstream_status,
            )
            upstream.raise_for_status()
            lines = upstream.aiter_lines()
            gufo_calls: dict[str, dict[str, Any]] = {}
            while True:
                try:
                    line = await lease.guard(anext(lines), cancelled=disconnected)
                except StopAsyncIteration:
                    break
                upstream_lines += 1
                if not line.startswith("data:"):
                    continue
                data = line[5:].lstrip()
                if is_gufo:
                    # gufo emits typed OpenResponses SSE (no [DONE] sentinel);
                    # response.completed/incomplete is the completion marker.
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    terminal, usage_data, incomplete = decode_gufo_stream_event(
                        state, event, gufo_calls
                    )
                    if usage_data is not None:
                        final_usage = usage_data
                    events = seq.drain_events()
                    emitted_events += len(events)
                    await _send_events(websocket, events)
                    if terminal:
                        if incomplete:
                            finish_reason = "length"
                        done_marker_seen = True
                        break
                    continue
                if data.strip() == "[DONE]":
                    done_marker_seen = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk:
                    raise RuntimeError(f"LLM error: {chunk['error']}")
                usage_data = chunk.get("usage")
                if isinstance(usage_data, dict) and usage_data:
                    final_usage = {
                        **usage_data,
                        "timings": chunk.get("timings") or {},
                    }
                if "tokens_evaluated" in chunk or "tokens_predicted" in chunk:
                    final_usage = {
                        "prompt_tokens": chunk.get("tokens_evaluated", 0),
                        "completion_tokens": chunk.get("tokens_predicted", 0),
                    }
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr
                reasoning_delta = delta.get("reasoning_content") or ""
                if reasoning_delta:
                    state.add_reasoning_delta(reasoning_delta)
                content_delta = delta.get("content") or ""
                if content_delta:
                    fallback_chars += len(content_delta)
                    state.add_text_delta(content_delta)
                for tc in delta.get("tool_calls") or []:
                    state.add_tool_call_delta(tc.get("index", 0), tc)
                events = seq.drain_events()
                emitted_events += len(events)
                await _send_events(websocket, events)

        if not done_marker_seen:
            raise RuntimeError("LLM stream ended without a completion marker")
        logger.info(
            "responses_ws_upstream_close request_id=%s response_id=%s server_id=%s outcome=%s close_reason=%s upstream_status=%s done_marker=%s upstream_lines=%d emitted_events=%d duration_ms=%.1f",
            lease.request_id,
            response_id,
            server.id,
            "completed"
            if upstream_status is not None and upstream_status < 400
            else "error",
            "done_marker" if done_marker_seen else "upstream_eof_without_done_marker",
            upstream_status,
            done_marker_seen,
            upstream_lines,
            emitted_events,
            (time.monotonic() - started) * 1000.0,
        )

        # Record before release() and the close-out yields: the upstream
        # work is complete at this point, so a cancel during release or the
        # final sends must not downgrade a finished inference to cancelled.
        success_latency_ms = (time.monotonic() - started) * 1000.0
        if is_gufo:
            usage = gufo_usage_to_spec(final_usage)
        else:
            usage = _build_usage(final_usage, fallback_chars, state.reasoning_tokens)
        record_request_telemetry(
            server,
            usage,
            final_usage.get("timings"),
            latency_ms=success_latency_ms,
            status="success",
        )
        telemetry_recorded = True

        await lease.release()
        lease = None

        state.finish_reasoning()
        state.finish_calls()
        state.finish_message()
        await _send_events(websocket, seq.drain_events())

        if finish_reason == "length":
            resource.status = "incomplete"
            resource.incomplete_details = IncompleteDetails(reason="max_output_tokens")
        else:
            resource.status = "completed"
        resource.completed_at = int(time.time())
        # Typed models (not dicts) so pydantic serializes without warnings —
        # matches the HTTP streaming path (_coerced_output_items)
        resource.output = _coerced_output_items(state.output_items)
        resource.usage = Usage.model_validate(usage)
        if resource.status == "incomplete":
            ev.response_incomplete(seq, resource)
        else:
            ev.response_completed(seq, resource)
        await _send_events(websocket, seq.drain_events())

        serialized = serialize_spec(resource)
        if request.store:
            # Persist so store=true chaining (WS or HTTP) finds this turn.
            _persist_response(
                request=request,
                model_id=model.id if model else None,
                agent_id=server.agent_id,
                response_id=response_id,
                output_items=[i.model_dump(exclude_none=True) for i in resource.output],
                status=resource.status,
                usage=usage,
                error=resource.error,
                incomplete_reason=(
                    resource.incomplete_details.reason
                    if resource.incomplete_details
                    else None
                ),
            )
            logger.info(
                "responses %s: stored (chain continues from this turn)", response_id
            )
        return serialized
    except asyncio.CancelledError as exc:
        terminal_outcome = "cancelled"
        logger.info(
            "responses_ws_stream_close request_id=%s server_id=%s outcome=cancelled close_reason=%s exception=%s upstream_status=%s upstream_opened=%s upstream_lines=%d emitted_events=%d duration_ms=%.1f",
            lease_request_id,
            server.id if server is not None else "-",
            "client_disconnect"
            if disconnected and disconnected.is_set()
            else "task_cancelled",
            type(exc).__name__,
            upstream_status,
            upstream_opened,
            upstream_lines,
            emitted_events,
            (time.monotonic() - started) * 1000.0,
        )
        if not telemetry_recorded:
            record_request_telemetry(
                server,
                None,
                None,
                latency_ms=(time.monotonic() - started) * 1000.0,
                status="cancelled",
            )
        raise
    except TargetError as e:
        terminal_outcome = "failed"
        if not telemetry_recorded:
            record_request_telemetry(
                server,
                None,
                None,
                latency_ms=(time.monotonic() - started) * 1000.0,
                status="error",
            )
        await websocket.send_text(
            _error_event(
                503 if e.status >= 500 else e.status, e.code, e.message, "model"
            )
        )
        return None
    except Exception as e:
        terminal_outcome = "failed"
        logger.warning(
            "responses_ws_stream_close request_id=%s server_id=%s outcome=error close_reason=%s error_type=%s error=%s upstream_status=%s upstream_opened=%s upstream_lines=%d emitted_events=%d duration_ms=%.1f",
            lease_request_id,
            server.id if server is not None else "-",
            "client_disconnect"
            if disconnected and disconnected.is_set()
            else "upstream_or_websocket_error",
            type(e).__name__,
            str(e),
            upstream_status,
            upstream_opened,
            upstream_lines,
            emitted_events,
            (time.monotonic() - started) * 1000.0,
        )
        if not telemetry_recorded:
            record_request_telemetry(
                server,
                None,
                None,
                latency_ms=(time.monotonic() - started) * 1000.0,
                status="error",
            )
        logger.error(f"responses WS turn failed: {e}")
        seq = ev.SSEmitter()
        failed = build_response_resource(request, new_id("resp"), int(time.time()))
        failed.status = "failed"
        failed.error = Error(code="server_error", message=str(e))
        ev.response_failed(seq, failed)
        await _send_events(websocket, seq.drain_events())
        return None
    finally:
        if lease is not None:
            await lease.release(terminal_outcome)


@router.websocket("/responses")
async def responses_websocket(websocket: WebSocket) -> None:
    """Spec WebSocket transport: sequential response.create turns."""
    await websocket.accept()
    deadline = time.monotonic() + CONNECTION_LIMIT_SECONDS
    # Connection-local continuation cache (latest turn, store=false)
    cache: dict[str, dict[str, Any]] = {}
    messages: asyncio.Queue[str | None] = asyncio.Queue(maxsize=16)
    disconnected = asyncio.Event()

    async def read_messages() -> None:
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    messages.put_nowait(raw)
                except asyncio.QueueFull:
                    disconnected.set()
                    return
        except WebSocketDisconnect:
            disconnected.set()
            try:
                messages.put_nowait(None)
            except asyncio.QueueFull:
                pass

    reader = asyncio.create_task(read_messages())

    try:
        while True:
            if disconnected.is_set():
                break
            if time.monotonic() > deadline:
                await websocket.send_text(
                    _error_event(
                        400,
                        "websocket_connection_limit_reached",
                        "WebSocket connections are limited to 60 minutes.",
                        None,
                    )
                )
                break

            raw = await messages.get()
            if raw is None:
                break
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(
                    _error_event(400, "invalid_request", "Invalid JSON envelope.", None)
                )
                continue
            if (
                not isinstance(message, dict)
                or message.get("type") != "response.create"
            ):
                await websocket.send_text(
                    _error_event(
                        400,
                        "invalid_request",
                        'Message must be {"type": "response.create", ...}.',
                        "type",
                    )
                )
                continue

            body = {k: v for k, v in message.items() if k != "type"}
            # Transport-specific fields MUST NOT be sent over WS
            for forbidden in ("stream", "stream_options", "background"):
                body.pop(forbidden, None)

            try:
                request = CreateResponseBody.model_validate(body)
            except ValidationError as e:
                await websocket.send_text(
                    _error_event(400, "invalid_request", str(e), None)
                )
                continue

            # previous resolution: DB for store=true, cache for store=false
            continuation_history: list[dict[str, Any]] | None = None
            if request.previous_response_id:
                missing = False
                if request.store:
                    try:
                        with Session(engine) as session:
                            _load_previous(session, request.previous_response_id)
                    except Exception:
                        missing = True
                else:
                    cached = cache.get(request.previous_response_id)
                    if cached is None:
                        missing = True
                    else:
                        continuation_history = cached["history"]
                if missing:
                    await websocket.send_text(
                        _error_event(
                            404,
                            "previous_response_not_found",
                            "Previous response with id "
                            f"'{request.previous_response_id}' not found"
                            + (
                                " (store=false is connection-local)."
                                if not request.store
                                else "."
                            ),
                            "previous_response_id",
                        )
                    )
                    continue

            terminal = await _stream_to_ws(
                websocket, request, continuation_history, disconnected
            )
            failed = terminal is None or terminal.get("status") == "failed"
            if failed:
                if request.previous_response_id:
                    cache.pop(request.previous_response_id, None)
            elif terminal is not None:
                output_items = terminal.get("output") or []
                cache[str(terminal.get("id"))] = {
                    "response": terminal,
                    "history": [
                        *(continuation_history or []),
                        *_stored_input_items(request),
                        *output_items,
                    ],
                }
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"responses WS error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)

"""OpenResponses API — POST /v1/responses (spec v2026-04-24 core).

Non-streaming JSON and SSE streaming, previous_response_id chaining,
function tools via llama.cpp native tool calling, reasoning passthrough.
"""

import json
import logging
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import TypeAdapter
from sqlmodel import Session, select

from app.api.deps import get_db
from app.api.routes.v1.responses import events as ev
from app.api.routes.v1.responses.schemas import (
    CreateResponseBody,
    Error,
    ErrorBody,
    ErrorResponse,
    IncompleteDetails,
    OutputItem,
    ResponseResource,
    Usage,
    new_id,
)
from app.api.routes.v1.responses.translator import (
    StreamState,
    TranslationError,
    allowed_tool_names,
    build_response_resource,
    input_items_to_llama_messages,
    tools_to_llama,
)
from app.core.db import engine
from app.models import Model, ResponseRecord
from app.services.agent_manager import agent_manager
from app.services.server_startup import (
    ServerStartupError,
    ensure_server_ready,
    ensure_server_ready_by_id,
    find_alias_instance,
)

logger = logging.getLogger(__name__)

router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _error_response(
    status: int, etype: str, code: str, message: str, param: str | None = None
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(message=message, type=etype, param=param, code=code)
    )
    return JSONResponse(status_code=status, content=body.model_dump())


def _resolve_model(db: Session, name: str) -> Model:
    """Model by name or id (server aliases are handled by the caller)."""
    model = db.exec(select(Model).where(Model.name == name)).first()
    if not model:
        try:
            model = db.exec(select(Model).where(Model.id == name)).first()
        except Exception:
            model = None
    if not model:
        raise HTTPException(404, f"Model {name} not found")
    return model


def _load_previous(
    db: Session, previous_response_id: str | None
) -> ResponseRecord | None:
    if not previous_response_id:
        return None
    record = db.exec(
        select(ResponseRecord).where(
            ResponseRecord.response_id == previous_response_id,
            ResponseRecord.store.is_(True),
        )
    ).first()
    if record is None:
        raise HTTPException(
            404, f"Previous response with id '{previous_response_id}' not found."
        )
    return record


def _llama_payload(
    request: CreateResponseBody, history: list[dict[str, Any]], stream: bool
) -> dict[str, Any]:
    messages = input_items_to_llama_messages(request, history)
    payload: dict[str, Any] = {
        "messages": messages,
        "stream": stream,
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.presence_penalty is not None:
        payload["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None:
        payload["frequency_penalty"] = request.frequency_penalty
    if request.max_output_tokens is not None:
        payload["max_tokens"] = request.max_output_tokens
    if request.text and request.text.format and request.text.format.type != "text":
        fmt = request.text.format
        if fmt.type == "json_schema" and getattr(fmt, "schema_", None):
            payload["json_schema"] = {"schema": fmt.schema_}
        elif fmt.type == "json_object":
            payload["response_format"] = {"type": "json_object"}
    payload.update(tools_to_llama(request))
    return payload


_OUTPUT_ITEM_ADAPTER: TypeAdapter = TypeAdapter(OutputItem)


def _coerce_output_item(item: dict[str, Any]) -> OutputItem:
    """Validate a raw item dict into the proper OutputItem model (spec union)."""
    return _OUTPUT_ITEM_ADAPTER.validate_python(item)


def _build_usage(
    usage_data: dict[str, Any], fallback_chars: int, reasoning_tokens: int
) -> dict[str, Any]:
    prompt_tokens = int(usage_data.get("prompt_tokens") or 0)
    completion_tokens = int(usage_data.get("completion_tokens") or 0)
    if not prompt_tokens and not completion_tokens:
        completion_tokens = fallback_chars // 4
    return ev.usage_from_llama(
        prompt_tokens, completion_tokens, reasoning_tokens
    ).model_dump()


# ============================================================================
# Non-streaming path
# ============================================================================


async def _complete(
    request: CreateResponseBody,
    server_id: str,
    agent_id: str,
    history: list[dict[str, Any]],
    response_id: str,
    created_at: int,
) -> ResponseResource:
    payload = _llama_payload(request, history, stream=False)
    response = await agent_manager.send_to_agent(
        agent_id,
        "POST",
        f"/proxy/{server_id}/v1/chat/completions",
        payload,
        timeout=300.0,
    )

    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")
    usage_data = response.get("usage") or {}

    seq = ev.SSEmitter()
    state = StreamState(seq, request.model)
    allowed = allowed_tool_names(request)

    reasoning_content = message.get("reasoning_content") or ""
    if reasoning_content:
        state.add_reasoning_delta(reasoning_content)
        state.finish_reasoning()

    content = message.get("content") or ""
    if content:
        state.add_text_delta(content)

    for tc in message.get("tool_calls") or []:
        index = tc.get("index", len(state._calls))
        state.add_tool_call_delta(index, tc)
    state.apply_allowed_tools(allowed)

    state.finish_calls()
    state.finish_message()

    incomplete = finish_reason == "length"
    usage = _build_usage(usage_data, len(content), state.reasoning_tokens)

    result = build_response_resource(
        request,
        response_id,
        created_at,
        status="incomplete" if incomplete else "completed",
        output=[_coerce_output_item(item) for item in state.output_items],
        usage=usage,
    )
    if incomplete:
        result.incomplete_details = IncompleteDetails(reason="max_output_tokens")
    return result


def _stored_input_items(request: CreateResponseBody) -> list[dict[str, Any]]:
    """Request input as plain dicts for the responses table."""
    if isinstance(request.input, str):
        return [{"type": "message", "role": "user", "content": request.input}]
    return [item.model_dump(exclude_none=True) for item in request.input]


def _persist_response(
    request: CreateResponseBody,
    model_id: Any,
    agent_id: Any,
    response_id: str,
    output_items: list[dict[str, Any]],
    status: str,
    usage: dict[str, Any] | None,
    error: Error | None,
    incomplete_reason: str | None,
) -> None:
    """Insert the ResponseRecord for a stored (store=true) response.

    Opens its own session: streaming generators run after the request's
    dependency session has been closed.
    """
    record = ResponseRecord(
        response_id=response_id,
        previous_response_id=request.previous_response_id,
        input_items=_stored_input_items(request),
        output_items=output_items,
        model_id=model_id,
        agent_id=agent_id,
        parameters={
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_output_tokens": request.max_output_tokens,
            "tool_choice": (
                request.tool_choice
                if isinstance(request.tool_choice, str)
                else str(request.tool_choice)
            ),
            "truncation": request.truncation,
        },
        response_metadata=request.metadata,
        status=status,
        error_code=error.code if error else None,
        error_message=error.message if error else None,
        incomplete_reason=incomplete_reason,
        input_tokens=usage["input_tokens"] if usage else 0,
        output_tokens=usage["output_tokens"] if usage else 0,
        total_tokens=usage["total_tokens"] if usage else 0,
        store=True,
        background=False,
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    with Session(engine) as session:
        session.add(record)
        session.commit()


# ============================================================================
# Streaming path
# ============================================================================


async def _stream_events(
    request: CreateResponseBody,
    server_id: str,
    agent_id: str,
    history: list[dict[str, Any]],
    response_id: str,
    created_at: int,
    model_id: Any | None = None,
    instance_agent_id: Any | None = None,
    persist: bool = False,
) -> AsyncIterator[str]:
    seq = ev.SSEmitter()
    response = build_response_resource(
        request,
        response_id,
        created_at,
        previous_response_id=request.previous_response_id,
    )
    # Build then drain: make() queues events so everything flows through
    # drain_frames() exactly once (no double-emitted lifecycle frames).
    ev.response_created(seq, response)
    ev.response_in_progress(seq, response)
    for frame in seq.drain_frames():
        yield frame

    state = StreamState(seq, request.model)
    allowed = allowed_tool_names(request)
    fallback_chars = 0
    final_usage_data: dict[str, Any] = {}
    finish_reason: str | None = None

    payload = _llama_payload(request, history, stream=True)

    try:
        # Hold the client connection while any in-flight startup completes
        # so the first event arrives as soon as the server is healthy.
        fresh_server = await ensure_server_ready_by_id(server_id)
        agent_id = str(fresh_server.agent_id)

        agent = await agent_manager.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        proxy_url = (
            f"http://{agent.host}:{agent.port}/proxy/{server_id}/v1/chat/completions"
        )
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", proxy_url, json=payload) as upstream:
                upstream.raise_for_status()
                async for line in upstream.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    usage_data = chunk.get("usage")
                    if isinstance(usage_data, dict) and usage_data:
                        final_usage_data = usage_data
                    if "tokens_evaluated" in chunk or "tokens_predicted" in chunk:
                        final_usage_data = {
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
                    if fr == "reasoning_content":
                        state.finish_reasoning()

                    content_delta = delta.get("content") or ""
                    if content_delta:
                        fallback_chars += len(content_delta)
                        state.add_text_delta(content_delta)

                    for tc in delta.get("tool_calls") or []:
                        state.add_tool_call_delta(tc.get("index", 0), tc)

                    # Emit the item/delta events built during this chunk
                    for frame in seq.drain_frames():
                        yield frame

        state.apply_allowed_tools(allowed)
        state.finish_reasoning()
        state.finish_calls()
        state.finish_message()

        # Drain the final close-out events (arguments done, item done, etc.)
        for frame in seq.drain_frames():
            yield frame

        usage = _build_usage(final_usage_data, fallback_chars, state.reasoning_tokens)

        if finish_reason == "length":
            response.status = "incomplete"
            response.incomplete_details = IncompleteDetails(reason="max_output_tokens")
        else:
            response.status = "completed"
        # Typed models (not dicts) so pydantic serializes without warnings
        response.output = [_coerce_output_item(item) for item in state.output_items]
        response.usage = Usage.model_validate(usage)

        if response.status == "incomplete":
            ev.response_incomplete(seq, response)
        else:
            ev.response_completed(seq, response)
        for frame in seq.drain_frames():
            yield frame

        if persist:
            _persist_response(
                request=request,
                model_id=model_id,
                agent_id=instance_agent_id,
                response_id=response_id,
                output_items=[i.model_dump(exclude_none=True) for i in response.output],
                status=response.status,
                usage=usage,
                error=response.error,
                incomplete_reason=(
                    response.incomplete_details.reason
                    if response.incomplete_details
                    else None
                ),
            )

    except Exception as e:
        logger.error(f"Responses streaming error: {e}")
        response.status = "failed"
        response.error = Error(code="server_error", message=str(e))
        ev.response_failed(seq, response)
        for frame in seq.drain_frames():
            yield frame

        if persist:
            _persist_response(
                request=request,
                model_id=model_id,
                agent_id=instance_agent_id,
                response_id=response_id,
                output_items=[],
                status=response.status,
                usage=None,
                error=response.error,
                incomplete_reason=None,
            )

    yield "data: [DONE]\n\n"


# ============================================================================
# Route
# ============================================================================


@router.post("/responses", response_model=None)
async def create_response(
    request: CreateResponseBody,
    db: Session = Depends(get_db),
) -> ResponseResource | StreamingResponse | JSONResponse:
    """Create a model response (Open Responses spec).

    Cold starts are handled transparently: a stopped/errored server for
    the requested alias is (re)started and the request waits for health
    before generation, so clients just see a longer first-token latency.
    """
    if request.background:
        return _error_response(
            400,
            "invalid_request",
            "background_not_supported",
            "background=true is not supported by this implementation.",
            param="background",
        )

    try:
        previous = _load_previous(db, request.previous_response_id)
    except HTTPException as e:
        return _error_response(
            404,
            "not_found",
            "previous_response_not_found",
            str(e.detail),
            param="previous_response_id",
        )

    from app.api.routes.v1.v1_chat_completions import _get_or_create_server

    # Resolve the target server. Server aliases route directly (with
    # auto-start); names/ids resolve to a model and a server is found or
    # started for it.
    server = None
    model: Model | None = None
    instance = await find_alias_instance(request.model)
    if instance is not None:
        try:
            server = await ensure_server_ready(instance)
        except ServerStartupError as e:
            return _error_response(
                503,
                "too_many_requests",
                "no_agent_available",
                f"Server not available: {e}",
            )
        model = db.exec(select(Model).where(Model.id == server.model_id)).first()
        if not model:
            return _error_response(
                404,
                "not_found",
                "model_not_found",
                f"Model for server alias {request.model} not found",
                param="model",
            )
    else:
        try:
            model = _resolve_model(db, request.model)
        except HTTPException as e:
            return _error_response(
                e.status_code or 404,
                "not_found" if e.status_code == 404 else "invalid_request",
                "model_not_found",
                str(e.detail),
                param="model",
            )
        try:
            server = await _get_or_create_server(model, None)
            server = await ensure_server_ready(server)
        except HTTPException as e:
            return _error_response(
                e.status_code or 503,
                "too_many_requests" if e.status_code == 503 else "server_error",
                "no_agent_available",
                str(e.detail),
            )
        except ServerStartupError as e:
            return _error_response(
                503,
                "too_many_requests",
                "no_agent_available",
                f"Server not available: {e}",
            )

    # Chaining context: previous.input + previous.output + this input
    history: list[dict[str, Any]] = []
    if previous is not None:
        history.extend(previous.input_items or [])
        history.extend(previous.output_items or [])

    response_id = new_id("resp")
    created_at = int(time.time())

    if request.stream:
        return StreamingResponse(
            _stream_events(
                request,
                str(server.id),
                str(server.agent_id),
                history,
                response_id,
                created_at,
                model_id=model.id,
                instance_agent_id=server.agent_id,
                persist=request.store,
            ),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    try:
        result = await _complete(
            request,
            str(server.id),
            str(server.agent_id),
            history,
            response_id,
            created_at,
        )
    except TranslationError as e:
        return _error_response(400, "invalid_request", "invalid_input", str(e))
    except Exception as e:
        logger.error(f"Responses completion error: {e}")
        return _error_response(500, "model_error", "model_error", str(e))

    if request.store:
        _persist_response(
            request=request,
            model_id=model.id,
            agent_id=server.agent_id,
            response_id=response_id,
            output_items=[i.model_dump(exclude_none=True) for i in result.output],
            status=result.status,
            usage=result.usage.model_dump() if result.usage else None,
            error=result.error,
            incomplete_reason=(
                result.incomplete_details.reason if result.incomplete_details else None
            ),
        )

    return result

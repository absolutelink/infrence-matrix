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
from pydantic import BaseModel, TypeAdapter
from sqlmodel import Session, select

from app.api.deps import get_db
from app.api.routes.v1.responses import events as ev
from app.api.routes.v1.responses.schemas import (
    CompactionItem,
    CompactRequestBody,
    CompactResource,
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
from app.models import Model, ResponseRecord, ServerInstance
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
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True))


def serialize_dict(obj: Any) -> Any:
    """Pydantic-safe dump for logging request shapes (models or plain dicts)."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(exclude_none=True)
    if isinstance(obj, dict):
        return {k: serialize_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_dict(v) for v in obj]
    return obj


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


def _build_chain_history(
    db: Session, previous: ResponseRecord | None
) -> list[dict[str, Any]]:
    """Full conversation context for chaining: walk previous_response_id
    links back to the root and collect input+output in order.

    Spec: the model is sampled over previous.input + previous.output + input.
    Each record stores only its own delta, so a single hop would drop
    earlier turns — the chain must be resolved recursively.
    """
    if previous is None:
        return []
    chain: list[ResponseRecord] = []
    seen: set[str] = set()
    current: ResponseRecord | None = previous
    while current is not None and current.response_id not in seen:
        seen.add(current.response_id)
        chain.append(current)
        if not current.previous_response_id:
            break
        current = db.exec(
            select(ResponseRecord).where(
                ResponseRecord.response_id == current.previous_response_id,
                ResponseRecord.store.is_(True),
            )
        ).first()
    history: list[dict[str, Any]] = []
    for record in reversed(chain):
        history.extend(record.input_items or [])
        history.extend(record.output_items or [])
    return history


def _chain_depth(db: Session, previous: ResponseRecord | None) -> int:
    """Number of stored records in the previous_response_id chain."""
    depth = 0
    seen: set[str] = set()
    current: ResponseRecord | None = previous
    while current is not None and current.response_id not in seen:
        seen.add(current.response_id)
        depth += 1
        if not current.previous_response_id:
            break
        current = db.exec(
            select(ResponseRecord).where(
                ResponseRecord.response_id == current.previous_response_id,
                ResponseRecord.store.is_(True),
            )
        ).first()
    return depth


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


def _coerced_output_items(items: list[dict[str, Any]]) -> list[OutputItem]:
    """Coerce raw item dicts; drop unset optional keys (phase/logprobs/
    encrypted_content serialize as None otherwise, which fails the
    conformance schemas' optional() (key-absent) semantics)."""
    return [
        _coerce_output_item({k: v for k, v in item.items() if v is not None})
        for item in items
    ]


def _last_assistant_phase(request: CreateResponseBody) -> str | None:
    """Phase label of the final assistant message in the request input,
    for echoing on this response's output message (spec 2026-04-24)."""
    if isinstance(request.input, str):
        return None
    phase: str | None = None
    for item in request.input:
        if (
            item.type == "message"
            and item.role == "assistant"
            and getattr(item, "phase", None)
        ):
            phase = item.phase
    return phase


def _build_usage(
    usage_data: dict[str, Any], fallback_chars: int, reasoning_tokens: int
) -> dict[str, Any]:
    prompt_tokens = int(usage_data.get("prompt_tokens") or 0)
    completion_tokens = int(usage_data.get("completion_tokens") or 0)
    if not prompt_tokens and not completion_tokens:
        completion_tokens = fallback_chars // 4
    return ev.usage_from_llama(
        prompt_tokens, completion_tokens, reasoning_tokens
    ).model_dump(exclude_none=True)


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
    state.assistant_phase = _last_assistant_phase(request)
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
        output=_coerced_output_items(state.output_items),
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
    state.assistant_phase = _last_assistant_phase(request)
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
        response.output = _coerced_output_items(state.output_items)
        response.usage = Usage.model_validate(usage)

        if response.status == "incomplete":
            ev.response_incomplete(seq, response)
        else:
            ev.response_completed(seq, response)
        for frame in seq.drain_frames():
            yield frame

        logger.info(
            "responses %s: %s status=%s items=%d usage=%s finish_reason=%s",
            response_id,
            "streamed" if request.stream else "completed",
            response.status,
            len(response.output),
            f"{response.usage.input_tokens}/{response.usage.output_tokens} tok"
            if response.usage
            else "n/a",
            finish_reason,
        )

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
            logger.info(
                "responses %s: stored (chain continues from this turn)",
                response_id,
            )
        else:
            logger.info("responses %s: not stored (store=false)", response_id)

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


class TargetError(Exception):
    """Server/model resolution failure carrying spec error fields."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


async def resolve_target(model_ref: str) -> tuple[ServerInstance, Model | None]:
    """Resolve a server alias / model name / model id to a ready server.

    Aliases route directly (auto-starting stopped/errored instances);
    names/ids resolve to a model and a server is found or started. Raises
    TargetError with spec error codes on failure.
    """
    from app.api.routes.v1.v1_chat_completions import _get_or_create_server

    instance = await find_alias_instance(model_ref)
    if instance is not None:
        try:
            server = await ensure_server_ready(instance)
        except ServerStartupError as e:
            raise TargetError(
                503, "no_agent_available", f"Server not available: {e}"
            ) from e
        with Session(engine) as session:
            model = session.exec(
                select(Model).where(Model.id == server.model_id)
            ).first()
        if not model:
            raise TargetError(
                404, "model_not_found", f"Model for server alias {model_ref} not found"
            )
        return server, model

    with Session(engine) as session:
        try:
            model = _resolve_model(session, model_ref)
        except HTTPException as e:
            if e.status_code == 404:
                raise TargetError(404, "model_not_found", str(e.detail)) from e
            raise TargetError(400, "invalid_request", str(e.detail)) from e

    try:
        server = await _get_or_create_server(model, None)
        server = await ensure_server_ready(server)
    except HTTPException as e:
        raise TargetError(
            e.status_code or 503, "no_agent_available", str(e.detail)
        ) from e
    except ServerStartupError as e:
        raise TargetError(
            503, "no_agent_available", f"Server not available: {e}"
        ) from e
    except Exception as e:
        raise TargetError(
            503, "no_agent_available", f"Failed to start server: {e}"
        ) from e
    return server, model


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
        logger.warning(
            "responses: previous_response_id '%s' not found — %s",
            request.previous_response_id,
            e.detail,
        )
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

    # Chaining context: walk the previous_response_id chain to the root and
    # concatenate each record's input+output in order (full conversation).
    history: list[dict[str, Any]] = _build_chain_history(db, previous)

    # Log the actual message list sent to llama.cpp (request input + chain
    # history) — this is the ground truth for context debugging.
    try:
        llama_messages = input_items_to_llama_messages(request, history)
    except TranslationError as e:
        return _error_response(400, "invalid_request", "invalid_input", str(e))
    continuation = previous is not None
    tool_names = [t.name for t in request.tools]
    tool_choice_dump = (
        request.tool_choice
        if isinstance(request.tool_choice, str)
        else serialize_dict(request.tool_choice)
    )
    logger.info(
        "responses: %s model=%s server=%s stream=%s store=%s "
        "previous_response_id=%s chain_depth=%d history_items=%d "
        "history_chars=%d llama_messages=%d prompt_chars=%d "
        "tools=%s tool_choice=%s parallel_tool_calls=%s",
        "continuation" if continuation else "new response",
        request.model,
        server.id,
        request.stream,
        request.store,
        request.previous_response_id or "none",
        _chain_depth(db, previous),
        len(history),
        sum(len(json.dumps(item)) for item in history),
        len(llama_messages),
        sum(
            len(m["content"])
            if isinstance(m["content"], str)
            else len(json.dumps(m["content"]))
            for m in llama_messages
        ),
        tool_names or "none",
        tool_choice_dump,
        request.parallel_tool_calls,
    )
    logger.info(
        "responses input items: %s",
        [
            (
                item.get("type", "message")
                if isinstance(item, dict)
                else getattr(item, "type", "unknown")
            )
            for item in (
                request.input if isinstance(request.input, list) else [request.input]
            )
        ]
        if request.input
        else [],
    )
    logger.info("responses history item types: %s", [h.get("type") for h in history])
    logger.debug(
        "responses prompt: %s",
        " | ".join(
            f"{m['role']}:{m['content'][:80] if isinstance(m['content'], str) else '<parts>'}"
            for m in llama_messages
        ),
    )
    logger.debug(
        "responses request: %s",
        json.dumps(serialize_dict(request), ensure_ascii=False)[:4000],
    )

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
            usage=result.usage.model_dump(exclude_none=True) if result.usage else None,
            error=result.error,
            incomplete_reason=(
                result.incomplete_details.reason if result.incomplete_details else None
            ),
        )
        logger.info(
            "responses %s: stored (chain continues from this turn)", response_id
        )
    else:
        logger.info("responses %s: not stored (store=false)", response_id)

    return result


# ============================================================================
# Compaction endpoint (spec 2026-04-24)
# ============================================================================


@router.post("/responses/compact", response_model=None)
async def compact_response(
    request: CompactRequestBody,
    db: Session = Depends(get_db),
) -> CompactResource | JSONResponse:
    """Compact a conversation into a portable compaction item.

    Runs a real summarization pass through the same llama.cpp proxy the
    Responses API uses; the summary becomes the compaction item's
    encrypted_content (opaque to clients, round-trippable as input).
    Stateless: results are not persisted; chain from them by passing the
    compacted output as input on a fresh response.
    """
    try:
        # Resolve server exactly like POST /responses (alias auto-start
        # or model name/id fallback).
        instance = await find_alias_instance(request.model)
        server = None
        if instance is not None:
            server = await ensure_server_ready(instance)
        else:
            model = _resolve_model(db, request.model)
            from app.api.routes.v1.v1_chat_completions import _get_or_create_server

            server = await _get_or_create_server(model, None)
            server = await ensure_server_ready(server)
    except ServerStartupError as e:
        return _error_response(
            503, "too_many_requests", "no_agent_available", f"Server not available: {e}"
        )
    except HTTPException as e:
        return _error_response(
            e.status_code or 500,
            "not_found" if e.status_code == 404 else "server_error",
            "model_not_found" if e.status_code == 404 else "server_error",
            str(e.detail),
        )

    history: list[dict[str, Any]] = []
    try:
        previous = _load_previous(db, request.previous_response_id)
    except HTTPException as e:
        return _error_response(
            404, "not_found", "previous_response_not_found", str(e.detail)
        )
    if previous is not None:
        history = _build_chain_history(db, previous)

    try:
        messages = input_items_to_llama_messages(request, history)
    except TranslationError as e:
        return _error_response(400, "invalid_request", "invalid_input", str(e))

    # Summarization prompt: a real sampling pass, not a passthrough.
    compact_system = (
        request.instructions
        or "Compress this conversation into a concise but complete summary "
        "preserving all key facts, decisions, and unresolved questions. "
        "Output only the compressed summary."
    )
    llama_payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": compact_system},
            *messages,
        ],
        "temperature": 0.3,
        "stream": False,
    }

    agent = await agent_manager.get_agent(str(server.agent_id))
    if not agent:
        return _error_response(
            503, "server_error", "no_agent_available", "Agent not found"
        )

    try:
        response = await agent_manager.send_to_agent(
            str(server.agent_id),
            "POST",
            f"/proxy/{server.id}/v1/chat/completions",
            llama_payload,
            timeout=300.0,
        )
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        summary_text = message.get("content") or ""
        usage_data = response.get("usage") or {}
    except Exception as e:
        logger.error(f"Compaction sampling error: {e}")
        return _error_response(500, "model_error", "model_error", str(e))

    prompt_tokens = int(usage_data.get("prompt_tokens") or 0)
    completion_tokens = int(usage_data.get("completion_tokens") or 0)

    item = CompactionItem(
        encrypted_content=summary_text,
    )
    resource = CompactResource(
        output=[item],
        created_at=int(time.time()),
        usage=Usage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    logger.info(
        "responses compact: model=%s input_chars=%d summary_chars=%d",
        request.model,
        sum(len(json.dumps(m)) for m in messages),
        len(summary_text),
    )
    return resource

"""OpenResponses API — POST /v1/responses (spec v2026-04-24 core).

Non-streaming JSON and SSE streaming, previous_response_id chaining,
function tools via llama.cpp native tool calling, reasoning passthrough.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
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
    serialize_spec,
)
from app.api.routes.v1.responses.translator import (
    StreamState,
    TranslationError,
    allowed_tool_names,
    build_response_resource,
    decode_gufo_stream_event,
    gufo_buffered_to_output_items,
    gufo_native_payload,
    gufo_usage_to_spec,
    input_items_to_llama_messages,
    tools_to_llama,
)
from app.core.db import engine
from app.models import Model, ResponseRecord, ServerInstance
from app.services.agent_manager import agent_manager
from app.services.http_client import get_http_client
from app.services.inference_scheduler import (
    InferenceLeaseHandle,
    inference_scheduler,
)
from app.services.inference_stream import release_lease_background
from app.services.inference_target import resolve_inference_target
from app.services.reasoning_metadata import (
    ReasoningPolicyError,
    resolve_reasoning_effort,
)
from app.services.server_startup import (
    ServerStartupError,
    ensure_server_ready_by_id,
    find_alias_instance,
    metadata_capability,
)
from app.services.token_stats import record_request_telemetry

logger = logging.getLogger(__name__)

# Keep intermediaries from treating a slow prompt prefill as a dead SSE
# connection. This is a transport comment, not an OpenResponses event.
SSE_KEEPALIVE_INTERVAL_SECONDS = 15.0

# Terminal marker for the keepalive reader queue. A unique object so it can
# never be confused with the None keepalive signal or a real upstream line.
_STREAM_END = object()

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
    request: CreateResponseBody,
    history: list[dict[str, Any]],
    stream: bool,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    messages = input_items_to_llama_messages(request, history)
    payload: dict[str, Any] = {
        "messages": messages,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.presence_penalty is not None:
        payload["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None:
        payload["frequency_penalty"] = request.frequency_penalty
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    # Only forward an explicit cap: defaulting to a small limit truncates
    # long outputs mid-stream (finish_reason "length"), matching the chat
    # completions pass-through semantics.
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
    if (
        "prompt_tokens" not in usage_data
        and "completion_tokens" not in usage_data
        and "input_tokens" not in usage_data
        and "output_tokens" not in usage_data
    ):
        completion_tokens = fallback_chars // 4
    prompt_details = usage_data.get("prompt_tokens_details") or {}
    cached_value = (
        prompt_details["cached_tokens"]
        if "cached_tokens" in prompt_details
        else (usage_data.get("timings") or {}).get("cache_n", 0)
    )
    cached_tokens = int(cached_value or 0)
    completion_details = usage_data.get("completion_tokens_details") or {}
    reported_reasoning_tokens = completion_details.get("reasoning_tokens")
    resolved_reasoning_tokens = (
        int(reported_reasoning_tokens)
        if reported_reasoning_tokens is not None
        else reasoning_tokens
    )
    return ev.usage_from_llama(
        prompt_tokens, completion_tokens, resolved_reasoning_tokens, cached_tokens
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
    lease: InferenceLeaseHandle,
    reasoning_effort: str | None = None,
    started_mono: float | None = None,
) -> ResponseResource:
    is_gufo = getattr(lease.server, "engine", None) == "gufo"
    if is_gufo:
        payload = gufo_native_payload(
            request, history, stream=False, reasoning_effort=reasoning_effort
        )
        upstream_path = f"/proxy/{server_id}/v1/responses"
    else:
        payload = _llama_payload(
            request, history, stream=False, reasoning_effort=reasoning_effort
        )
        upstream_path = f"/proxy/{server_id}/v1/chat/completions"
    started = started_mono if started_mono is not None else time.monotonic()
    response = await lease.guard(
        agent_manager.send_to_agent(
            agent_id,
            "POST",
            upstream_path,
            payload,
            timeout=1800.0,
            headers={
                **lease.dispatch_headers(),
            },
        )
    )

    if is_gufo:
        # gufo's native buffered response already carries final, correctly
        # ordered spec output items; pass them through instead of rebuilding
        # via StreamState (which would reorder calls before the message).
        # allowed_tools suppression is a no-op here: gufo's native path
        # rejects allowed_tools upstream, so no suppression set is applied.
        items = gufo_buffered_to_output_items(response)
        phase = _last_assistant_phase(request)
        if phase:
            for item in items:
                if item.get("type") == "message" and "phase" not in item:
                    item["phase"] = phase
        output_items = items
        usage = gufo_usage_to_spec(response.get("usage") or {})
        incomplete = response.get("status") == "incomplete"
        usage_data = {"timings": response.get("timings") or {}}
    else:
        state = StreamState(ev.SSEmitter(), request.model)
        state.assistant_phase = _last_assistant_phase(request)
        allowed = allowed_tool_names(request)

        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")
        usage_data = {
            **(response.get("usage") or {}),
            "timings": response.get("timings") or {},
        }

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

        output_items = state.output_items
        incomplete = finish_reason == "length"
        usage = _build_usage(usage_data, len(content), state.reasoning_tokens)

    result = build_response_resource(
        request,
        response_id,
        created_at,
        status="incomplete" if incomplete else "completed",
        output=_coerced_output_items(output_items),
        usage=usage,
    )
    result.completed_at = int(time.time())
    if incomplete:
        result.incomplete_details = IncompleteDetails(reason="max_output_tokens")
    # Record last: any failure above surfaces as an error from the route's
    # except handler instead of double-counting this request as success+error.
    record_request_telemetry(
        lease.server,
        usage,
        usage_data.get("timings"),
        latency_ms=(time.monotonic() - started) * 1000.0,
        status="success",
    )
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


async def _upstream_lines_with_keepalive(
    upstream: httpx.Response,
    lease: InferenceLeaseHandle | None = None,
    deadline: list[float] | None = None,
) -> AsyncIterator[str | None]:
    """Read upstream SSE lines while emitting comments during idle periods.

    The idle timer must never cancel the pending upstream read: cancelling a
    suspended ``__anext__`` closes the async generator, so the next read
    raises ``StopAsyncIteration`` and the rest of the stream is silently
    lost.  Instead a long-lived reader task feeds a size-1 queue and only
    the queue pop is timed out, which is always safe.  Yields ``None`` for
    "emit a keepalive comment now" and ``str`` for an upstream line.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
    if deadline is None:
        deadline = [time.monotonic() + SSE_KEEPALIVE_INTERVAL_SECONDS]
    shutting_down = False

    async def reader() -> None:
        lines = upstream.aiter_lines()
        try:
            while True:
                try:
                    if lease is not None:
                        line = await lease.guard(anext(lines))
                    else:
                        line = await anext(lines)
                except StopAsyncIteration:
                    break
                await queue.put(line)
        except asyncio.CancelledError as exc:
            # Our own teardown: drop the cancellation instead of reporting it.
            # Lease loss and client cancellation raise CancelledError subclasses
            # from guard while the consumer is still running, so those must be
            # delivered with their real type.
            if shutting_down:
                raise
            await queue.put(exc)
        except Exception as exc:
            await queue.put(exc)
        else:
            await queue.put(_STREAM_END)

    task = asyncio.create_task(reader())
    try:
        while True:
            # Transport lines are not client-visible. Even a steady stream of
            # discarded comments must not extend the downstream idle window.
            remaining = deadline[0] - time.monotonic()
            if remaining <= 0:
                yield None
                deadline[0] = time.monotonic() + SSE_KEEPALIVE_INTERVAL_SECONDS
                continue
            try:
                item = await asyncio.wait_for(queue.get(), timeout=remaining)
            except TimeoutError:
                yield None
                deadline[0] = time.monotonic() + SSE_KEEPALIVE_INTERVAL_SECONDS
                continue
            if item is _STREAM_END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        shutting_down = True
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _setup_keepalives(
    ready: asyncio.Future[Any],
    deadline: list[float],
    worker: asyncio.Task[Any] | None = None,
) -> AsyncIterator[None]:
    """Keep the same visible idle clock running while setup is pending.

    Waiting on task completion via asyncio.wait never cancels the owned work.
    The caller owns cancellation/join of the worker on every exit path.
    """
    while not ready.done():
        if worker is not None and worker.done():
            await worker  # propagate a failed header acquisition
        remaining = deadline[0] - time.monotonic()
        if remaining <= 0:
            yield None
            deadline[0] = time.monotonic() + SSE_KEEPALIVE_INTERVAL_SECONDS
            continue
        await asyncio.wait(
            {ready, worker} if worker is not None else {ready},
            timeout=remaining,
            return_when=asyncio.FIRST_COMPLETED,
        )
    if worker is not None and worker.done():
        await worker
    await ready


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
    lease: InferenceLeaseHandle | None = None,
    reasoning_effort: str | None = None,
    started_mono: float | None = None,
) -> AsyncIterator[str]:
    started = started_mono if started_mono is not None else time.monotonic()
    telemetry_recorded = False
    is_gufo = getattr(lease.server, "engine", None) == "gufo" if lease else False
    # Stash the serving server up front: the mid-body release sets the
    # lease to None, and later except handlers must keep attribution.
    sample_server = lease.server if lease is not None else None
    seq = ev.SSEmitter()
    response = build_response_resource(
        request,
        response_id,
        created_at,
        previous_response_id=request.previous_response_id,
    )
    # Build then drain: make() queues events so everything flows through
    # drain_frames() exactly once (no double-emitted lifecycle frames).
    # Keep the lease until the first frame has been resumed. If the client
    # disconnects at the initial yield, the generator's finally block below
    # has not started yet, so explicitly cover that early-close window.
    initial_frames_sent = False
    request_id = lease.request_id if lease is not None else response_id
    upstream_status: int | None = None
    upstream_opened = False
    upstream_lines = 0
    emitted_frames = 0
    done_marker_seen = False
    deadline = [time.monotonic() + SSE_KEEPALIVE_INTERVAL_SECONDS]
    try:
        ev.response_created(seq, response)
        ev.response_in_progress(seq, response)
        for frame in seq.drain_frames():
            yield frame
            deadline[0] = time.monotonic() + SSE_KEEPALIVE_INTERVAL_SECONDS
        initial_frames_sent = True
    except GeneratorExit, asyncio.CancelledError:
        logger.info(
            "responses_stream_close request_id=%s response_id=%s server_id=%s outcome=cancelled close_reason=downstream_disconnect_before_stream_setup duration_ms=%.1f",
            request_id,
            response_id,
            server_id,
            (time.monotonic() - started) * 1000.0,
        )
        # Early-close window: the later try/finally has not started, so
        # release here and label the failure by its real kind.
        if not initial_frames_sent and lease is not None:
            record_request_telemetry(
                lease.server,
                None,
                None,
                latency_ms=(time.monotonic() - started) * 1000.0,
                status="cancelled",
            )
            await lease.release("cancelled")
            lease = None
        raise
    except Exception:
        if not initial_frames_sent and lease is not None:
            record_request_telemetry(
                lease.server,
                None,
                None,
                latency_ms=(time.monotonic() - started) * 1000.0,
                status="error",
            )
            await lease.release("failed")
            lease = None
        raise

    try:
        state = StreamState(seq, request.model)
        state.assistant_phase = _last_assistant_phase(request)
        allowed = allowed_tool_names(request)
        fallback_chars = 0
        final_usage_data: dict[str, Any] = {}
        finish_reason: str | None = None

        payload = (
            gufo_native_payload(
                request, history, stream=True, reasoning_effort=reasoning_effort
            )
            if is_gufo
            else _llama_payload(
                request, history, stream=True, reasoning_effort=reasoning_effort
            )
        )

        # Hold the client connection while any in-flight startup completes
        # so the first event arrives as soon as the server is healthy.
        async def prepare_agent() -> tuple[str, Any]:
            fresh_server = await ensure_server_ready_by_id(server_id)
            fresh_agent_id = str(fresh_server.agent_id)
            agent = await agent_manager.get_agent(fresh_agent_id)
            if not agent:
                raise ValueError(f"Agent {fresh_agent_id} not found")
            return fresh_agent_id, agent

        readiness = asyncio.create_task(
            lease.guard(prepare_agent()) if lease is not None else prepare_agent()
        )
        try:
            async with aclosing(_setup_keepalives(readiness, deadline)) as ticks:
                async for _ in ticks:
                    yield ": keep-alive\n\n"
            agent_id, agent = await readiness
        finally:
            if not readiness.done():
                readiness.cancel()
            await asyncio.gather(readiness, return_exceptions=True)
        proxy_url = (
            f"http://{agent.host}:{agent.port}/proxy/{server_id}/v1/responses"
            if is_gufo
            else f"http://{agent.host}:{agent.port}/proxy/{server_id}/v1/chat/completions"
        )
        if lease is not None:
            lease.mark_upstream_started()
        client = get_http_client()
        # The worker owns the response context from header acquisition to
        # body close. If cancellation races with headers arriving, its
        # async-with still closes the response before the client closes.
        opened: asyncio.Future[httpx.Response] = (
            asyncio.get_running_loop().create_future()
        )
        close_upstream = asyncio.Event()

        async def upstream_context() -> None:
            async with client.stream(
                "POST",
                proxy_url,
                json=payload,
                headers={
                    **lease.dispatch_headers(),
                }
                if lease is not None
                else None,
            ) as response_stream:
                opened.set_result(response_stream)
                await close_upstream.wait()

        worker = asyncio.create_task(
            lease.guard(upstream_context()) if lease is not None else upstream_context()
        )
        finished_body = False
        try:
            async with aclosing(_setup_keepalives(opened, deadline, worker)) as ticks:
                async for _ in ticks:
                    yield ": keep-alive\n\n"
            upstream = opened.result()
            upstream_opened = True
            upstream_status = upstream.status_code
            logger.info(
                "responses_stream_open request_id=%s response_id=%s server_id=%s agent_id=%s upstream_status=%s",
                request_id,
                response_id,
                server_id,
                agent_id,
                upstream_status,
            )
            upstream.raise_for_status()
            if is_gufo:
                # gufo emits typed OpenResponses SSE events (no [DONE]
                # sentinel); the terminal response.completed/incomplete is
                # the completion marker. Feed deltas into the same
                # StreamState so broker IDs/sequence numbers are our own.
                gufo_calls: dict[str, dict[str, Any]] = {}
                async with aclosing(
                    _upstream_lines_with_keepalive(upstream, lease, deadline)
                ) as upstream_stream:
                    async for line in upstream_stream:
                        if line is None:
                            yield ": keep-alive\n\n"
                            continue
                        upstream_lines += 1
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].lstrip()
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        terminal, usage_data, incomplete = decode_gufo_stream_event(
                            state, event, gufo_calls
                        )
                        if usage_data is not None:
                            final_usage_data = usage_data
                        for frame in seq.drain_frames():
                            emitted_frames += 1
                            yield frame
                            deadline[0] = (
                                time.monotonic() + SSE_KEEPALIVE_INTERVAL_SECONDS
                            )
                        if terminal:
                            if incomplete:
                                finish_reason = "length"
                            done_marker_seen = True
                            break
            else:
                async with aclosing(
                    _upstream_lines_with_keepalive(upstream, lease, deadline)
                ) as upstream_stream:
                    async for line in upstream_stream:
                        if line is None:
                            yield ": keep-alive\n\n"
                            continue
                        upstream_lines += 1
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].lstrip()
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
                            final_usage_data = {
                                **usage_data,
                                "timings": chunk.get("timings") or {},
                            }
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
                            emitted_frames += 1
                            yield frame
                            deadline[0] = (
                                time.monotonic() + SSE_KEEPALIVE_INTERVAL_SECONDS
                            )
            if not done_marker_seen:
                raise RuntimeError("LLM stream ended without a completion marker")
            finished_body = True
        finally:
            close_upstream.set()
            if not finished_body and not worker.done():
                worker.cancel()
            if finished_body:
                await worker
            else:
                await asyncio.gather(worker, return_exceptions=True)

        logger.info(
            "responses_stream_upstream_close request_id=%s response_id=%s server_id=%s outcome=%s close_reason=%s upstream_status=%s upstream_opened=%s done_marker=%s upstream_lines=%d emitted_frames=%d duration_ms=%.1f",
            request_id,
            response_id,
            server_id,
            "completed"
            if upstream_status is not None and upstream_status < 400
            else "error",
            "done_marker" if done_marker_seen else "upstream_eof_without_done_marker",
            upstream_status,
            upstream_opened,
            done_marker_seen,
            upstream_lines,
            emitted_frames,
            (time.monotonic() - started) * 1000.0,
        )
        # Record before release() and the close-out yields: the upstream
        # work is complete at this point, so a cancel during release or the
        # final sends must not downgrade a finished inference to cancelled.
        success_latency_ms = (time.monotonic() - started) * 1000.0
        if is_gufo:
            usage = gufo_usage_to_spec(final_usage_data)
        else:
            usage = _build_usage(
                final_usage_data, fallback_chars, state.reasoning_tokens
            )
        record_request_telemetry(
            sample_server,
            usage,
            final_usage_data.get("timings"),
            latency_ms=success_latency_ms,
            status="success",
            request_id=request_id,
        )
        telemetry_recorded = True

        release_task: asyncio.Task[None] | None = None
        if lease is not None:
            # Release off the client-facing path: the terminal SSE frames
            # below must not be gated on the DB write.
            release_task = release_lease_background(lease, "completed")
            lease = None

        state.apply_allowed_tools(allowed)
        state.finish_reasoning()
        state.finish_calls()
        state.finish_message()

        # Drain the final close-out events (arguments done, item done, etc.)
        for frame in seq.drain_frames():
            yield frame

        if finish_reason == "length":
            response.status = "incomplete"
            response.incomplete_details = IncompleteDetails(reason="max_output_tokens")
        else:
            response.status = "completed"
        response.completed_at = int(time.time())
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
            # Keep the release-before-persist ordering without gating the
            # client-visible frames on the DB write.
            if release_task is not None:
                await release_task
            await asyncio.to_thread(
                _persist_response,
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

    except GeneratorExit:
        logger.info(
            "responses_stream_close request_id=%s response_id=%s server_id=%s outcome=cancelled close_reason=downstream_generator_closed upstream_status=%s upstream_opened=%s upstream_lines=%d emitted_frames=%d duration_ms=%.1f",
            request_id,
            response_id,
            server_id,
            upstream_status,
            upstream_opened,
            upstream_lines,
            emitted_frames,
            (time.monotonic() - started) * 1000.0,
        )
        if not telemetry_recorded:
            record_request_telemetry(
                sample_server,
                None,
                None,
                latency_ms=(time.monotonic() - started) * 1000.0,
                status="cancelled",
            )
        raise
    except asyncio.CancelledError as exc:
        logger.info(
            "responses_stream_close request_id=%s response_id=%s server_id=%s outcome=cancelled close_reason=downstream_disconnect_or_task_cancel exception=%s upstream_status=%s upstream_opened=%s upstream_lines=%d emitted_frames=%d duration_ms=%.1f",
            request_id,
            response_id,
            server_id,
            type(exc).__name__,
            upstream_status,
            upstream_opened,
            upstream_lines,
            emitted_frames,
            (time.monotonic() - started) * 1000.0,
        )
        if not telemetry_recorded:
            record_request_telemetry(
                sample_server,
                None,
                None,
                latency_ms=(time.monotonic() - started) * 1000.0,
                status="cancelled",
            )
        raise
    except Exception as e:
        logger.warning(
            "responses_stream_close request_id=%s response_id=%s server_id=%s outcome=error close_reason=upstream_or_proxy_error error_type=%s error=%s upstream_status=%s upstream_opened=%s upstream_lines=%d emitted_frames=%d duration_ms=%.1f",
            request_id,
            response_id,
            server_id,
            type(e).__name__,
            str(e),
            upstream_status,
            upstream_opened,
            upstream_lines,
            emitted_frames,
            (time.monotonic() - started) * 1000.0,
        )
        if not telemetry_recorded:
            record_request_telemetry(
                sample_server,
                None,
                None,
                latency_ms=(time.monotonic() - started) * 1000.0,
                status="error",
            )
        logger.error(f"Responses streaming error: {e}")
        response.status = "failed"
        response.error = Error(code="server_error", message=str(e))
        ev.response_failed(seq, response)
        for frame in seq.drain_frames():
            yield frame

        if persist:
            await asyncio.to_thread(
                _persist_response,
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

    finally:
        if lease is not None:
            await lease.release(
                "failed" if response.status == "failed" else "cancelled"
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


async def resolve_target(
    model_ref: str,
) -> tuple[ServerInstance, Model | None, uuid.UUID | None]:
    """Resolve a server alias / model name / model id to a ready server.

    Aliases route directly (auto-starting stopped/errored instances);
    names/ids resolve to a model and a server is found or started. Raises
    TargetError with spec error codes on failure.
    """
    from app.api.routes.v1.v1_chat_completions import _get_or_create_server

    with Session(engine) as session:
        try:
            target = resolve_inference_target(session, model_ref)
        except (LookupError, ValueError) as e:
            raise TargetError(404, "model_not_found", str(e)) from e

    if target.server is not None:
        return target.server, target.model, target.preferred_server_id

    try:
        server = await _get_or_create_server(target.model, None, start=False)
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
    return server, target.model, target.preferred_server_id


@router.post("/responses", response_model=None)
async def create_response(
    request: CreateResponseBody,
    db: Session = Depends(get_db),
    http_request: Request = None,
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

    # Materialize all chain data before the first async wait. Closing the
    # dependency session and then reading the chain reopens an idle transaction
    # that can hold server-row locks for the entire inference request.
    history: list[dict[str, Any]] = _build_chain_history(db, previous)
    chain_depth = _chain_depth(db, previous)
    # Target resolution may start a server and wait on the agent.
    db.close()
    try:
        server, model, lease_preference = await resolve_target(request.model)
    except TargetError as e:
        return _error_response(
            e.status,
            "not_found" if e.status == 404 else "too_many_requests",
            e.code,
            e.message,
            param="model",
        )
    if request.tools and metadata_capability(server, "tools") is False:
        return _error_response(
            400,
            "invalid_request",
            "unsupported_capability",
            f"Model '{request.model}' does not support tools",
            param="tools",
        )
    requested_effort = request.reasoning.effort if request.reasoning else None
    try:
        reasoning_effort = resolve_reasoning_effort(server, requested_effort)
    except ReasoningPolicyError as e:
        return _error_response(
            400,
            "invalid_request",
            "unsupported_capability",
            str(e),
            param="reasoning.effort",
        )

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
        chain_depth,
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
    created_at_mono = time.monotonic()

    # Admission can wait for another request or a server slot. Release the
    # dependency session before entering that potentially long async wait.
    db.close()
    try:
        lease = await inference_scheduler.acquire(
            model.id,
            response_id,
            preferred_server_id=lease_preference,
            is_cancelled=http_request.is_disconnected if http_request else None,
        )
        server = lease.server
    except TimeoutError as e:
        return _error_response(503, "too_many_requests", "no_slot", str(e))

    # The lease may land on a different instance of the same model; apply
    # that instance's reasoning policy to the upstream payload.
    try:
        reasoning_effort = resolve_reasoning_effort(
            server, request.reasoning.effort if request.reasoning else None
        )
    except ReasoningPolicyError as e:
        await lease.release("failed")
        return _error_response(
            400,
            "invalid_request",
            "unsupported_capability",
            str(e),
            param="reasoning.effort",
        )

    if request.stream:
        # The synchronous dependency session is no longer needed once the
        # response has been resolved. Close it before handing control to the
        # long-lived SSE generator so it cannot hold a DB transaction open.
        db.close()
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
                lease=lease,
                reasoning_effort=reasoning_effort,
                started_mono=created_at_mono,
            ),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    terminal_outcome = "completed"
    try:
        result = await _complete(
            request,
            str(server.id),
            str(server.agent_id),
            history,
            response_id,
            created_at,
            lease,
            reasoning_effort=reasoning_effort,
            started_mono=created_at_mono,
        )
    except asyncio.CancelledError:
        terminal_outcome = "cancelled"
        record_request_telemetry(
            lease.server,
            None,
            None,
            latency_ms=(time.monotonic() - created_at_mono) * 1000.0,
            status="cancelled",
        )
        raise
    except TranslationError as e:
        terminal_outcome = "failed"
        record_request_telemetry(
            lease.server,
            None,
            None,
            latency_ms=(time.monotonic() - created_at_mono) * 1000.0,
            status="error",
        )
        return _error_response(400, "invalid_request", "invalid_input", str(e))
    except Exception as e:
        terminal_outcome = "failed"
        record_request_telemetry(
            lease.server,
            None,
            None,
            latency_ms=(time.monotonic() - created_at_mono) * 1000.0,
            status="error",
        )
        logger.error(f"Responses completion error: {e}")
        return _error_response(500, "model_error", "model_error", str(e))
    finally:
        await lease.release(terminal_outcome)

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

    # Spec serialization: concrete echo params, nullable keys present as
    # null. Return JSONResponse so FastAPI's null-filling encoder can't
    # re-break the shape.
    return JSONResponse(content=serialize_spec(result))


# ============================================================================
# Compaction endpoint (spec 2026-04-24)
# ============================================================================


@router.post("/responses/compact", response_model=None)
async def compact_response(
    request: CompactRequestBody,
    db: Session = Depends(get_db),
    http_request: Request = None,
) -> CompactResource | JSONResponse:
    """Compact a conversation into a portable compaction item.

    Runs a real summarization pass through the same llama.cpp proxy the
    Responses API uses; the summary becomes the compaction item's
    encrypted_content (opaque to clients, round-trippable as input).
    Stateless: results are not persisted; chain from them by passing the
    compacted output as input on a fresh response.
    """
    # Server resolution may wait on an agent; do not hold the dependency
    # session across that await.
    db.close()
    try:
        # Resolve server exactly like POST /responses (alias auto-start
        # or model name/id fallback).
        instance = await find_alias_instance(request.model)
        server = None
        if instance is not None:
            server = instance
        else:
            model = _resolve_model(db, request.model)
            from app.api.routes.v1.v1_chat_completions import _get_or_create_server

            server = await _get_or_create_server(model, None, start=False)
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

    db.close()
    agent = await agent_manager.get_agent(str(server.agent_id))
    if not agent:
        return _error_response(
            503, "server_error", "no_agent_available", "Agent not found"
        )

    lease = None
    terminal_outcome = "failed"
    try:
        lease = await inference_scheduler.acquire(
            server.model_id,
            new_id("compact"),
            preferred_server_id=instance.id if instance is not None else None,
            is_cancelled=http_request.is_disconnected if http_request else None,
        )
        server = lease.server
        agent = await agent_manager.get_agent(str(server.agent_id))
        if not agent:
            return _error_response(
                503, "server_error", "no_agent_available", "Agent not found"
            )
        response = await lease.guard(
            agent_manager.send_to_agent(
                str(server.agent_id),
                "POST",
                f"/proxy/{server.id}/v1/chat/completions",
                llama_payload,
                timeout=1800.0,
                headers={
                    **lease.dispatch_headers(),
                },
            )
        )
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        summary_text = message.get("content") or ""
        usage_data = response.get("usage") or {}
        terminal_outcome = "completed"
    except asyncio.CancelledError:
        terminal_outcome = "cancelled"
        raise
    except TimeoutError as e:
        return _error_response(503, "too_many_requests", "no_slot", str(e))
    except Exception as e:
        logger.error(f"Compaction sampling error: {e}")
        return _error_response(500, "model_error", "model_error", str(e))
    finally:
        if lease is not None:
            await lease.release(terminal_outcome)

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
    return JSONResponse(content=serialize_spec(resource))

"""V1 Chat Completions endpoint - OpenAI-compatible chat API with Agent support."""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlmodel import Session, select

from app.api.deps import get_db
from app.core.db import engine
from app.models import InferenceLease, Model, ServerInstance
from app.services.agent_manager import agent_manager
from app.services.http_client import get_http_client
from app.services.inference_scheduler import (
    InferenceLeaseHandle,
    inference_scheduler,
)
from app.services.inference_stream import lines_with_keepalive, upstream_with_keepalive
from app.services.inference_target import resolve_inference_target
from app.services.reasoning_metadata import (
    ReasoningPolicyError,
    resolve_reasoning_effort,
)
from app.services.server_startup import (
    ServerStartupError,
    ensure_server_ready,
    metadata_capability,
)
from app.services.token_stats import record_request_telemetry

logger = logging.getLogger(__name__)

router = APIRouter()

# Strong references to detached lease-release tasks so they are not garbage
# collected before completing. Kept at module scope because the owning stream
# generator may already be gone (cancelled/closed) when the release finishes.
_release_tasks: set[asyncio.Task[None]] = set()


def _release_lease_background(
    lease: InferenceLeaseHandle, outcome: str = "completed"
) -> None:
    """Finalize a lease off the client-facing stream path.

    The client must never wait on the DB write that releases the inference
    slot. Under concurrent load that write can stall on the async connection
    pool (``DB_POOL_TIMEOUT``), which would delay the terminal ``[DONE]``
    marker and look exactly like an inference stall. The upstream slot is
    already free the moment its stream closes, so we hand the release to an
    independent task that survives the generator being cancelled.
    """

    async def _run() -> None:
        try:
            await asyncio.shield(lease.release(outcome))
        except Exception:
            logger.exception(
                "background_lease_release_failed request_id=%s",
                lease.request_id,
            )

    task = asyncio.get_running_loop().create_task(_run())
    _release_tasks.add(task)
    task.add_done_callback(_release_tasks.discard)


class ChatMessage(BaseModel):
    """Chat message with OpenAI-compatible text content."""

    role: Literal["system", "user", "assistant", "developer", "tool"]
    content: str | list[dict[str, Any]] | None = None
    # Thinking models: raw reasoning text (assistant responses only)
    reasoning_content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def llama_content(self) -> str:
        """Flatten OpenAI text parts into the string llama.cpp expects."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "".join(
            part.get("text", "")
            for part in self.content
            if part.get("type") == "text" and isinstance(part.get("text"), str)
        )


class ToolFunction(BaseModel):
    """OpenAI tool function definition."""

    name: str
    description: str = ""
    # JSON Schema for the function parameters (arbitrary structure)
    parameters: dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel):
    """OpenAI tool definition ({"type": "function", "function": {...}})."""

    type: str = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""

    model: str
    agent_id: str | None = None  # Optional: specify which agent to use
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    # OpenAI tools shape: [{"type": "function", "function": {name, description, parameters}}]
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] | None = None
    prompt_cache_options: dict[str, Any] | None = None
    # OpenAI-compatible: {"include_usage": true} adds a final usage chunk
    stream_options: StreamOptions | None = None


class ChatChoice(BaseModel):
    """Chat completion choice."""

    index: int
    message: ChatMessage
    finish_reason: str | None = None


class UsageInfo(BaseModel):
    """Usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: dict[str, int] | None = None
    completion_tokens_details: dict[str, Any] | None = None

    @classmethod
    def build(cls, prompt_tokens: int, completion_tokens: int) -> UsageInfo:
        """Build with OpenAI-style detail sub-objects (zeros)."""
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details={"cached_tokens": 0, "audio_tokens": 0},
            completion_tokens_details={
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        )


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo | None = None


class StreamChoice(BaseModel):
    """Streaming choice."""

    index: int
    delta: dict[str, Any]
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """Chat completion chunk for streaming."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]
    # Populated only on the final usage chunk (choices: []); null otherwise
    usage: UsageInfo | None = None


class StreamOptions(BaseModel):
    """OpenAI stream_options (include_usage emits a final usage chunk)."""

    include_usage: bool = True


def _convert_messages_to_llama_format(
    messages: list[ChatMessage],
) -> list[dict[str, Any]]:
    """Convert messages to llama.cpp chat format.

    Keep assistant tool calls and tool results in the native OpenAI shape.
    llama.cpp's Jinja chat templates use these fields to reconstruct the
    tool-call turn; flattening them into prose can make the model emit the
    marker (for example, ``[Called tools: ...]``) as its next response.
    """
    converted: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.llama_content()
        if msg.role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "content": content,
                    **({"tool_call_id": msg.tool_call_id} if msg.tool_call_id else {}),
                    **({"name": msg.name} if msg.name else {}),
                }
            )
        elif msg.role == "assistant" and msg.tool_calls:
            converted.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": msg.tool_calls,
                }
            )
        else:
            converted.append({"role": msg.role, "content": content})
    return converted


def _extract_delta_content(chunk_data: dict) -> str:
    """Extract content from a streamed chunk (OpenAI format or llama.cpp native)."""
    choices = chunk_data.get("choices")
    if choices and isinstance(choices, list):
        choice = choices[0]
        delta = choice.get("delta") or {}
        if "content" in delta:
            return delta.get("content") or ""
        if "text" in choice:
            return choice.get("text") or ""
        if "content" in choice:
            return choice.get("content") or ""
    return chunk_data.get("content", "") or ""


def _extract_finish_reason(chunk_data: dict[str, Any]) -> str | None:
    """finish_reason from an OpenAI-style chunk (top-level choice key).

    llama.cpp native /completion streams put it inside the choice object;
    the chat/completions OpenAI endpoint puts it on the choice too. Reading
    it from the delta (previous bug) meant tool-call finishes were dropped.
    """
    choices = chunk_data.get("choices")
    if choices and isinstance(choices, list):
        reason: Any = choices[0].get("finish_reason")
        return str(reason) if reason is not None else None
    return None


def _extract_tool_call_deltas(chunk_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Tool-call deltas from an OpenAI-format stream chunk (pass-through)."""
    choices = chunk_data.get("choices")
    if choices and isinstance(choices, list):
        delta: Any = choices[0].get("delta") or {}
        tcs: Any = delta.get("tool_calls") if isinstance(delta, dict) else None
        if isinstance(tcs, list):
            return tcs
    return []


def _extract_delta_reasoning(chunk_data: dict) -> str:
    """Extract reasoning delta from a streamed chunk (thinking models)."""
    choices = chunk_data.get("choices")
    if choices and isinstance(choices, list):
        delta = choices[0].get("delta") or {}
        return delta.get("reasoning_content") or ""
    return ""


def _build_tools_payload(request: ChatCompletionRequest) -> dict[str, Any]:
    """Serialize OpenAI tool fields for the llama.cpp payload."""
    extra: dict[str, Any] = {}
    if request.tools:
        extra["tools"] = [tool.model_dump(exclude_none=True) for tool in request.tools]
    if request.tool_choice is not None:
        extra["tool_choice"] = request.tool_choice
    return extra


def _extract_usage(chunk_data: dict) -> UsageInfo | None:
    """Extract usage from a streamed chunk if the server provides one.

    Two shapes recognized:
    - OpenAI final usage chunk: top-level "usage" object
    - llama.cpp native timings chunk: tokens_evaluated/tokens_predicted
      at the top level
    """
    usage = chunk_data.get("usage")
    if not isinstance(usage, dict) or not usage:
        # llama.cpp native /completion stream (timings chunk)
        if "tokens_evaluated" in chunk_data or "tokens_predicted" in chunk_data:
            usage = {
                "prompt_tokens": chunk_data.get("tokens_evaluated", 0),
                "completion_tokens": chunk_data.get("tokens_predicted", 0),
            }
        else:
            return None

    # llama.cpp native streaming (timings) carries token counts at top level
    prompt_tokens = int(
        usage.get("prompt_tokens", chunk_data.get("tokens_evaluated", 0)) or 0
    )
    completion_tokens = int(
        usage.get("completion_tokens", chunk_data.get("tokens_predicted", 0)) or 0
    )
    total = usage.get("total_tokens")
    total_tokens = (
        int(total) if total is not None else prompt_tokens + completion_tokens
    )

    timings = chunk_data.get("timings") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_value = (
        prompt_details["cached_tokens"]
        if "cached_tokens" in prompt_details
        else timings.get("cache_n", 0)
    )
    cached_tokens = int(cached_value or 0)
    prompt_details = {
        **prompt_details,
        "cached_tokens": cached_tokens,
    }
    prompt_details.setdefault("audio_tokens", 0)
    completion_details = usage.get("completion_tokens_details") or {
        "reasoning_tokens": 0,
        "audio_tokens": 0,
        "accepted_prediction_tokens": 0,
        "rejected_prediction_tokens": 0,
    }

    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=prompt_details,
        completion_tokens_details=completion_details,
    )


async def _stream_completion_via_agent(
    agent_id: str,
    server: ServerInstance,
    request: ChatCompletionRequest,
    request_id: str,
    lease: InferenceLeaseHandle,
) -> AsyncGenerator[str]:
    """Stream completion via Agent proxy."""
    messages = _convert_messages_to_llama_format(request.messages)

    payload = {
        "messages": messages,
        "stream": True,
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens

    # Add optional parameters
    for key in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
        value = getattr(request, key)
        if value is not None:
            payload[key] = value
    if request.reasoning_effort is not None:
        payload["reasoning_effort"] = request.reasoning_effort

    payload.update(_build_tools_payload(request))
    # Ask the upstream server for a final usage chunk; llama.cpp complies
    # and sends it with empty choices.
    payload["stream_options"] = {"include_usage": True}

    created = int(time.time())
    started = time.monotonic()
    include_usage = not (
        request.stream_options and not request.stream_options.include_usage
    )
    # Fallback accounting when the server does not send usage chunks
    fallback_completion_chars = 0
    final_usage: UsageInfo | None = None
    final_timings: dict = {}
    telemetry_recorded = False
    lease_released = False
    terminal_outcome = "completed"
    upstream_status: int | None = None
    upstream_opened = False
    upstream_lines = 0
    emitted_chunks = 0
    done_marker_seen = False
    waiting = None

    try:
        # Hold the client connection while a cold start completes so the
        # first token arrives as soon as the server is healthy.
        server = await ensure_server_ready(server)

        # Get agent
        agent = await agent_manager.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        # Stream through the acknowledged agent reservation.
        lease.mark_upstream_started()
        client = get_http_client()
        proxy_url = (
            f"http://{agent.host}:{agent.port}/proxy/{server.id}/v1/chat/completions"
        )

        waiting = upstream_with_keepalive(client, proxy_url, payload, lease)
        async for upstream in waiting:
            if upstream is None:
                yield ": keep-alive\n\n"
                continue
            response, deadline = upstream
            upstream_opened = True
            upstream_status = response.status_code
            logger.info(
                "backend_inference_stream_open request_id=%s server_id=%s agent_id=%s upstream_status=%s",
                lease.request_id,
                server.id,
                agent_id,
                upstream_status,
            )
            response.raise_for_status()

            async for line in lines_with_keepalive(
                response.aiter_lines(), lease, deadline=deadline
            ):
                if line is None:
                    yield ": keep-alive\n\n"
                    continue
                upstream_lines += 1
                if line.startswith("data:"):
                    data = line[5:].lstrip()
                    if data.strip() == "[DONE]":
                        done_marker_seen = True
                        break

                    try:
                        chunk_data = json.loads(data)
                        if "error" in chunk_data:
                            raise RuntimeError(f"LLM error: {chunk_data['error']}")

                        # Server-provided usage (final chunk) wins
                        chunk_usage = _extract_usage(chunk_data)
                        if chunk_usage is not None:
                            final_usage = chunk_usage
                            final_timings = chunk_data.get("timings") or {}

                        delta_content = _extract_delta_content(chunk_data)
                        delta_reasoning = _extract_delta_reasoning(chunk_data)
                        fallback_completion_chars += len(delta_content)
                        tool_call_deltas = _extract_tool_call_deltas(chunk_data)
                        # OpenAI chunk shape: delta carries only the
                        # fields present in this fragment (content is
                        # omitted entirely on tool-call-only chunks).
                        delta_out: dict[str, Any] = {}
                        if delta_content or not tool_call_deltas:
                            delta_out["content"] = delta_content
                        if delta_reasoning:
                            delta_out["reasoning_content"] = delta_reasoning
                        if tool_call_deltas:
                            delta_out["tool_calls"] = tool_call_deltas

                        stream_chunk = ChatCompletionChunk(
                            id=request_id,
                            created=created,
                            model=request.model,
                            choices=[
                                StreamChoice(
                                    index=0,
                                    delta=delta_out,
                                    finish_reason=_extract_finish_reason(chunk_data),
                                )
                            ],
                            usage=None,
                        )

                        emitted_chunks += 1
                        yield f"data: {stream_chunk.model_dump_json()}\n\n"

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in stream: {data}")
                        continue

        if not done_marker_seen:
            raise RuntimeError("LLM stream ended without a completion marker")
        if final_usage is None:
            # Fallback: estimate tokens from content length.
            # ~4 chars/token heuristic for prompt; completion
            # counts actual streamed characters.
            prompt_chars = sum(len(m["content"]) for m in messages)
            final_usage = UsageInfo.build(
                prompt_tokens=max(prompt_chars // 4, 0),
                completion_tokens=fallback_completion_chars // 4,
            )
        logger.info(
            "backend_inference_stream_close request_id=%s server_id=%s agent_id=%s outcome=completed close_reason=upstream_eof upstream_status=%s upstream_opened=%s upstream_lines=%d emitted_chunks=%d duration_ms=%.1f",
            lease.request_id,
            server.id,
            agent_id,
            upstream_status,
            upstream_opened,
            upstream_lines,
            emitted_chunks,
            (time.monotonic() - started) * 1000.0,
        )
        # Record before yielding: latency is upstream production time,
        # not downstream consumption time.
        record_request_telemetry(
            lease.server,
            final_usage.model_dump(),
            final_timings,
            latency_ms=(time.monotonic() - started) * 1000.0,
            status="success",
        )
        telemetry_recorded = True
        # The upstream slot is free the moment its response stream closes.
        # Release it in the background so the client-facing terminal marker
        # below is never gated on the DB write, which can stall on the async
        # connection pool under concurrent load and look like an inference
        # stall. Mark released now so the finally block does not double-release.
        _release_lease_background(lease, "completed")
        lease_released = True

        if include_usage:
            final_chunk = ChatCompletionChunk(
                id=request_id,
                created=created,
                model=request.model,
                choices=[],
                usage=final_usage,
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    except GeneratorExit:
        terminal_outcome = "cancelled"
        logger.info(
            "backend_inference_stream_close request_id=%s server_id=%s agent_id=%s outcome=cancelled close_reason=downstream_generator_closed upstream_status=%s upstream_opened=%s upstream_lines=%d emitted_chunks=%d duration_ms=%.1f",
            lease.request_id,
            server.id,
            agent_id,
            upstream_status,
            upstream_opened,
            upstream_lines,
            emitted_chunks,
            (time.monotonic() - started) * 1000.0,
        )
        if not telemetry_recorded:
            record_request_telemetry(
                lease.server,
                None,
                None,
                latency_ms=(time.monotonic() - started) * 1000.0,
                status="cancelled",
            )
        raise
    except asyncio.CancelledError as exc:
        terminal_outcome = "cancelled"
        logger.info(
            "backend_inference_stream_close request_id=%s server_id=%s agent_id=%s outcome=cancelled close_reason=downstream_disconnect_or_task_cancel exception=%s upstream_status=%s upstream_opened=%s upstream_lines=%d emitted_chunks=%d duration_ms=%.1f",
            lease.request_id,
            server.id,
            agent_id,
            type(exc).__name__,
            upstream_status,
            upstream_opened,
            upstream_lines,
            emitted_chunks,
            (time.monotonic() - started) * 1000.0,
        )
        if not telemetry_recorded:
            record_request_telemetry(
                lease.server,
                None,
                None,
                latency_ms=(time.monotonic() - started) * 1000.0,
                status="cancelled",
            )
        raise
    except Exception as e:
        terminal_outcome = "failed"
        if waiting is not None:
            await waiting.aclose()
        logger.warning(
            "backend_inference_stream_close request_id=%s server_id=%s agent_id=%s outcome=error close_reason=upstream_or_proxy_error error_type=%s error=%s upstream_status=%s upstream_opened=%s upstream_lines=%d emitted_chunks=%d duration_ms=%.1f",
            lease.request_id,
            server.id,
            agent_id,
            type(e).__name__,
            str(e),
            upstream_status,
            upstream_opened,
            upstream_lines,
            emitted_chunks,
            (time.monotonic() - started) * 1000.0,
        )
        if not telemetry_recorded:
            record_request_telemetry(
                lease.server,
                None,
                None,
                latency_ms=(time.monotonic() - started) * 1000.0,
                status="error",
            )
        logger.error(f"Streaming error: {e}")
        error_chunk = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
    finally:
        if waiting is not None:
            await waiting.aclose()
        if not lease_released:
            await lease.release(terminal_outcome)


def _find_existing_server(
    model_id: Any, agent_id: str | None = None
) -> ServerInstance | None:
    """Find a reusable server instance for the model."""
    with Session(engine) as session:
        stmt = select(ServerInstance).where(
            ServerInstance.model_id == model_id,
            ServerInstance.status.in_(["running", "stopped", "starting"]),
        )
        if agent_id:
            stmt = stmt.where(ServerInstance.agent_id == agent_id)
        return session.exec(stmt).first()


def _find_starting_server(
    model_id: Any, agent_id: str | None = None
) -> ServerInstance | None:
    """Find an in-flight (starting) instance so parallel requests share one startup."""
    with Session(engine) as session:
        stmt = select(ServerInstance).where(
            ServerInstance.model_id == model_id,
            ServerInstance.status == "starting",
        )
        if agent_id:
            stmt = stmt.where(ServerInstance.agent_id == agent_id)
        return session.exec(stmt).first()


async def _get_or_create_server(
    model: Model, agent_id: str | None = None, *, start: bool = True
) -> ServerInstance:
    """Get existing server or create new one via Agent.

    Deduplication: if another request already has a server starting for
    this model, return that instance so parallel cold-starts share one
    llama-server instead of racing to create duplicates.
    """
    server = _find_existing_server(model.id, agent_id)

    if server:
        logger.info(f"Using existing server {server.id}")
        return server

    starting = _find_starting_server(model.id, agent_id)
    if starting:
        logger.info(f"Joining in-flight startup for server {starting.id}")
        return starting

    # Need to create a new server row. The scheduler may defer the actual
    # process start until this request reaches the FIFO queue head.
    logger.info("No existing server found, creating server row")

    # Find an agent
    if agent_id:
        agent = await agent_manager.get_agent(agent_id)
        if not agent:
            raise HTTPException(404, f"Agent {agent_id} not found")
    else:
        # Find first online agent
        agents = await agent_manager.list_agents()
        if not agents:
            raise HTTPException(503, "No agents available")
        agent = await agent_manager.get_agent(agents[0]["id"])

    if not agent or agent.status != "online":
        raise HTTPException(503, "No online agents available")

    # Create the row before dispatching so a fast healthy start cannot lose
    # its server.started event before the backend has a matching instance.
    server_id = str(uuid.uuid4())
    with Session(engine) as session:
        session.execute(
            text("SELECT pg_advisory_xact_lock(:key)"),
            {"key": model.id.int & ((1 << 63) - 1)},
        )
        statement = select(ServerInstance).where(
            ServerInstance.model_id == model.id,
            ServerInstance.status.in_(["running", "stopped", "starting"]),
        )
        if agent_id:
            statement = statement.where(ServerInstance.agent_id == agent.id)
        existing = session.exec(statement).first()
        if existing is not None:
            return existing
        server = ServerInstance(
            id=uuid.UUID(server_id),
            model_id=model.id,
            agent_id=agent.id,
            alias=f"auto-{model.id}",
            process_command=model.source_file or model.path,
            gpu_layers=35,
            context_size=4096,
            flash_attn=True,
            config={},
            status="stopped",
            inactivity_timeout_seconds=300,
        )
        session.add(server)
        session.commit()
        session.refresh(server)

    if not start:
        return server
    return await ensure_server_ready(server)


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
    db: Session = Depends(get_db),
    http_request: Request = None,
) -> ChatCompletionResponse | StreamingResponse:
    """Create chat completion via Agent proxy.

    Cold starts are handled transparently: a stopped/errored server is
    (re)started and the request waits for it to become healthy before the
    completion is generated, so the client just sees a longer first-token
    latency.
    """
    client_request_id = (
        http_request.headers.get("X-Inference-Request-ID") if http_request else None
    )
    try:
        request_id = (
            f"chatcmpl-{uuid.UUID(client_request_id.removeprefix('chatcmpl-'))}"
            if client_request_id and client_request_id.startswith("chatcmpl-")
            else f"chatcmpl-{uuid.uuid4()}"
        )
    except ValueError:
        raise HTTPException(400, "Invalid inference request ID") from None
    if (
        client_request_id
        and db.exec(
            select(InferenceLease.id).where(InferenceLease.request_id == request_id)
        ).first()
    ):
        raise HTTPException(409, "Inference request ID already exists")
    created = int(time.time())
    started = time.monotonic()
    required_agent_id = uuid.UUID(request.agent_id) if request.agent_id else None

    # Debug visibility: what the client actually sent (tools present? which?)
    logger.info(
        "chat.completions: model=%s stream=%s messages=%d "
        "tools=%s tool_choice=%s roles=%s",
        request.model,
        request.stream,
        len(request.messages),
        [t.function.name for t in request.tools] if request.tools else "none",
        request.tool_choice,
        [m.role for m in request.messages],
    )
    logger.debug("chat.completions request: %s", request.model_dump_json())

    # Resolve model field: a server alias (public name from /v1/models)
    # routes directly to that server (auto-starting stopped instances);
    # otherwise it is a model name/id and a server is found or started.
    server: ServerInstance | None = None
    try:
        target = resolve_inference_target(db, request.model, request.agent_id)
    except (LookupError, ValueError) as e:
        raise HTTPException(404, str(e)) from e
    model = target.model
    instance = target.server
    db.close()

    if instance is not None:
        if request.tools and metadata_capability(instance, "tools") is False:
            raise HTTPException(400, f"Model '{request.model}' does not support tools")
        server = instance
    else:
        # Prefer an already-running instance so model-name requests can be
        # load-balanced. Cold-start only when no compatible instance exists.
        try:
            candidates = await inference_scheduler.candidates_for_model(
                model.id, required_agent_id=required_agent_id
            )
            if candidates:
                lease = await inference_scheduler.acquire(
                    model.id,
                    request_id,
                    required_agent_id=required_agent_id,
                    is_cancelled=http_request.is_disconnected if http_request else None,
                )
                server = lease.server
            else:
                server = await _get_or_create_server(
                    model, request.agent_id, start=False
                )
                lease = await inference_scheduler.acquire(
                    model.id,
                    request_id,
                    preferred_server_id=server.id,
                    required_agent_id=target.required_agent_id,
                    is_cancelled=http_request.is_disconnected if http_request else None,
                )
        except HTTPException:
            raise
        except ServerStartupError as e:
            raise HTTPException(503, f"Server not available: {e}")
        except Exception as e:
            logger.error(f"Server creation failed: {e}")
            raise HTTPException(503, f"Failed to start server: {e}")

    if server is None:
        raise HTTPException(503, "No inference server available")

    # Alias requests and model-name requests both hold a lease for the whole
    # upstream call; the streaming generator releases it on disconnect/end.
    if instance is not None:
        lease = await inference_scheduler.acquire(
            model.id,
            request_id,
            preferred_server_id=target.preferred_server_id,
            required_agent_id=target.required_agent_id,
            is_cancelled=http_request.is_disconnected if http_request else None,
        )

    # Enforce the reasoning capability metadata of the instance that will
    # actually serve the request.
    try:
        reasoning_effort = resolve_reasoning_effort(
            lease.server, request.reasoning_effort
        )
    except ReasoningPolicyError as e:
        await lease.release("failed")
        raise HTTPException(400, str(e)) from e
    request.reasoning_effort = reasoning_effort

    if request.stream:
        # Stream response. The generator awaits readiness itself, so the
        # client connection is held during a cold start (invisible retry).
        db.close()
        return StreamingResponse(
            _stream_completion_via_agent(
                str(server.agent_id),
                server,
                request,
                request_id,
                lease,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming response
    terminal_outcome = "completed"
    try:
        agent = await agent_manager.get_agent(str(server.agent_id))
        if not agent:
            raise ValueError("Agent not found")

        non_stream_payload: dict[str, Any] = {
            "messages": _convert_messages_to_llama_format(request.messages),
            "stream": False,
        }
        if request.temperature is not None:
            non_stream_payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            non_stream_payload["max_tokens"] = request.max_tokens
        for key in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
            value = getattr(request, key)
            if value is not None:
                non_stream_payload[key] = value
        if request.reasoning_effort is not None:
            non_stream_payload["reasoning_effort"] = request.reasoning_effort
        non_stream_payload.update(_build_tools_payload(request))

        response = await lease.guard(
            agent_manager.send_to_agent(
                str(server.agent_id),
                "POST",
                f"/proxy/{server.id}/v1/chat/completions",
                non_stream_payload,
                timeout=1800.0,
                headers=lease.dispatch_headers(),
            )
        )

        # Usage: server-provided numbers win; fall back to estimating
        # prompt tokens from message length and completion from content.
        usage_data = response.get("usage") or {}
        message = response["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning_content = message.get("reasoning_content") or None
        tool_calls = message.get("tool_calls") or None
        prompt_tokens = (
            int(usage_data["prompt_tokens"])
            if usage_data.get("prompt_tokens") is not None
            else sum(len(m["content"]) for m in non_stream_payload["messages"]) // 4
        )
        completion_tokens = (
            int(usage_data["completion_tokens"])
            if usage_data.get("completion_tokens") is not None
            else len(content) // 4
        )
        total_tokens = (
            int(usage_data["total_tokens"])
            if usage_data.get("total_tokens") is not None
            else prompt_tokens + completion_tokens
        )
        prompt_details = usage_data.get("prompt_tokens_details") or {}
        cached_value = (
            prompt_details["cached_tokens"]
            if "cached_tokens" in prompt_details
            else (response.get("timings") or {}).get("cache_n", 0)
        )
        cached_tokens = int(cached_value or 0)
        prompt_details = {**prompt_details, "cached_tokens": cached_tokens}
        prompt_details.setdefault("audio_tokens", 0)
        completion_details = dict(usage_data.get("completion_tokens_details") or {})
        completion_details.setdefault("reasoning_tokens", 0)
        completion_details.setdefault("audio_tokens", 0)
        completion_details.setdefault("accepted_prediction_tokens", 0)
        completion_details.setdefault("rejected_prediction_tokens", 0)
        result = ChatCompletionResponse(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=content,
                        reasoning_content=reasoning_content,
                        tool_calls=tool_calls,
                    ),
                    finish_reason=response["choices"][0].get("finish_reason"),
                )
            ],
            usage=UsageInfo.build(prompt_tokens, completion_tokens).model_copy(
                update={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "prompt_tokens_details": prompt_details,
                    "completion_tokens_details": completion_details,
                }
            ),
        )
        # Record after the response object is built: a construction
        # failure must count as an error, not success + error.
        record_request_telemetry(
            lease.server,
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            },
            response.get("timings"),
            latency_ms=(time.monotonic() - started) * 1000.0,
            status="success",
        )
        return result

    except asyncio.CancelledError:
        terminal_outcome = "cancelled"
        record_request_telemetry(
            lease.server,
            None,
            None,
            latency_ms=(time.monotonic() - started) * 1000.0,
            status="cancelled",
        )
        raise
    except Exception as e:
        terminal_outcome = "failed"
        record_request_telemetry(
            lease.server,
            None,
            None,
            latency_ms=(time.monotonic() - started) * 1000.0,
            status="error",
        )
        logger.error(f"Completion error: {e}")
        raise HTTPException(500, f"Inference failed: {e}")
    finally:
        await lease.release(terminal_outcome)

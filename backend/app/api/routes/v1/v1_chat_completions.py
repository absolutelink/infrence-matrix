"""V1 Chat Completions endpoint - OpenAI-compatible chat API with Agent support."""

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.api.deps import get_db
from app.core.db import engine
from app.models import Model, ServerInstance
from app.services.agent_manager import agent_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message (content may be null for assistant tool-call turns)."""

    role: Literal["system", "user", "assistant", "developer", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def llama_content(self) -> str:
        """Content as llama.cpp expects it (null -> empty string)."""
        return self.content if self.content is not None else ""


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
    prompt_cache_options: dict | None = None
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
    completion_tokens_details: dict[str, int] | None = None

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
    delta: dict[str, str]
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
) -> list[dict[str, str]]:
    """Convert messages to llama.cpp chat format.

    llama.cpp's OpenAI endpoint has limited tool-role support, so tool
    results are flattened into user messages and assistant tool_calls are
    described in text — the model still sees the full exchange.
    """
    converted: list[dict[str, str]] = []
    for msg in messages:
        content = msg.llama_content()
        if msg.role == "tool":
            label = msg.name or msg.tool_call_id or "tool"
            converted.append(
                {"role": "user", "content": f"[Tool result ({label})]: {content}"}
            )
        elif msg.role == "assistant" and msg.tool_calls:
            calls = ", ".join(
                c.get("function", {}).get("name", "unknown") for c in msg.tool_calls
            )
            converted.append(
                {
                    "role": "assistant",
                    "content": f"{content}\n[Called tools: {calls}]".strip(),
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

    prompt_details = usage.get("prompt_tokens_details") or {
        "cached_tokens": 0,
        "audio_tokens": 0,
    }
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
) -> AsyncGenerator[str]:
    """Stream completion via Agent proxy."""
    messages = _convert_messages_to_llama_format(request.messages)

    payload = {
        "messages": messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": True,
    }

    # Add optional parameters
    for key in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
        value = getattr(request, key)
        if value is not None:
            payload[key] = value

    payload.update(_build_tools_payload(request))

    created = int(time.time())
    include_usage = not (
        request.stream_options and not request.stream_options.include_usage
    )
    # Fallback accounting when the server does not send usage chunks
    fallback_completion_chars = 0
    final_usage: UsageInfo | None = None

    try:
        # Get agent
        agent = await agent_manager.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        # Stream through agent proxy
        async with httpx.AsyncClient(timeout=300.0) as client:
            proxy_url = f"http://{agent.host}:{agent.port}/proxy/{server.id}/v1/chat/completions"

            async with client.stream(
                "POST",
                proxy_url,
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data)

                            # Server-provided usage (final chunk) wins
                            chunk_usage = _extract_usage(chunk_data)
                            if chunk_usage is not None:
                                final_usage = chunk_usage

                            delta_content = _extract_delta_content(chunk_data)
                            fallback_completion_chars += len(delta_content)

                            stream_chunk = ChatCompletionChunk(
                                id=request_id,
                                created=created,
                                model=request.model,
                                choices=[
                                    StreamChoice(
                                        index=0,
                                        delta={"content": delta_content},
                                        finish_reason=chunk_data.get("finish_reason"),
                                    )
                                ],
                                usage=None,
                            )

                            yield f"data: {stream_chunk.model_dump_json()}\n\n"

                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in stream: {data}")
                            continue

                if include_usage:
                    usage = final_usage
                    if usage is None:
                        # Fallback: estimate tokens from content length.
                        # ~4 chars/token heuristic for prompt; completion
                        # counts actual streamed characters.
                        prompt_chars = sum(len(m["content"]) for m in messages)
                        usage = UsageInfo.build(
                            prompt_tokens=max(prompt_chars // 4, 0),
                            completion_tokens=fallback_completion_chars // 4,
                        )
                    final_chunk = ChatCompletionChunk(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[],
                        usage=usage,
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"

                yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"


def _find_existing_server(
    model_id: Any, agent_id: str | None = None
) -> ServerInstance | None:
    """Find a running server instance for the model (optionally on a specific agent)."""
    with Session(engine) as session:
        stmt = select(ServerInstance).where(
            ServerInstance.model_id == model_id,
            ServerInstance.status == "running",
        )
        if agent_id:
            stmt = stmt.where(ServerInstance.agent_id == agent_id)
        return session.exec(stmt).first()


async def _get_or_create_server(
    model: Model, agent_id: str | None = None
) -> ServerInstance:
    """Get existing server or create new one via Agent."""
    server = _find_existing_server(model.id, agent_id)

    if server:
        logger.info(f"Using existing server {server.id}")
        return server

    # Need to start a new server
    logger.info("No existing server found, starting new one")

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

    # Start server via agent. The agent allocates a random free port; we
    # generate the instance id here since the record is created after the
    # start call returns.
    server_id = str(uuid.uuid4())
    start_response = await agent_manager.send_to_agent(
        agent.id,
        "POST",
        "/servers/start",
        {
            "model_id": str(model.id),
            "model_path": model.path,
            "config": {
                "id": server_id,
                "gpu_layers": 35,
                "context_size": model.context_length or 4096,
                "batch_size": 512,
                "cache_prompt": True,
            },
            # If the model file is not on the agent yet, it downloads it
            # from the source repo before launching llama-server.
            "source": {
                "source": model.source,
                "repo_id": model.source_repo_id,
                "filename": model.source_file or (model.path.rsplit("/", 1)[-1]),
            }
            if model.source_repo_id
            else None,
        },
    )

    # Create ServerInstance record. The port is agent-allocated per start
    # and not persisted; stash it in the JSON config for display only.
    allocated_port = (
        start_response.get("port") if isinstance(start_response, dict) else None
    )
    with Session(engine) as session:
        server = ServerInstance(
            model_id=model.id,
            agent_id=agent.id,
            gpu_layers=35,
            context_size=model.context_length or 4096,
            flash_attn=True,
            config={"port": str(allocated_port)} if allocated_port else {},
            proxy_url=start_response.get("proxy_url"),
            status="starting",
            inactivity_timeout_seconds=300,
        )
        session.add(server)
        session.commit()
        session.refresh(server)

    logger.info(f"Started new server {server.id} on agent {agent.id}")
    return server


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
    db: Session = Depends(get_db),
) -> ChatCompletionResponse | StreamingResponse:
    """Create chat completion via Agent proxy."""
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    # Get model by name (OpenAI style) or id (UI legacy)
    model = db.exec(select(Model).where(Model.name == request.model)).first()
    if not model:
        try:
            model = db.exec(select(Model).where(Model.id == request.model)).first()
        except Exception:
            model = None
    if not model:
        raise HTTPException(404, f"Model {request.model} not found")

    # Get or create server
    try:
        server = await _get_or_create_server(model, request.agent_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Server creation failed: {e}")
        raise HTTPException(503, f"Failed to start server: {e}")

    if request.stream:
        # Stream response
        return StreamingResponse(
            _stream_completion_via_agent(
                str(server.agent_id),
                server,
                request,
                request_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming response
    try:
        agent = await agent_manager.get_agent(str(server.agent_id))
        if not agent:
            raise ValueError("Agent not found")

        non_stream_payload: dict[str, Any] = {
            "messages": _convert_messages_to_llama_format(request.messages),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        for key in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
            value = getattr(request, key)
            if value is not None:
                non_stream_payload[key] = value
        non_stream_payload.update(_build_tools_payload(request))

        response = await agent_manager.send_to_agent(
            str(server.agent_id),
            "POST",
            f"/proxy/{server.id}/v1/chat/completions",
            non_stream_payload,
            timeout=300.0,
        )

        # Usage: server-provided numbers win; fall back to estimating
        # prompt tokens from message length and completion from content.
        usage_data = response.get("usage") or {}
        content = response["choices"][0]["message"]["content"]
        prompt_tokens = (
            usage_data.get("prompt_tokens")
            or sum(len(m["content"]) for m in non_stream_payload["messages"]) // 4
        )
        completion_tokens = usage_data.get("completion_tokens") or len(content) // 4
        total_tokens = usage_data.get("total_tokens") or (
            prompt_tokens + completion_tokens
        )

        return ChatCompletionResponse(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=content,
                    ),
                    finish_reason=response["choices"][0].get("finish_reason"),
                )
            ],
            usage=UsageInfo.build(prompt_tokens, completion_tokens).model_copy(
                update={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
            ),
        )

    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(500, f"Inference failed: {e}")

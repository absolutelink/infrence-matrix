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
                            yield "data: [DONE]\n\n"
                            break

                        try:
                            chunk_data = json.loads(data)
                            delta_content = _extract_delta_content(chunk_data)

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
                            )

                            yield f"data: {stream_chunk.model_dump_json()}\n\n"

                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in stream: {data}")
                            continue

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

        return ChatCompletionResponse(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response["choices"][0]["message"]["content"],
                    ),
                    finish_reason=response["choices"][0].get("finish_reason"),
                )
            ],
            usage=UsageInfo(
                prompt_tokens=response.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=response.get("usage", {}).get("completion_tokens", 0),
                total_tokens=response.get("usage", {}).get("total_tokens", 0),
            )
            if response.get("usage")
            else None,
        )

    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(500, f"Inference failed: {e}")

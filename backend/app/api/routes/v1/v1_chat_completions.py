"""V1 Chat Completions endpoint - OpenAI-compatible chat API with Agent support."""

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.api.deps import get_db
from app.models import Model, ServerInstance, Agent
from app.services.agent_manager import agent_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message."""
    role: Literal["system", "user", "assistant", "developer"]
    content: str


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
    tools: list[dict[str, str]] | None = None
    tool_choice: str | dict[str, str] | None = None
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


def _convert_messages_to_llama_format(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Convert messages to llama.cpp format."""
    return [{"role": msg.role, "content": msg.content} for msg in messages]


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
                            delta = {"content": chunk_data.get("content", "")}
                            
                            stream_chunk = ChatCompletionChunk(
                                id=request_id,
                                created=created,
                                model=request.model,
                                choices=[StreamChoice(
                                    index=0,
                                    delta=delta,
                                    finish_reason=chunk_data.get("finish_reason"),
                                )],
                            )
                            
                            yield f"data: {stream_chunk.model_dump_json()}\n\n"
                            
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in stream: {data}")
                            continue
                            
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = {
            "error": {"message": str(e), "type": "server_error"}
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


async def _get_or_create_server(model: Model, agent_id: str | None = None) -> ServerInstance:
    """Get existing server or create new one via Agent."""
    async with Session() as session:
        # Try to find existing server
        if agent_id:
            # Look for server on specific agent
            stmt = select(ServerInstance).where(
                ServerInstance.model_id == model.id,
                ServerInstance.agent_id == agent_id,
                ServerInstance.status == "running",
            )
        else:
            # Find any running server for this model
            stmt = select(ServerInstance).where(
                ServerInstance.model_id == model.id,
                ServerInstance.status == "running",
            )
        
        result = session.exec(stmt)
        server = result.first()
        
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
        
        # Start server via agent
        start_response = await agent_manager.send_to_agent(
            agent.id,
            "POST",
            "/servers/start",
            {
                "model_id": str(model.id),
                "model_path": model.path,
                "config": {
                    "gpu_layers": 35,
                    "context_size": model.context_length or 4096,
                    "batch_size": 512,
                    "cache_prompt": True,
                }
            }
        )
        
        # Create ServerInstance record
        server = ServerInstance(
            model_id=model.id,
            agent_id=agent.id,
            port=8081,  # Default agent port
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
    
    # Get model
    model = db.exec(select(Model).where(Model.name == request.model)).first()
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
        )
    
    # Non-streaming response
    try:
        agent = await agent_manager.get_agent(str(server.agent_id))
        if not agent:
            raise ValueError("Agent not found")
        
        response = await agent_manager.send_to_agent(
            str(server.agent_id),
            "POST",
            f"/proxy/{server.id}/v1/chat/completions",
            {
                "messages": _convert_messages_to_llama_format(request.messages),
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": False,
            },
            timeout=300.0,
        )
        
        return ChatCompletionResponse(
            id=request_id,
            created=created,
            model=request.model,
            choices=[ChatChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=response["choices"][0]["message"]["content"],
                ),
                finish_reason=response["choices"][0].get("finish_reason"),
            )],
            usage=UsageInfo(
                prompt_tokens=response.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=response.get("usage", {}).get("completion_tokens", 0),
                total_tokens=response.get("usage", {}).get("total_tokens", 0),
            ) if response.get("usage") else None,
        )
        
    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(500, f"Inference failed: {e}")

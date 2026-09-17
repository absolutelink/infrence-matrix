"""V1 Chat Completions endpoint - OpenAI-compatible chat API."""

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.api.deps import get_db
from app.models import Model, ServerInstance

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message."""
    role: Literal["system", "user", "assistant", "developer"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""
    model: str
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


def _convert_messages_to_llama_format(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Convert messages to llama.cpp format."""
    return [{"role": msg.role, "content": msg.content} for msg in messages]


class ChatCompletionChunk(BaseModel):
    """Chat completion chunk for streaming."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]





async def _stream_completion(
    server_port: int,
    request: ChatCompletionRequest,
    request_id: str,
) -> AsyncGenerator[str]:
    """Stream completion from llama-server."""
    messages = _convert_messages_to_llama_format(request.messages)

    payload = {
        "messages": messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": True,
    }

    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.frequency_penalty is not None:
        payload["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None:
        payload["presence_penalty"] = request.presence_penalty
    if request.stop is not None:
        payload["stop"] = request.stop

    created = int(time.time())

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"http://localhost:{server_port}/completion",
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
                            continue

    except httpx.HTTPError as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


async def _get_completion(
    server_port: int,
    request: ChatCompletionRequest,
    request_id: str,
) -> ChatCompletionResponse:
    """Get non-streaming completion from llama-server."""
    messages = _convert_messages_to_llama_format(request.messages)

    payload = {
        "messages": messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": False,
    }

    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.frequency_penalty is not None:
        payload["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None:
        payload["presence_penalty"] = request.presence_penalty
    if request.stop is not None:
        payload["stop"] = request.stop

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"http://localhost:{server_port}/completion",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            created = int(time.time())
            content = result.get("content", "")

            prompt_tokens = result.get("prompt_tokens", 0)
            completion_tokens = result.get("completion_tokens", 0)

            return ChatCompletionResponse(
                id=request_id,
                created=created,
                model=request.model,
                choices=[ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason=result.get("finish_reason", "stop"),
                )],
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

    except httpx.HTTPError as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get completion: {e}")


async def _find_model_server(
    db: Session,
    model_name: str,
) -> int:
    """Find or start a server for the model."""
    statement = select(Model).where(Model.name == model_name)
    model = db.exec(statement).first()

    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    from app.services.llama_server import ServerConfig, llama_server_manager

    statement = select(ServerInstance).where(
        ServerInstance.model_id == model.id,
        ServerInstance.status == "running",
    )
    instance = db.exec(statement).first()

    if instance:
        health = await llama_server_manager.check_health(instance.port)
        if health == "healthy":
            return instance.port

    config = ServerConfig(
        model_path=model.path,
        port=8080,
    )

    new_instance = ServerInstance(
        model_id=model.id,
        port=config.port,
        status="starting",
        inactivity_timeout_seconds=300,
    )

    if await llama_server_manager.start_server(config, new_instance):
        db.add(new_instance)
        db.commit()
        return config.port

    raise HTTPException(status_code=503, detail="Failed to start model server")


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    db: Session = Depends(get_db),
):
    """
    Create a chat completion.

    Supports both streaming (SSE) and non-streaming responses.
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    try:
        server_port = await _find_model_server(db, request.model)
    except HTTPException:
        raise

    if request.stream:
        from fastapi.responses import StreamingResponse

        async def generate() -> AsyncGenerator[str]:
            async for chunk in _stream_completion(server_port, request, request_id):
                yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
            },
        )
    else:
        response = await _get_completion(server_port, request, request_id)
        return response

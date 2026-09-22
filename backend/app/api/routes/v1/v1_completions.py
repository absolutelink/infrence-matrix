"""V1 Completions endpoint - OpenAI-compatible legacy completions API via Agent proxy."""

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.api.deps import get_db
from app.api.routes.v1.v1_chat_completions import _get_or_create_server
from app.models import Model
from app.services.agent_manager import agent_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class CompletionRequest(BaseModel):
    """Completion request for legacy /v1/completions endpoint."""

    model: str
    agent_id: str | None = None  # Optional: specify which agent to use
    prompt: str | list[str] | list[int] | list[list[int]]
    stream: bool = False
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    echo: bool = False
    n: int = 1
    logprobs: int | None = None
    suffix: str | None = None


class CompletionChoice(BaseModel):
    """Completion choice."""

    text: str
    index: int
    logprobs: dict[str, str] | None = None
    finish_reason: str | None = None


class UsageInfo(BaseModel):
    """Usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Completion response."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo | None = None


class StreamChoice(BaseModel):
    """Streaming choice."""

    text: str
    index: int
    logprobs: dict[str, str] | None = None
    finish_reason: str | None = None


class CompletionChunk(BaseModel):
    """Completion chunk for streaming."""

    id: str
    object: str = "text_completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


def _convert_prompt_to_llama_format(
    prompt: str | list[str] | list[int] | list[list[int]],
) -> str:
    """Convert prompt to llama.cpp format."""
    if isinstance(prompt, str):
        return prompt
    elif isinstance(prompt, list):
        if len(prompt) == 0:
            return ""
        if isinstance(prompt[0], int):
            return " ".join(str(p) for p in prompt)
        elif isinstance(prompt[0], str):
            return " ".join(prompt)
        elif isinstance(prompt[0], list):
            return " ".join(" ".join(str(p) for p in sub) for sub in prompt)
    return str(prompt)


def _build_payload(request: CompletionRequest, stream: bool) -> dict:
    """Build llama.cpp /v1/completions payload."""
    payload: dict = {
        "prompt": _convert_prompt_to_llama_format(request.prompt),
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": stream,
    }

    for key in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
        value = getattr(request, key)
        if value is not None:
            payload[key] = value

    if request.echo:
        payload["echo"] = True
    if request.n != 1:
        payload["n"] = request.n
    if request.suffix is not None:
        payload["suffix"] = request.suffix

    return payload


async def _stream_completion_via_agent(
    agent_id: str,
    server_id: str,
    request: CompletionRequest,
    request_id: str,
) -> AsyncGenerator[str]:
    """Stream completion via Agent proxy."""
    payload = _build_payload(request, stream=True)
    created = int(time.time())

    try:
        agent = await agent_manager.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        async with httpx.AsyncClient(timeout=300.0) as client:
            proxy_url = (
                f"http://{agent.host}:{agent.port}/proxy/{server_id}/v1/completions"
            )

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
                            text = _extract_chunk_text(chunk_data)

                            stream_chunk = CompletionChunk(
                                id=request_id,
                                created=created,
                                model=request.model,
                                choices=[
                                    StreamChoice(
                                        text=text,
                                        index=0,
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


def _extract_chunk_text(chunk_data: dict) -> str:
    """Extract text from a streaming chunk (OpenAI or llama.cpp native format)."""
    choices = chunk_data.get("choices")
    if choices and isinstance(choices, list):
        choice = choices[0]
        if "text" in choice:
            return choice.get("text", "")
        if "delta" in choice:
            return choice["delta"].get("content", "") or choice["delta"].get("text", "")
        if "content" in choice:
            return choice.get("content", "")
    return chunk_data.get("content", "")


@router.post("/completions", response_model=None)
async def create_completion(
    request: CompletionRequest,
    db: Session = Depends(get_db),
) -> CompletionResponse | StreamingResponse:
    """Create a completion (legacy GPT-3 style endpoint) via Agent proxy."""
    request_id = f"cmpl-{uuid.uuid4()}"
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

    try:
        server = await _get_or_create_server(model, request.agent_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Server creation failed: {e}")
        raise HTTPException(503, f"Failed to start server: {e}")

    if request.stream:
        return StreamingResponse(
            _stream_completion_via_agent(
                str(server.agent_id),
                str(server.id),
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

    try:
        response = await agent_manager.send_to_agent(
            str(server.agent_id),
            "POST",
            f"/proxy/{server.id}/v1/completions",
            _build_payload(request, stream=False),
            timeout=300.0,
        )

        text = ""
        finish_reason = None
        prompt_tokens = 0
        completion_tokens = 0

        choices = response.get("choices")
        if choices and isinstance(choices, list):
            choice = choices[0]
            text = choice.get("text", "")
            finish_reason = choice.get("finish_reason", "stop")
            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            # llama.cpp native /completion response
            text = response.get("content", "")
            finish_reason = response.get("finish_reason", "stop")
            prompt_tokens = response.get("tokens_evaluated", 0)
            completion_tokens = response.get("tokens_predicted", 0)

        return CompletionResponse(
            id=request_id,
            created=created,
            model=request.model,
            choices=[
                CompletionChoice(
                    text=text,
                    index=0,
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(500, f"Inference failed: {e}")

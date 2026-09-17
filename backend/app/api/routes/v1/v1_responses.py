"""V1 Responses endpoint - OpenAI Responses API with tree-structured conversations."""

import logging
import time
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.api.deps import get_db
from app.models import Conversation, Model

logger = logging.getLogger(__name__)

router = APIRouter()


class InputItem(BaseModel):
    """Input item for Responses API."""
    type: Literal["message", "function_call", "function_call_output", "reasoning"]
    role: Literal["user", "assistant", "system", "developer"] | None = None
    content: str | list[dict[str, str]] | None = None


class ReasoningConfig(BaseModel):
    """Reasoning configuration."""
    effort: Literal["none", "low", "medium", "high", "xhigh"] = "medium"
    summary: Literal["concise", "detailed", "auto"] = "concise"


class ResponseRequest(BaseModel):
    """Response request."""
    model: str
    input: str | list[InputItem]
    previous_response_id: str | None = None
    stream: bool = False
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_output_tokens: int | None = None
    tools: list[dict[str, str]] = Field(default_factory=list)
    tool_choice: str | dict[str, str] = "auto"
    metadata: dict[str, str] = Field(default_factory=dict)
    reasoning: ReasoningConfig | None = None
    include: list[str] = Field(default_factory=list)
    store: bool = True
    truncation: Literal["auto", "disabled"] = "auto"


def _convert_input_to_messages(input_data: str | list[InputItem]) -> list[dict[str, str]]:
    """Convert input items to message format."""
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]

    messages = []
    for item in input_data:
        if item.type == "message":
            messages.append({
                "role": item.role or "user",
                "content": item.content if item.content else "",
            })
    return messages


class OutputContent(BaseModel):
    """Output content."""
    type: str = "output_text"
    text: str


class OutputItem(BaseModel):
    """Output item."""
    type: Literal["message", "function_call", "reasoning"]
    id: str
    role: Literal["assistant"]
    content: list[OutputContent]


class ResponseUsage(BaseModel):
    """Response usage."""
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ResponseData(BaseModel):
    """Response data."""
    id: str
    object: str = "response"
    created_at: int
    model: str
    input: list[dict[str, str]]
    output: list[OutputItem]
    status: Literal["in_progress", "completed", "failed"]
    usage: ResponseUsage | None = None


def _convert_input_to_messages(input_data: str | list[InputItem]) -> list[dict[str, str]]:
    """Convert input items to message format."""
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]

    messages: list[dict[str, str]] = []
    for item in input_data:
        if item.type == "message":
            messages.append({
                "role": item.role or "user",  # type: ignore[dict-item]
                "content": str(item.content) if item.content else "",
            })
    return messages


async def _find_parent_conversation(
    db: Session,
    previous_response_id: str | None,
) -> Conversation | None:
    """Find parent conversation for tree structure."""
    if not previous_response_id:
        return None

    statement = select(Conversation).where(
        Conversation.response_id == previous_response_id
    )
    return db.exec(statement).first()


async def _create_conversation_record(
    db: Session,
    model: Model,
    request: ResponseRequest,
    response_id: str,
    parent: Conversation | None = None,
) -> Conversation:
    """Create a conversation record in the database."""
    input_items = [
        item.model_dump() if hasattr(item, "model_dump") else item
        for item in request.input
    ] if isinstance(request.input, list) else []

    parameters = {
        "temperature": request.temperature,
        "max_output_tokens": request.max_output_tokens,
        "tools": request.tools,
        "tool_choice": request.tool_choice,
        "reasoning": request.reasoning.model_dump() if request.reasoning else None,
    }

    conversation = Conversation(
        parent_id=parent.id if parent else None,
        response_id=response_id,
        input_items=input_items,
        output_items=[],
        model_id=model.id,
        parameters=parameters,
        metadata=request.metadata,
        status="in_progress",
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return conversation


async def _update_conversation_completion(
    db: Session,
    conversation: Conversation,
    output_items: list[dict],
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Update conversation with completion data."""
    conversation.output_items = output_items
    conversation.status = "completed"
    conversation.input_tokens = input_tokens
    conversation.output_tokens = output_tokens
    conversation.total_tokens = input_tokens + output_tokens
    conversation.completed_at = None

    db.add(conversation)
    db.commit()


@router.post("/v1/responses", response_model=ResponseData)
async def create_response(
    request: ResponseRequest,
    db: Session = Depends(get_db),
) -> ResponseData:
    """
    Create a response using the Responses API.

    Supports tree-structured conversations with branching.
    """
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    created_at = int(time.time())

    statement = select(Model).where(Model.name == request.model)
    model: Model | None = db.exec(statement).first()

    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")

    parent = await _find_parent_conversation(db, request.previous_response_id)

    conversation = await _create_conversation_record(
        db, model, request, response_id, parent
    )

    from app.api.routes.v1.v1_chat_completions import _find_model_server

    try:
        server_port = await _find_model_server(db, request.model)
    except HTTPException as e:
        conversation.status = "failed"
        conversation.error_message = str(e.detail)
        db.add(conversation)
        db.commit()
        raise

    messages = _convert_input_to_messages(request.input)

    import httpx
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            payload = {
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_output_tokens,
                "stream": False,
            }

            if request.reasoning:
                payload["reasoning_effort"] = request.reasoning.effort

            response = await client.post(
                f"http://localhost:{server_port}/completion",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            content = result.get("content", "")
            prompt_tokens = result.get("prompt_tokens", 0)
            completion_tokens = result.get("completion_tokens", 0)

            output_item = OutputItem(
                type="message",
                id=f"msg_{uuid.uuid4().hex[:12]}",
                role="assistant",
                content=[OutputContent(text=content)],
            )

            await _update_conversation_completion(
                db,
                conversation,
                [output_item.model_dump()],
                prompt_tokens,
                completion_tokens,
            )

            return ResponseData(
                id=response_id,
                created_at=created_at,
                model=request.model,
                input=[item.model_dump() if hasattr(item, "model_dump") else {} for item in request.input] if isinstance(request.input, list) else [],  # type: ignore[misc]
                output=[output_item],
                status="completed",
                usage=ResponseUsage(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

    except httpx.HTTPError as e:
        logger.error(f"Response API error: {e}")
        conversation.status = "failed"
        conversation.error_message = str(e)
        db.add(conversation)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to create response: {e}")


@router.get("/v1/responses/{response_id}")
async def retrieve_response(
    response_id: str,
    db: Session = Depends(get_db),
) -> ResponseData:
    """Retrieve a specific response by ID."""
    statement = select(Conversation).where(Conversation.response_id == response_id)
    conversation = db.exec(statement).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Response not found")

    statement = select(Model).where(Model.id == conversation.model_id)
    model = db.exec(statement).first()

    return ResponseData(
        id=response_id,
        created_at=int(conversation.created_at.timestamp()),
        model=model.name if model else "unknown",
        input=conversation.input_items,  # type: ignore[arg-type]
        output=conversation.output_items,  # type: ignore[arg-type]
        status=conversation.status,  # type: ignore[arg-type]
        usage=ResponseUsage(
            input_tokens=conversation.input_tokens,
            output_tokens=conversation.output_tokens,
            total_tokens=conversation.total_tokens,
        ) if conversation.total_tokens else None,
    )

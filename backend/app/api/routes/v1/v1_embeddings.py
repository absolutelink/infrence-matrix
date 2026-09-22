"""V1 Embeddings endpoint - OpenAI-compatible embeddings API via Agent proxy."""

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.api.deps import get_db
from app.api.routes.v1.v1_chat_completions import _get_or_create_server
from app.models import Model
from app.services.agent_manager import agent_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class EmbeddingRequest(BaseModel):
    """Embedding request."""

    model: str
    agent_id: str | None = None  # Optional: specify which agent to use
    input: str | list[str] = Field(..., description="Input text or list of texts")
    encoding_format: Literal["float", "base64"] = "float"


class EmbeddingData(BaseModel):
    """Embedding data."""

    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    """Embedding usage."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """Embedding response."""

    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    db: Session = Depends(get_db),
) -> EmbeddingResponse:
    """Create embeddings for the given input text via Agent proxy."""
    model = db.exec(select(Model).where(Model.name == request.model)).first()
    if not model:
        raise HTTPException(404, f"Model {request.model} not found")

    if not model.supports_embeddings:
        raise HTTPException(400, f"Model '{request.model}' does not support embeddings")

    try:
        server = await _get_or_create_server(model, request.agent_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Server creation failed: {e}")
        raise HTTPException(503, f"Failed to start server: {e}")

    inputs = request.input if isinstance(request.input, list) else [request.input]

    try:
        response = await agent_manager.send_to_agent(
            str(server.agent_id),
            "POST",
            f"/proxy/{server.id}/v1/embeddings",
            {
                "input": inputs,
                "encoding_format": request.encoding_format,
            },
            timeout=300.0,
        )
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(500, f"Failed to create embeddings: {e}")

    if not isinstance(response, dict) or "data" not in response:
        raise HTTPException(500, "Invalid embedding response from server")

    embeddings_data = [
        EmbeddingData(
            embedding=item["embedding"],
            index=item.get("index", i),
        )
        for i, item in enumerate(response["data"])
    ]

    usage = response.get("usage", {})
    total_tokens = usage.get("prompt_tokens", usage.get("total_tokens", len(inputs)))

    return EmbeddingResponse(
        data=embeddings_data,
        model=request.model,
        usage=EmbeddingUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens,
        ),
    )

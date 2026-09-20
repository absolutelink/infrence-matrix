"""V1 Embeddings endpoint - OpenAI-compatible embeddings API."""

import logging
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.api.deps import get_db
from app.models import Model, ServerInstance

logger = logging.getLogger(__name__)

router = APIRouter()


class EmbeddingRequest(BaseModel):
    """Embedding request."""
    model: str
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


async def _find_embedding_server(
    db: Session,
    model_name: str,
) -> int:
    """Find or start a server for the embedding model."""
    statement = select(Model).where(Model.name == model_name)
    model = db.exec(statement).first()

    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    if not model.supports_embeddings:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' does not support embeddings"
        )

    from app.services.llama_server import ServerConfig, llama_server_manager

    statement = select(ServerInstance).where(
        ServerInstance.model_id == model.id,
        ServerInstance.status == "running",
    )
    instance: ServerInstance | None = db.exec(statement).first()

    if instance:
        health = await llama_server_manager.check_health(instance.port)
        if health == "healthy":
            return instance.port

    config = ServerConfig(
        model_path=model.path,
        port=8081,
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

    raise HTTPException(status_code=503, detail="Failed to start embedding server")


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    db: Session = Depends(get_db),
) -> EmbeddingResponse:
    """
    Create embeddings for the given input text.

    Args:
        request: Embedding request with model and input text
    """
    try:
        server_port = await _find_embedding_server(db, request.model)
    except HTTPException:
        raise

    inputs = request.input if isinstance(request.input, list) else [request.input]

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"http://localhost:{server_port}/embedding",
                json={
                    "content": inputs,
                },
            )
            response.raise_for_status()
            result = response.json()

            embeddings_data = []
            total_tokens = 0

            if "embedding" in result:
                embedding = result["embedding"]
                embeddings_data.append(EmbeddingData(
                    embedding=embedding,
                    index=0,
                ))
                total_tokens = len(inputs)

            elif "data" in result:
                for i, emb in enumerate(result["data"]):
                    embeddings_data.append(EmbeddingData(
                        embedding=emb["embedding"],
                        index=i,
                    ))
                total_tokens = result.get("usage", {}).get("prompt_tokens", len(inputs))

            else:
                raise HTTPException(status_code=500, detail="Invalid embedding response")

            return EmbeddingResponse(
                data=embeddings_data,
                model=request.model,
                usage=EmbeddingUsage(
                    prompt_tokens=total_tokens,
                    total_tokens=total_tokens,
                ),
            )

    except httpx.HTTPError as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {e}")

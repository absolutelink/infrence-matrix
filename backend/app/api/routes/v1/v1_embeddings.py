"""V1 Embeddings endpoint - OpenAI-compatible embeddings API via Agent proxy."""

import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.api.deps import get_db
from app.api.routes.v1.v1_chat_completions import _get_or_create_server
from app.models import Model, ServerInstance
from app.services.agent_manager import agent_manager
from app.services.inference_scheduler import InferenceLeaseHandle, inference_scheduler
from app.services.server_startup import metadata_capability

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
    # Resolve model field: a server alias routes to that server directly;
    # otherwise it is a model name/id and a server is found or started.
    server: ServerInstance | None = None
    server_q = select(ServerInstance).where(
        ServerInstance.alias == request.model,
    )
    instance = db.exec(server_q).first()

    if instance is not None:
        if metadata_capability(instance, "embeddings") is False:
            raise HTTPException(
                400, f"Model '{request.model}' does not support embeddings"
            )
        server = instance
        model = db.exec(select(Model).where(Model.id == instance.model_id)).first()
        if not model:
            raise HTTPException(
                404, f"Model for server alias {request.model} not found"
            )
    else:
        # Get model by name (OpenAI style) or id (UI legacy)
        model = db.exec(select(Model).where(Model.name == request.model)).first()
        if not model:
            try:
                model = db.exec(select(Model).where(Model.id == request.model)).first()
            except Exception:
                model = None
        if not model:
            raise HTTPException(404, f"Model {request.model} not found")

        if not model.supports_embeddings:
            raise HTTPException(
                400, f"Model '{request.model}' does not support embeddings"
            )

        try:
            server = await _get_or_create_server(model, request.agent_id, start=False)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Server creation failed: {e}")
            raise HTTPException(503, f"Failed to start server: {e}")

    inputs = request.input if isinstance(request.input, list) else [request.input]

    lease: InferenceLeaseHandle | None = None
    try:
        lease = await inference_scheduler.acquire(
            model.id, f"embed-{uuid.uuid4()}", preferred_server_id=server.id
        )
        server = lease.server
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
    except TimeoutError as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(500, f"Failed to create embeddings: {e}")
    finally:
        if lease is not None:
            await lease.release()

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

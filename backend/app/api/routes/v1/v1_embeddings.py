"""V1 Embeddings endpoint - OpenAI-compatible embeddings API via Agent proxy."""

import asyncio
import logging
import time
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.api.deps import get_db
from app.api.routes.v1.v1_chat_completions import _get_or_create_server
from app.models import ServerInstance
from app.services.agent_manager import agent_manager
from app.services.inference_scheduler import InferenceLeaseHandle, inference_scheduler
from app.services.inference_target import resolve_inference_target
from app.services.server_startup import metadata_capability
from app.services.token_stats import record_request_telemetry

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
    http_request: Request = None,
) -> EmbeddingResponse:
    """Create embeddings for the given input text via Agent proxy."""
    started = time.monotonic()
    # Resolve model field: a server alias routes to that server directly;
    # otherwise it is a model name/id and a server is found or started.
    try:
        target = resolve_inference_target(db, request.model, request.agent_id)
    except (LookupError, ValueError) as e:
        raise HTTPException(404, str(e)) from e
    instance = target.server
    server: ServerInstance | None = instance
    db.close()

    if instance is not None:
        if (
            getattr(instance, "engine", None) == "halogen-flash"
            and not target.npu_model
        ):
            raise HTTPException(
                400,
                f"Model '{request.model}' is a halogen-flash server; request "
                "embeddings via its '<alias>-embed' NPU model name",
            )
        # An NPU alias's capability is implied by the model being enabled on
        # the instance; do not gate it on the Flash checkpoint's metadata,
        # which describes the big model, not the NPU embedder.
        if not target.npu_model and metadata_capability(instance, "embeddings") is False:
            raise HTTPException(
                400, f"Model '{request.model}' does not support embeddings"
            )
        server = instance
        model = target.model
    else:
        model = target.model
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
    terminal_outcome = "completed"
    try:
        lease = await inference_scheduler.acquire(
            model.id,
            f"embed-{uuid.uuid4()}",
            preferred_server_id=target.preferred_server_id,
            required_agent_id=target.required_agent_id,
            is_cancelled=http_request.is_disconnected if http_request else None,
        )
        server = lease.server
        response = await lease.guard(
            agent_manager.send_to_agent(
                str(server.agent_id),
                "POST",
                f"/proxy/{server.id}/v1/embeddings",
                {
                    "input": inputs,
                    "encoding_format": request.encoding_format,
                    **({"model": target.npu_model} if target.npu_model else {}),
                },
                timeout=1800.0,
                headers={
                    **lease.dispatch_headers(),
                },
            )
        )
        # Capture before the finally's release cooldown contaminates it.
        success_latency_ms = (time.monotonic() - started) * 1000.0

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
        total_tokens = usage.get(
            "prompt_tokens", usage.get("total_tokens", len(inputs))
        )
        result = EmbeddingResponse(
            data=embeddings_data,
            model=request.model,
            usage=EmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens,
            ),
        )
        record_request_telemetry(
            server,
            {"prompt_tokens": total_tokens},
            response.get("timings"),
            latency_ms=success_latency_ms,
            status="success",
        )
        return result
    except HTTPException:
        terminal_outcome = "failed"
        record_request_telemetry(
            server,
            None,
            None,
            latency_ms=(time.monotonic() - started) * 1000.0,
            status="error",
        )
        raise
    except asyncio.CancelledError:
        terminal_outcome = "cancelled"
        record_request_telemetry(
            server,
            None,
            None,
            latency_ms=(time.monotonic() - started) * 1000.0,
            status="cancelled",
        )
        raise
    except TimeoutError as e:
        terminal_outcome = "failed"
        record_request_telemetry(
            server,
            None,
            None,
            latency_ms=(time.monotonic() - started) * 1000.0,
            status="error",
        )
        raise HTTPException(503, str(e)) from e
    except Exception as e:
        terminal_outcome = "failed"
        record_request_telemetry(
            server,
            None,
            None,
            latency_ms=(time.monotonic() - started) * 1000.0,
            status="error",
        )
        logger.error(f"Embedding error: {e}")
        raise HTTPException(500, f"Failed to create embeddings: {e}")
    finally:
        if lease is not None:
            await lease.release(terminal_outcome)

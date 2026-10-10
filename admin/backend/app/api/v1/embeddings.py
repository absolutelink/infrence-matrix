"""Public OpenAI Embeddings endpoint: ``POST /v1/embeddings`` (Phase 18).

Mirrors the Phase 7 ``/v1/chat/completions`` **non-stream** path
(``app/api/v1/chat_completions.py``) against ``litellm.aembedding`` instead
of ``acompletion``: same scheduler admission, same provider port, same
cancellation-safe release + persist discipline. Embedding differences:

- Embeddings are **non-streaming only** — the OpenAI embeddings response is
  a single JSON object, so there is no SSE to frame. Every error is a real
  HTTP JSON response (400/404/503/504 pre-stream, 502 for upstream
  failures).
- The alias must resolve to a ``modality == "embedding"`` definition; an
  ``llm`` alias is a clean 404 (the reverse gate lives in the chat routes).
- litellm registration uses ``mode: "embedding"`` (a distinct litellm mode;
  ``aembedding`` for ``custom_llm_provider="openai"`` posts to
  ``{api_base}/embeddings`` — verified against litellm 1.103.2).
- Persistence is a ``TokenUsageSample`` **only** (prompt tokens) — embeddings
  are not conversation turns, so no ``ResponseRecord`` is written.
- The admin owns the ``model`` name on the response (never leaks the raw
  backend path the provider echoes back).

Auth model: trusted LAN — this public route is unauthenticated.
"""

import asyncio
import contextlib
import logging
import uuid
from typing import Any

import litellm
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlmodel import Session

from app.api.v1.responses import get_scheduler, map_exception_to_error
from app.core.db import engine
from app.models import TokenUsageSample
from app.services.alias_registry import ensure_registered
from app.services.scheduler import NoProviderAvailable, QueueCleared, QueueTimeout
from app.services.served_models import resolve_served
from app.services.sse import to_dict

logger = logging.getLogger("admin.v1.embeddings")

router = APIRouter(tags=["embeddings"])


def _persist_embedding_usage(
    *,
    definition_id: uuid.UUID | None,
    instance_id: uuid.UUID,
    prompt_tokens: int,
) -> None:
    """Write ONLY a TokenUsageSample for an embedding call (no ResponseRecord).

    Embeddings are telemetry, not conversation turns — see ARCHITECTURE.md §7.
    A DB failure must never break the response, but it must not vanish
    silently either, so log loudly (same discipline as ``persist_turn_logged``).
    """
    try:
        sample = TokenUsageSample(
            provider_instance_id=instance_id,
            provider_definition_id=definition_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            cached_tokens=0,
        )
        with Session(engine) as session:
            session.add(sample)
            session.commit()
    except Exception:
        logger.exception(
            "failed to persist embedding token usage for instance %s", instance_id
        )


@router.post("/v1/embeddings")
async def create_embedding(request: Request) -> Any:
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400, detail="request body must be a JSON object"
        )

    alias = body.get("model")
    if not alias:
        raise HTTPException(status_code=400, detail="'model' is required")

    input_ = body.get("input")
    if isinstance(input_, str):
        if not input_:
            raise HTTPException(status_code=400, detail="'input' must not be empty")
    elif isinstance(input_, list):
        if not input_:
            raise HTTPException(status_code=400, detail="'input' must be non-empty")
    else:
        raise HTTPException(
            status_code=400,
            detail="'input' must be a non-empty string or list",
        )

    with Session(engine) as session:
        resolved = resolve_served(session, alias)
        if resolved is None:
            raise HTTPException(
                status_code=404, detail=f"unknown model alias '{alias}'"
            )
        definition, spec = resolved
        if not definition.enabled:
            raise HTTPException(
                status_code=404, detail=f"model alias '{alias}' is disabled"
            )
        if spec.modality != "embedding":
            raise HTTPException(
                status_code=404,
                detail=f"model alias '{alias}' is not an embedding model",
            )
        definition_id = definition.id

    request_id = f"emb-{uuid.uuid4().hex}"
    ensure_registered(alias, mode="embedding")

    scheduler = get_scheduler(request)
    try:
        admission = await scheduler.acquire(alias, request_id)
    except NoProviderAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QueueCleared as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QueueTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc

    kwargs: dict[str, Any] = {"input": input_}
    if "dimensions" in body:
        kwargs["dimensions"] = body["dimensions"]
    if "user" in body:
        kwargs["user"] = body["user"]

    try:
        try:
            result = await litellm.aembedding(
                model=alias,
                custom_llm_provider="openai",
                # The openai embedding path demands a non-empty key even for
                # unauthenticated local upstreams; the provider ignores it.
                api_key="unused",
                api_base=f"{admission.base_url}/v1",
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - upstream litellm failure
            error = map_exception_to_error(exc)
            logger.warning("litellm embedding call failed for %s: %s", request_id, exc)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "type": error["type"],
                        "code": error["code"],
                        "message": error["message"],
                    }
                },
            )

        data = to_dict(result)
        # The admin owns the model name: the provider may echo its raw
        # backend path (e.g. a GGUF path), which must not leak to clients.
        data["model"] = alias
        usage = data.get("usage") or {}
        _persist_embedding_usage(
            definition_id=definition_id,
            instance_id=uuid.UUID(admission.instance_id),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
        )
        return JSONResponse(content=data)
    finally:
        # Guaranteed slot release on EVERY exit path — including
        # asyncio.CancelledError (client disconnect during the await), which
        # `except Exception` does not catch. Release is idempotent.
        with contextlib.suppress(Exception):
            await asyncio.shield(scheduler.release(alias, request_id))

"""V1 Rerank endpoint - OpenAI-style reranking via halogen-flash NPU.

Clients send a broker-shaped request; the body is translated to halogen's
Cohere-shaped ``/v1/rerank`` (upstream task line is ``instruction``). The
target NPU model is selected by the ``<alias>-rerank`` virtual name.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.api.deps import get_db
from app.services.npu_dispatch import dispatch_npu, resolve_npu_target

router = APIRouter()


class RerankRequest(BaseModel):
    """Broker-shaped rerank request."""

    model: str = Field(
        ..., description="NPU rerank virtual name, e.g. 'myflash-rerank'"
    )
    agent_id: str | None = None
    query: str
    documents: list[str] = Field(..., min_length=1)
    top_n: int | None = Field(default=None, ge=1)
    instruct: str | None = Field(
        default=None, description="Replaces the model's default task line"
    )
    return_documents: bool = False


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: str | None = None


class RerankUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResult]
    usage: RerankUsage


@router.post("/rerank", response_model=RerankResponse)
async def rerank(
    request: RerankRequest,
    db: Session = Depends(get_db),
    http_request: Request = None,
) -> RerankResponse:
    target = resolve_npu_target(db, request.model, request.agent_id, "rerank")
    db.close()

    body: dict[str, Any] = {
        "model": target.npu_model,
        "query": request.query,
        "documents": request.documents,
        "return_documents": request.return_documents,
    }
    if request.top_n is not None:
        body["top_n"] = request.top_n
    if request.instruct is not None:
        body["instruction"] = request.instruct

    response = await dispatch_npu(
        target,
        "v1/rerank",
        body,
        request_id_prefix="rerank",
        is_disconnected=http_request.is_disconnected if http_request else None,
    )

    raw_results = response.get("results")
    if not isinstance(raw_results, list):
        raise HTTPException(500, "Invalid rerank response from server")

    results = [
        RerankResult(
            index=int(item.get("index", i)),
            relevance_score=float(item.get("relevance_score", 0.0)),
            document=(
                item.get("document")
                if isinstance(item.get("document"), str)
                else (item.get("document") or {}).get("text")
                if isinstance(item.get("document"), dict)
                else None
            ),
        )
        for i, item in enumerate(raw_results)
    ]
    usage_data = response.get("usage") or {}
    total = usage_data.get("total_tokens", usage_data.get("prompt_tokens", 0))
    return RerankResponse(
        model=request.model,
        results=results,
        usage=RerankUsage(
            prompt_tokens=int(usage_data.get("prompt_tokens", 0) or 0),
            total_tokens=int(total or 0),
        ),
    )

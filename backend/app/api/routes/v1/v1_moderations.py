"""V1 Moderations endpoint - content moderation on the halogen-flash NPU.

A passthrough of OpenAI's ``/v1/moderations`` shape to the NPU
``qwen3guard-gen-0.6b`` model reached through the ``<alias>-guard`` virtual
name. The upstream reply is already OpenAI-shaped and additionally carries
``label`` and ``label_scores`` per result, which are passed through.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import Session

from app.api.deps import get_db
from app.services.npu_dispatch import dispatch_npu, resolve_npu_target

router = APIRouter()


class ModerationRequest(BaseModel):
    model: str = Field(
        ..., description="NPU moderation virtual name, e.g. 'myflash-guard'"
    )
    agent_id: str | None = None
    input: str | list[str] | None = None
    messages: list[dict[str, Any]] | None = Field(
        default=None,
        description="Chat-shaped conversation; judges the last reply",
    )
    strict: bool | None = Field(
        default=None, description="Flag 'Controversial' as well as 'Unsafe'"
    )
    model_config = ConfigDict(extra="ignore")


class ModerationResponse(BaseModel):
    id: str
    model: str
    results: list[dict[str, Any]]


@router.post("/moderations", response_model=ModerationResponse)
async def moderate(
    request: ModerationRequest,
    db: Session = Depends(get_db),
    http_request: Request = None,
) -> ModerationResponse:
    target = resolve_npu_target(db, request.model, request.agent_id, "guard")
    db.close()

    body: dict[str, Any] = {"model": target.npu_model}
    if request.messages is not None:
        body["messages"] = request.messages
    elif request.input is not None:
        body["input"] = request.input
    else:
        raise HTTPException(400, "Provide 'input' or 'messages'")
    if request.strict is not None:
        body["strict"] = request.strict

    response = await dispatch_npu(
        target,
        "v1/moderations",
        body,
        request_id_prefix="moderation",
        is_disconnected=http_request.is_disconnected if http_request else None,
    )

    results = response.get("results")
    if not isinstance(results, list):
        raise HTTPException(500, "Invalid moderation response")
    return ModerationResponse(
        id=response.get("id", ""),
        model=request.model,
        results=results,
    )

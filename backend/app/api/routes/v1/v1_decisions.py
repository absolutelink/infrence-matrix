"""V1 Decisions endpoint - single-pass classification on the halogen-flash NPU.

A decision asks a 2-10 option question about a text. The broker builds the
halogen decision request (a chat completion whose ``response_format`` carries
the question in ``description`` and the options in an ``enum`` schema),
reads the first token's ``top_logprobs`` for every option's probability, and
returns a structured decision. Generation runs on the NPU ``decider-0.8b``
model reached through the ``<alias>-decide`` virtual name.
"""

import json
import math
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.api.deps import get_db
from app.services.npu_dispatch import dispatch_npu, resolve_npu_target

router = APIRouter()


class DecisionRequest(BaseModel):
    model: str = Field(
        ..., description="NPU decision virtual name, e.g. 'myflash-decide'"
    )
    agent_id: str | None = None
    text: str = Field(..., description="The text to judge")
    question: str = Field(..., description="The question asked of the text")
    options: list[str] = Field(
        ..., min_length=2, max_length=10, description="2 to 10 answer options"
    )
    system: str | None = Field(
        default=None,
        description="Optional system message carrying the question instead "
        "of the schema description",
    )


class DecisionResponse(BaseModel):
    model: str
    decision: str
    probabilities: dict[str, float]


@router.post("/decisions", response_model=DecisionResponse)
async def decide(
    request: DecisionRequest,
    db: Session = Depends(get_db),
    http_request: Request = None,
) -> DecisionResponse:
    target = resolve_npu_target(db, request.model, request.agent_id, "decide")
    db.close()

    if len(set(request.options)) != len(request.options):
        raise HTTPException(400, "Decision options must be unique")

    messages: list[dict[str, str]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.text})

    body: dict[str, Any] = {
        "model": target.npu_model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "decision",
                "description": request.question,
                "schema": {"enum": request.options},
            },
        },
        "logprobs": True,
        "top_logprobs": len(request.options),
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }

    response = await dispatch_npu(
        target,
        "v1/chat/completions",
        body,
        request_id_prefix="decision",
        is_disconnected=http_request.is_disconnected if http_request else None,
    )

    choices = response.get("choices") or []
    if not choices:
        raise HTTPException(500, "Invalid decision response from server")
    choice = choices[0]
    raw = ((choice.get("message") or {}).get("content") or "").strip()
    # The answer is the option in the schema's JSON shape: an enum of strings
    # comes back quoted (``"yes"``). Unwrap so the decision is the bare option.
    decision = raw
    if raw.startswith('"') and raw.endswith('"'):
        try:
            decision = str(json.loads(raw))
        except (ValueError, TypeError):
            decision = raw.strip('"')

    probabilities: dict[str, float] = {}
    logprobs = choice.get("logprobs") or {}
    content = logprobs.get("content") or []
    if content:
        for entry in content[0].get("top_logprobs") or []:
            token = entry.get("token")
            logprob = entry.get("logprob")
            if token is None or logprob is None:
                continue
            label = token.strip('"') if token.startswith('"') else token
            probabilities[label] = probabilities.get(label, 0.0) + math.exp(
                float(logprob)
            )

    if decision not in request.options:
        ranked = [o for o in request.options if o in probabilities]
        if ranked:
            decision = max(ranked, key=lambda o: probabilities[o])

    return DecisionResponse(
        model=request.model,
        decision=decision,
        probabilities=probabilities,
    )

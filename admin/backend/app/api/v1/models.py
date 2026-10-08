"""Public model listing: ``GET /v1/models`` (Phase 7).

Derived entirely from enabled ``ProviderDefinition`` rows — no scheduler
or provider traffic. Each entry is an OpenAI model object: ``id`` = the
alias, ``owned_by`` = the provider type, ``created`` = the definition's
creation timestamp (int epoch seconds), with the definition's discovered
``model_metadata`` merged on top (``context_length``, capability flags,
... — metadata wins over the base fields except for the four core
fields, which stay authoritative).

Order is deterministic: alias ascending.
"""

import logging
from typing import Any

from fastapi import APIRouter
from sqlmodel import Session, col, select

from app.core.db import engine
from app.models import ProviderDefinition

logger = logging.getLogger("admin.v1.models")

router = APIRouter(tags=["models"])


def model_object(definition: ProviderDefinition) -> dict[str, Any]:
    """Build one OpenAI model object from a ProviderDefinition.

    The four core fields are authoritative from the definition row;
    ``model_metadata`` (discovered capabilities, context_length, ...) is
    merged on top for everything else.
    """
    obj: dict[str, Any] = {
        "id": definition.alias,
        "object": "model",
        "created": int(definition.created_at.timestamp()),
        "owned_by": definition.provider_type,
        # Phase 18: core routing marker so clients can filter by endpoint
        # family (llm | embedding). Authoritative from the definition row.
        "modality": definition.modality,
    }
    metadata = definition.model_metadata or {}
    for key, value in metadata.items():
        if key in ("id", "object", "created", "owned_by", "modality"):
            continue
        obj[key] = value
    return obj


@router.get("/v1/models")
def list_models() -> dict[str, Any]:
    with Session(engine) as session:
        definitions = session.exec(
            select(ProviderDefinition)
            .where(
                col(ProviderDefinition.enabled) == True,  # noqa: E712
                # Phase 14: shell definitions (no backend_config yet) are
                # not client-visible — they cannot serve traffic.
                col(ProviderDefinition.backend_config).is_not(None),
            )
            .order_by(col(ProviderDefinition.alias).asc())
        ).all()
        data = [model_object(d) for d in definitions]
    return {"object": "list", "data": data}

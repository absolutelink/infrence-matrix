"""Public model listing: ``GET /v1/models`` (Phase 7).

Derived entirely from enabled ``ProviderDefinition`` rows — no scheduler
or provider traffic. Each entry is an OpenAI model object: ``id`` = the
served name, ``owned_by`` = the provider type, ``created`` = the definition's
creation timestamp (int epoch seconds), with the definition's discovered
``model_metadata`` merged on top (``context_length``, capability flags, ... —
metadata wins over the base fields except for the core fields, which stay
authoritative).

Phase 25: a multi-model definition lists **every enabled served name** (one
object per ``ProviderModel`` with ``enabled=true``), each carrying its own
per-model ``modality`` marker; a single-model definition lists its alias as
before. ``modality`` is the served spec's modality (the endpoint routing key).

Order is deterministic: served name ascending.
"""

import logging
from typing import Any

from fastapi import APIRouter
from sqlmodel import Session, col, select

from app.core.db import engine
from app.models import ProviderDefinition
from app.services.served_models import ModelSpec, resolve_models

logger = logging.getLogger("admin.v1.models")

router = APIRouter(tags=["models"])


def model_object(definition: ProviderDefinition, spec: ModelSpec) -> dict[str, Any]:
    """Build one OpenAI model object for one served name on a definition.

    The core fields are authoritative from the definition row + the served
    spec; ``model_metadata`` (discovered capabilities, context_length, ...) is
    merged on top for everything else.
    """
    obj: dict[str, Any] = {
        "id": spec.name,
        "object": "model",
        "created": int(definition.created_at.timestamp()),
        "owned_by": definition.provider_type,
        # Phase 18/25: core routing marker so clients can filter by endpoint
        # family. Authoritative from the served spec (per-model for multi-model,
        # the definition's own modality for single-model).
        "modality": spec.modality,
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
        data: list[dict[str, Any]] = []
        seen: set[str] = set()
        for d in definitions:
            for spec in resolve_models(session, d):
                if not spec.enabled or spec.name in seen:
                    continue
                seen.add(spec.name)
                data.append(model_object(d, spec))
    return {"object": "list", "data": data}

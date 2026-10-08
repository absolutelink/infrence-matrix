"""Admin ProviderType registry endpoints (Phase 12).

``/admin/api/provider-types`` — read the committed schema (the UI renders
the backend-config form from it) and the schema-consensus state, plus the
operator overrides for a stuck consensus:

- ``POST /{name}/pending/commit``  force-promote the pending schema to
  committed (e.g. a dead machine can never vote). Affects **future
  registrations and new/edited definitions only** — connected
  old-schema instances keep running until they upgrade and re-register.
- ``POST /{name}/pending/dismiss`` drop the pending schema; the type
  returns to ``active`` on the committed schema.

The consensus gate itself lives in ``app.api.admin.providers``; this
module only reads/overrides its state. See docs/ws-protocol.md §2.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.api.admin.providers import (
    _apply_max_running_from_schema,
    _apply_serves_modalities_from_schema,
    type_agents_of,
)
from app.core.db import get_session
from app.models import ProviderType

logger = logging.getLogger("admin.provider_types")

router = APIRouter(prefix="/admin/api/provider-types", tags=["admin"])


def _get_type(session: Session, name: str) -> ProviderType:
    ptype = session.exec(select(ProviderType).where(ProviderType.name == name)).first()
    if ptype is None:
        raise HTTPException(status_code=404, detail=f"provider type '{name}' not found")
    return ptype


def _consensus_dict(session: Session, ptype: ProviderType) -> dict[str, Any]:
    voters = [str(v) for v in (ptype.pending_voters or [])]
    agents = type_agents_of(session, ptype.name)
    universe = [str(agent.id) for agent in agents]
    summary: dict[str, Any] = {
        "status": ptype.status,
        "committed_fingerprint": ptype.schema_fingerprint,
        "universe": universe,
        "universe_count": len(universe),
        # Agents currently on the committed schema, per their last
        # registration report.
        "on_committed": sum(
            1
            for agent in agents
            if agent.reported_schema_fingerprint == ptype.schema_fingerprint
        ),
    }
    if ptype.pending_fingerprint is not None:
        summary["pending_fingerprint"] = ptype.pending_fingerprint
        summary["voters"] = voters
        summary["voter_count"] = len(voters)
        summary["waiting_on"] = [uid for uid in universe if uid not in voters]
    else:
        summary["pending_fingerprint"] = None
        summary["voters"] = []
        summary["voter_count"] = 0
        summary["waiting_on"] = []
    return summary


@router.get("")
def list_provider_types(
    session: Session = Depends(get_session),
) -> list[dict[str, Any]]:
    types = session.exec(select(ProviderType).order_by(ProviderType.name)).all()
    return [
        {
            "name": t.name,
            "status": t.status,
            "schema_fingerprint": t.schema_fingerprint,
            # Phase 16: per-agent running cap (0 = unlimited) — surfaced so the
            # Agents UI can show each agent's type cap without a second fetch.
            "max_running_backends": t.max_running_backends,
            # Phase 18: modalities this type can host (default ["llm"]).
            "serves_modalities": t.serves_modalities,
            "consensus": _consensus_dict(session, t),
        }
        for t in types
    ]


@router.get("/{name}")
def get_provider_type(
    name: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    """Full committed schema (for the UI form) + consensus state."""
    ptype = _get_type(session, name)
    return {
        "name": ptype.name,
        "status": ptype.status,
        "schema": ptype.schema,
        "schema_fingerprint": ptype.schema_fingerprint,
        "max_running_backends": ptype.max_running_backends,
        "serves_modalities": ptype.serves_modalities,
        "pending_schema": ptype.pending_schema,
        "consensus": _consensus_dict(session, ptype),
        "created_at": ptype.created_at.isoformat(),
        "updated_at": ptype.updated_at.isoformat() if ptype.updated_at else None,
    }


@router.post("/{name}/pending/commit")
def commit_pending(
    name: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    """Operator override: promote the pending schema to committed."""
    ptype = _get_type(session, name)
    if ptype.pending_fingerprint is None:
        raise HTTPException(
            status_code=409, detail=f"provider type '{name}' has nothing pending"
        )
    ptype.schema = ptype.pending_schema or {}
    ptype.schema_fingerprint = ptype.pending_fingerprint
    ptype.pending_schema = None
    ptype.pending_fingerprint = None
    ptype.pending_voters = []
    ptype.status = "active"
    _apply_max_running_from_schema(ptype, ptype.schema)
    _apply_serves_modalities_from_schema(ptype, ptype.schema)
    ptype.updated_at = datetime.now(UTC)
    session.add(ptype)
    session.commit()
    session.refresh(ptype)
    logger.info(
        "force-committed pending schema %s for provider type %s",
        ptype.schema_fingerprint[:8],
        name,
    )
    return {
        "ok": True,
        "name": name,
        "schema_fingerprint": ptype.schema_fingerprint,
        "consensus": _consensus_dict(session, ptype),
    }


@router.post("/{name}/pending/dismiss")
def dismiss_pending(
    name: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    """Operator override: drop the pending schema, back to active."""
    ptype = _get_type(session, name)
    if ptype.pending_fingerprint is None:
        raise HTTPException(
            status_code=409, detail=f"provider type '{name}' has nothing pending"
        )
    dismissed_fp = ptype.pending_fingerprint
    ptype.pending_schema = None
    ptype.pending_fingerprint = None
    ptype.pending_voters = []
    ptype.status = "active"
    ptype.updated_at = datetime.now(UTC)
    session.add(ptype)
    session.commit()
    session.refresh(ptype)
    logger.info(
        "dismissed pending schema %s for provider type %s", dismissed_fp[:8], name
    )
    return {
        "ok": True,
        "name": name,
        "dismissed_fingerprint": dismissed_fp,
        "schema_fingerprint": ptype.schema_fingerprint,
        "consensus": _consensus_dict(session, ptype),
    }

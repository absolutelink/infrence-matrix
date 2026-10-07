"""Admin provider-agent reads (Phase 16).

``/admin/api/agents`` — read-only listing of ``ProviderAgent`` rows (the
hardware-local containers bound to a (machine, provider_type, agent_id)
triple). Backends (``ProviderInstance``) nest under their agent so the UI can
show what each container hosts. Placement cherry-picking on definitions
references these agent ids.

Additive slice: agents are not yet created by registration (the registration
cutover lands in a follow-on slice), so this reads whatever rows exist.
"""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.api.admin.serializers import iso_utc
from app.core.db import get_session
from app.models import DefinitionAgent, ProviderAgent, ProviderInstance

logger = logging.getLogger("admin.agents")

router = APIRouter(prefix="/admin/api/agents", tags=["admin"])


def _agent_dict(
    agent: ProviderAgent,
    *,
    machine_uid: str | None = None,
    backends: list[dict[str, Any]] | None = None,
    definition_count: int | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": str(agent.id),
        "machine_id": str(agent.machine_id),
        "machine_uid": machine_uid,
        "provider_type": agent.provider_type,
        "agent_id": agent.agent_id,
        "base_port": agent.base_port,
        "version": agent.version,
        "agent_status": agent.agent_status,
        "websocket_connected": agent.websocket_connected,
        "epoch": agent.epoch,
        "reported_schema_fingerprint": agent.reported_schema_fingerprint,
        "assigned_gpus": agent.assigned_gpus,
        "error_message": agent.error_message,
        "last_seen": iso_utc(agent.last_seen),
        "created_at": iso_utc(agent.created_at),
        "updated_at": iso_utc(agent.updated_at),
    }
    if backends is not None:
        d["backends"] = backends
    if definition_count is not None:
        d["definition_count"] = definition_count
    return d


def _backend_dicts(session: Session, agent_id: uuid.UUID) -> list[dict[str, Any]]:
    rows = session.exec(
        select(ProviderInstance).where(ProviderInstance.machine_id == agent_id)
    ).all()
    # NOTE: ProviderInstance is still keyed by machine_id until the cutover
    # slice re-keys it to agent_id; the placeholder query returns nothing
    # meaningful yet and is finalized with the re-key.
    return [
        {
            "id": str(i.id),
            "provider_definition_id": str(i.provider_definition_id),
            "backend_status": i.backend_status,
            "port": i.port,
            "config_fingerprint": i.config_fingerprint,
        }
        for i in rows
    ]


def _definition_count(session: Session, agent_id: uuid.UUID) -> int:
    return len(
        session.exec(
            select(DefinitionAgent).where(DefinitionAgent.agent_id == agent_id)
        ).all()
    )


@router.get("")
def list_agents(session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    agents = session.exec(
        select(ProviderAgent).order_by(ProviderAgent.machine_id, ProviderAgent.agent_id)
    ).all()
    out: list[dict[str, Any]] = []
    for agent in agents:
        out.append(
            _agent_dict(
                agent,
                machine_uid=agent.machine.uid if agent.machine else None,
                backends=_backend_dicts(session, agent.id),
                definition_count=_definition_count(session, agent.id),
            )
        )
    return out


@router.get("/{agent_id}")
def get_agent(agent_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    try:
        agent = session.get(ProviderAgent, uuid.UUID(agent_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="agent not found") from None
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return _agent_dict(
        agent,
        machine_uid=agent.machine.uid if agent.machine else None,
        backends=_backend_dicts(session, agent.id),
        definition_count=_definition_count(session, agent.id),
    )

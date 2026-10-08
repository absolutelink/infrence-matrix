"""Admin machine CRUD (Phase 9).

``/admin/api/machines`` — unauthenticated trusted-LAN management of the
``Machine`` rows that provider agents register against (a machine must
pre-exist by ``uid``; see docs/ws-protocol.md §2).

Semantics:
- ``uid`` is immutable after creation: it is referenced by provider
  containers' ``MACHINE_UID`` env and by Redis metrics leases. Changing
  it would silently orphan agents; PATCH rejects ``uid``.
- ``DELETE`` is refused (409) while provider agents are still
  attached — delete the agents (or the definitions they belong to)
  first. No cascade: an accidental machine delete would take every
  agent row (and their FK cascades) with it.
"""

import logging
import secrets
import uuid
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, func, select

from app.api.admin.serializers import iso_utc
from app.core.db import get_session
from app.core.redis import get_redis
from app.models import Machine, ProviderAgent
from app.services import metrics_service

logger = logging.getLogger("admin.machines")

router = APIRouter(prefix="/admin/api/machines", tags=["admin"])


class MachineCreate(BaseModel):
    uid: str = Field(min_length=1, max_length=255)
    name: str = Field(min_length=1, max_length=255)
    host: str | None = Field(default=None, max_length=255)
    dns: str | None = Field(default=None, max_length=255)
    ip: str | None = Field(default=None, max_length=255)
    total_vram_bytes: int = Field(default=0, ge=0)


class MachinePatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    host: str | None = Field(default=None, max_length=255)
    dns: str | None = Field(default=None, max_length=255)
    ip: str | None = Field(default=None, max_length=255)
    total_vram_bytes: int | None = Field(default=None, ge=0)


def machine_dict(machine: Machine, agent_count: int | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": str(machine.id),
        "uid": machine.uid,
        "name": machine.name,
        "host": machine.host,
        "dns": machine.dns,
        "ip": machine.ip,
        "total_vram_bytes": machine.total_vram_bytes,
        "hardware": machine.hardware,
        # Phase 16: shared agent registration secret (trusted LAN; the UI
        # masks it and offers copy/rotate, mirroring the definition token).
        "registration_secret": machine.registration_secret,
        "created_at": iso_utc(machine.created_at),
        "updated_at": iso_utc(machine.updated_at),
    }
    if agent_count is not None:
        d["agent_count"] = agent_count
    return d


def _agent_count(session: Session, machine_id: Any) -> int:
    return int(
        session.exec(
            select(func.count())
            .select_from(ProviderAgent)
            .where(ProviderAgent.machine_id == machine_id)
        ).one()
    )


def _get_machine(session: Session, machine_id: str) -> Machine:
    try:
        machine = session.get(Machine, uuid.UUID(machine_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="machine not found") from None
    if machine is None:
        raise HTTPException(status_code=404, detail="machine not found")
    return machine


@router.post("")
def create_machine(
    body: MachineCreate, session: Session = Depends(get_session)
) -> dict[str, Any]:
    machine = Machine(**body.model_dump())
    session.add(machine)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"machine uid or name already exists ('{body.uid}' / '{body.name}')",
        ) from None
    session.refresh(machine)
    logger.info("created machine %s (%s)", machine.uid, machine.id)
    return machine_dict(machine, agent_count=0)


@router.get("")
def list_machines(session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    machines = session.exec(select(Machine).order_by(Machine.uid)).all()
    return [machine_dict(m, agent_count=_agent_count(session, m.id)) for m in machines]


@router.get("/{machine_id}")
def get_machine(
    machine_id: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    machine = _get_machine(session, machine_id)
    return machine_dict(machine, agent_count=_agent_count(session, machine.id))


@router.get("/{machine_id}/metrics")
async def get_machine_metrics(
    machine_id: str,
    session: Session = Depends(get_session),
    redis_client: aioredis.Redis = Depends(get_redis),
) -> dict[str, Any]:
    """Merged live machine metrics (Phase 17): per-GPU union across every
    agent's partial + the owner-gated machine-wide snapshot, computed on read.
    """
    machine = _get_machine(session, machine_id)
    return await metrics_service.read_machine_metrics(redis_client, machine.uid)


@router.patch("/{machine_id}")
def patch_machine(
    machine_id: str, body: MachinePatch, session: Session = Depends(get_session)
) -> dict[str, Any]:
    machine = _get_machine(session, machine_id)
    changes = body.model_dump(exclude_unset=True)
    for key, value in changes.items():
        setattr(machine, key, value)
    machine.updated_at = datetime.now(UTC)
    session.add(machine)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(
            status_code=409, detail="machine name already exists"
        ) from None
    session.refresh(machine)
    return machine_dict(machine, agent_count=_agent_count(session, machine.id))


@router.delete("/{machine_id}")
def delete_machine(
    machine_id: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    machine = _get_machine(session, machine_id)
    count = _agent_count(session, machine.id)
    if count > 0:
        raise HTTPException(
            status_code=409,
            detail=(
                f"machine '{machine.uid}' still has {count} provider agent(s); "
                "remove the agents (or their definitions) before deleting it"
            ),
        )
    session.delete(machine)
    session.commit()
    logger.info("deleted machine %s (%s)", machine.uid, machine_id)
    return {"ok": True, "id": machine_id}


@router.post("/{machine_id}/rotate-secret")
def rotate_machine_secret(
    machine_id: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    """Mint a fresh shared registration secret for this machine (Phase 16).

    Agents keep authenticating with the OLD secret until they are redeployed
    with the new value (trusted-LAN manual rotation, mirroring the old
    per-definition token rotation). Returns the new secret in full.
    """
    machine = _get_machine(session, machine_id)
    machine.registration_secret = secrets.token_urlsafe(32)
    machine.updated_at = datetime.now(UTC)
    session.add(machine)
    session.commit()
    session.refresh(machine)
    logger.info(
        "rotated registration secret for machine %s (%s)", machine.uid, machine_id
    )
    return {
        "ok": True,
        "id": machine_id,
        "registration_secret": machine.registration_secret,
    }

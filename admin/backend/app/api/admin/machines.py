"""Admin machine CRUD (Phase 9).

``/admin/api/machines`` — unauthenticated trusted-LAN management of the
``Machine`` rows that provider instances register against (a machine must
pre-exist by ``uid``; see docs/ws-protocol.md §2).

Semantics:
- ``uid`` is immutable after creation: it is referenced by provider
  containers' ``MACHINE_UID`` env and by Redis metrics leases. Changing
  it would silently orphan instances; PATCH rejects ``uid``.
- ``DELETE`` is refused (409) while provider instances are still
  attached — delete the instances (or the definitions they belong to)
  first. No cascade: an accidental machine delete would take every
  instance row (and their FK cascades) with it.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, func, select

from app.api.admin.serializers import iso_utc
from app.core.db import get_session
from app.models import Machine, ProviderInstance

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


def machine_dict(machine: Machine, instance_count: int | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": str(machine.id),
        "uid": machine.uid,
        "name": machine.name,
        "host": machine.host,
        "dns": machine.dns,
        "ip": machine.ip,
        "total_vram_bytes": machine.total_vram_bytes,
        "hardware": machine.hardware,
        "created_at": iso_utc(machine.created_at),
        "updated_at": iso_utc(machine.updated_at),
    }
    if instance_count is not None:
        d["instance_count"] = instance_count
    return d


def _instance_count(session: Session, machine_id: Any) -> int:
    return int(
        session.exec(
            select(func.count())
            .select_from(ProviderInstance)
            .where(ProviderInstance.machine_id == machine_id)
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
    return machine_dict(machine, instance_count=0)


@router.get("")
def list_machines(session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    machines = session.exec(select(Machine).order_by(Machine.uid)).all()
    return [
        machine_dict(m, instance_count=_instance_count(session, m.id)) for m in machines
    ]


@router.get("/{machine_id}")
def get_machine(
    machine_id: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    machine = _get_machine(session, machine_id)
    return machine_dict(machine, instance_count=_instance_count(session, machine.id))


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
    return machine_dict(machine, instance_count=_instance_count(session, machine.id))


@router.delete("/{machine_id}")
def delete_machine(
    machine_id: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    machine = _get_machine(session, machine_id)
    count = _instance_count(session, machine.id)
    if count > 0:
        raise HTTPException(
            status_code=409,
            detail=(
                f"machine '{machine.uid}' still has {count} provider instance(s); "
                "remove the instances (or their definitions) before deleting it"
            ),
        )
    session.delete(machine)
    session.commit()
    logger.info("deleted machine %s (%s)", machine.uid, machine_id)
    return {"ok": True, "id": machine_id}

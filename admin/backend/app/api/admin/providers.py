"""Admin provider registration API.

POST /admin/api/providers/register

A provider container calls this at startup with its machine uid and the
registration token of the ProviderDefinition it serves. The admin validates
the binding (token -> definition -> provider_type, machine pre-created by
the operator, exact version match) and upserts the ProviderInstance row.

The per-instance WebSocket secret is returned exactly once here and stored
in Redis under ``im:ws:secret:{instance_id}`` (trusted-LAN design: the
plaintext secret deliberately does NOT go into Postgres). The provider then
dials /provider/ws with ``Authorization: Bearer <secret>``.

See docs/ws-protocol.md for the full handshake.
"""

import hashlib
import json
import logging
import secrets
from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.core.config import settings
from app.core.db import get_session
from app.core.redis import get_redis
from app.models import Machine, ProviderDefinition, ProviderInstance
from app.services import redis_keys

logger = logging.getLogger("admin.providers")

router = APIRouter(prefix="/admin/api/providers", tags=["admin"])

# Instance secrets live in Redis until the provider connects and rotates;
# a long TTL keeps orphans from accumulating forever.
SECRET_TTL_SECONDS = 30 * 24 * 3600


class RegistrationRequest(BaseModel):
    """Body sent by provider_lib.admin_client.AdminClient.register()."""

    machine_uid: str
    registration_token: str
    provider_type: str
    version: str
    port: int = 8081
    hardware: dict[str, Any] = Field(default_factory=dict)
    metrics_categories: list[str] = Field(default_factory=list)
    registered_at: str | None = None


def compute_config_fingerprint(backend_config: dict[str, Any]) -> str:
    canonical = json.dumps(backend_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _machine_dict(machine: Machine) -> dict[str, Any]:
    return {
        "id": str(machine.id),
        "uid": machine.uid,
        "name": machine.name,
        "host": machine.host,
        "dns": machine.dns,
        "ip": machine.ip,
        "total_vram_bytes": machine.total_vram_bytes,
        "hardware": machine.hardware,
    }


def _definition_dict(
    definition: ProviderDefinition, config_fingerprint: str
) -> dict[str, Any]:
    return {
        "id": str(definition.id),
        "alias": definition.alias,
        "provider_type": definition.provider_type,
        "backend_config": definition.backend_config,
        "config_fingerprint": config_fingerprint,
        "idle_timeout_seconds": definition.idle_timeout_seconds,
        "capacity": definition.capacity,
        "vram_required_bytes": definition.vram_required_bytes,
        "model_metadata": definition.model_metadata,
    }


@router.post("/register")
async def register_provider(
    body: RegistrationRequest,
    session: Session = Depends(get_session),
    redis_client: aioredis.Redis = Depends(get_redis),
) -> dict[str, Any]:
    """Register (or re-register) a provider instance against its definition."""
    # 1. Registration token -> provider definition.
    definition = session.exec(
        select(ProviderDefinition).where(
            ProviderDefinition.registration_token == body.registration_token
        )
    ).first()
    if definition is None:
        raise HTTPException(status_code=401, detail="unknown registration token")
    if not definition.enabled:
        raise HTTPException(
            status_code=403,
            detail=f"provider definition '{definition.alias}' is disabled",
        )

    # 2. Provider type must match the definition.
    if body.provider_type != definition.provider_type:
        raise HTTPException(
            status_code=409,
            detail=(
                f"provider type mismatch: container reports '{body.provider_type}', "
                f"definition '{definition.alias}' requires '{definition.provider_type}'"
            ),
        )

    # 3. Machine must be pre-registered in the UI.
    machine = session.exec(
        select(Machine).where(Machine.uid == body.machine_uid)
    ).first()
    if machine is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown machine_uid '{body.machine_uid}'",
        )

    # 4. Version gate: exact match required (trusted-LAN, lockstep deploy).
    if body.version != settings.VERSION:
        raise HTTPException(
            status_code=409,
            detail=(
                f"version mismatch: provider reports '{body.version}', "
                f"admin runs '{settings.VERSION}'"
            ),
        )

    # 5. Config fingerprint from the canonical backend_config.
    config_fingerprint = compute_config_fingerprint(definition.backend_config)

    # 6. Upsert the instance for (machine, definition).
    instance = session.exec(
        select(ProviderInstance).where(
            ProviderInstance.machine_id == machine.id,
            ProviderInstance.provider_definition_id == definition.id,
        )
    ).first()
    if instance is None:
        instance = ProviderInstance(
            machine_id=machine.id,
            provider_definition_id=definition.id,
        )
    instance.port = body.port
    instance.version = body.version
    instance.instance_status = "registering"
    instance.config_fingerprint = config_fingerprint
    session.add(instance)

    # 7. Merge the hardware report into the machine (latest report wins).
    machine.hardware = body.hardware
    reported_total = body.hardware.get("total_vram_bytes")
    if isinstance(reported_total, int):
        machine.total_vram_bytes = reported_total
    session.add(machine)
    session.commit()
    session.refresh(instance)

    # 8. Mint a fresh per-instance secret; store in Redis only (never Postgres).
    instance_secret = secrets.token_urlsafe(32)
    await redis_client.set(
        redis_keys.secret_key(str(instance.id)), instance_secret, ex=SECRET_TTL_SECONDS
    )

    # 9. Persist the declared machine-metrics categories so the metrics
    #    ownership service can read them back at WS-connect time.
    await redis_client.set(
        redis_keys.metrics_cats_key(str(instance.id)),
        json.dumps(sorted(set(body.metrics_categories))),
    )

    logger.info(
        "registered instance %s for definition %s on machine %s (port %s)",
        instance.id,
        definition.alias,
        machine.uid,
        body.port,
    )

    return {
        "instance_id": str(instance.id),
        "instance_secret": instance_secret,
        "machine": _machine_dict(machine),
        "provider_definition": _definition_dict(definition, config_fingerprint),
    }

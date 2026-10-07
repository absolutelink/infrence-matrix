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

import json
import logging
import secrets
import uuid
from typing import Any

import jsonschema
import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.core.config import settings
from app.core.db import get_session
from app.core.redis import get_redis
from app.models import (
    Machine,
    ProviderDefinition,
    ProviderInstance,
    ProviderType,
)
from app.services import redis_keys
from app.services.hashing import canonical_json_sha256

logger = logging.getLogger("admin.providers")

router = APIRouter(prefix="/admin/api/providers", tags=["admin"])

# Instance secrets live in Redis until the provider connects and rotates;
# a long TTL keeps orphans from accumulating forever.
SECRET_TTL_SECONDS = 30 * 24 * 3600


class RegistrationRequest(BaseModel):
    """Body sent by provider_lib.admin_client.AdminClient.register().

    ``schema`` (Phase 12) is the provider package's committed
    ``schema.json`` (JSON Schema 2020-12 for this type's
    ``backend_config``). The admin derives the fingerprint itself; the
    provider never sends it separately.

    TRANSITION (Phase 12): ``schema`` is optional until every provider
    package ships a real ``schema.json`` (Phase D) — an admin-first
    deploy must not 422 the currently-deployed providers that don't send
    it yet. See the schema-omitted path in ``register_provider``.
    """

    machine_uid: str
    registration_token: str
    provider_type: str
    schema: dict[str, Any] | None = None
    version: str
    port: int = 8081
    hardware: dict[str, Any] = Field(default_factory=dict)
    metrics_categories: list[str] = Field(default_factory=list)
    registered_at: str | None = None


def compute_config_fingerprint(backend_config: dict[str, Any]) -> str:
    return canonical_json_sha256(backend_config)


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
    definition: ProviderDefinition, config_fingerprint: str | None
) -> dict[str, Any]:
    # Phase 14: shells carry null backend_config/fingerprint (never the
    # hash of {}); provider_lib persists and re-reads both forms.
    authored = definition.backend_config is not None
    return {
        "id": str(definition.id),
        "alias": definition.alias,
        "provider_type": definition.provider_type,
        "backend_config": definition.backend_config if authored else None,
        "config_fingerprint": config_fingerprint,
        "idle_timeout_seconds": definition.idle_timeout_seconds,
        "capacity": definition.capacity,
        "vram_required_bytes": definition.vram_required_bytes,
        "model_metadata": definition.model_metadata,
    }


def _schema_refusal_detail(
    provider_type: str,
    committed_fingerprint: str,
    pending_fingerprint: str | None,
    voted: list[str],
    waiting_on: list[str],
    error: str,
) -> dict[str, Any]:
    """Structured 409 detail per docs/ws-protocol.md §2 (Phase 12).

    ``voted`` / ``waiting_on`` values are ProviderInstance id UUID strings.
    """
    return {
        "error": error,
        "provider_type": provider_type,
        "committed_fingerprint": committed_fingerprint,
        "pending_fingerprint": pending_fingerprint,
        "voted": voted,
        "waiting_on": waiting_on,
    }


def type_instances_of(session: Session, provider_type: str) -> list[ProviderInstance]:
    """Voter universe: every ProviderInstance row whose definition is of
    this provider type, regardless of connection state
    (docs/ws-protocol.md §2)."""
    return list(
        session.exec(
            select(ProviderInstance)
            .join(
                ProviderDefinition,
                ProviderInstance.provider_definition_id == ProviderDefinition.id,
            )
            .where(ProviderDefinition.provider_type == provider_type)
        ).all()
    )


# TRANSITION (Phase 12): committed schema recorded when a provider
# registers without presenting one (old provider packages that predate
# schema.json). Accepts any JSON object; replace when Phase D ships real
# schemas in every provider package.
PERMISSIVE_SCHEMA: dict[str, Any] = {"type": "object"}
PERMISSIVE_FINGERPRINT = canonical_json_sha256(PERMISSIVE_SCHEMA)


def _gate_existing_type(
    session: Session,
    ptype: ProviderType,
    provider_type: str,
    schema: dict[str, Any],
    instance_id: uuid.UUID,
    fp: str,
) -> dict[str, Any] | None:
    """Consensus decision for an already-registered type. Mutates ``ptype``
    in the caller's session; returns None to allow the registration or
    the structured 409 detail to refuse it."""
    sid = str(instance_id)

    if fp == ptype.schema_fingerprint:
        # L2: a match against committed resolves any stale conflict as
        # long as nothing is pending (a live pending vote is kept).
        if ptype.pending_fingerprint is None and ptype.status != "active":
            ptype.status = "active"
        return None

    universe = [str(i.id) for i in type_instances_of(session, provider_type)]

    if ptype.pending_fingerprint is None:
        waiting_on = [uid for uid in universe if uid != sid]
        if not waiting_on:
            # L1: this instance is the only member of the voter universe,
            # so its vote is unanimous — commit immediately instead of
            # staging a pending that nobody else can ever complete.
            ptype.schema = schema
            ptype.schema_fingerprint = fp
            ptype.status = "active"
            return None
        # Stage the new schema; this instance is the first voter.
        ptype.pending_schema = schema
        ptype.pending_fingerprint = fp
        ptype.pending_voters = [sid]
        ptype.status = "consensus_pending"
        return _schema_refusal_detail(
            provider_type,
            ptype.schema_fingerprint,
            fp,
            [sid],
            waiting_on,
            "schema_pending",
        )

    if fp == ptype.pending_fingerprint:
        voters = [str(v) for v in (ptype.pending_voters or [])]
        if sid not in voters:
            voters.append(sid)
        ptype.pending_voters = voters
        waiting_on = [uid for uid in universe if uid not in voters]
        if not waiting_on:
            # Consensus reached: promote pending -> committed.
            ptype.schema = ptype.pending_schema or {}
            ptype.schema_fingerprint = fp
            ptype.pending_schema = None
            ptype.pending_fingerprint = None
            ptype.pending_voters = []
            ptype.status = "active"
            return None
        return _schema_refusal_detail(
            provider_type,
            ptype.schema_fingerprint,
            fp,
            voters,
            waiting_on,
            "schema_pending",
        )

    # Third distinct fingerprint while a pending is staged: conflict.
    # The pending state is left untouched (the fleet's in-flight vote is
    # preserved for the operator to inspect / force-commit / dismiss).
    ptype.status = "conflict"
    voters = [str(v) for v in (ptype.pending_voters or [])]
    waiting_on = [uid for uid in universe if uid not in voters]
    return _schema_refusal_detail(
        provider_type,
        ptype.schema_fingerprint,
        ptype.pending_fingerprint,
        voters,
        waiting_on,
        "schema_conflict",
    )


def _reject_port_clash(
    session: Session, machine_id: object, definition_id: object, port: int
) -> None:
    """409 when another CONNECTED instance on this machine holds `port`.

    See the call site: the admin addresses instances by
    machine address + instance port, so a clash silently misroutes one
    alias's traffic into another provider's engine.
    """
    peer = session.exec(
        select(ProviderInstance, ProviderDefinition)
        .join(
            ProviderDefinition,
            ProviderInstance.provider_definition_id == ProviderDefinition.id,
        )
        .where(
            ProviderInstance.machine_id == machine_id,
            ProviderInstance.port == port,
            ProviderInstance.provider_definition_id != definition_id,
            ProviderInstance.websocket_connected == True,  # noqa: E712
        )
    ).first()
    if peer is None:
        return
    _, peer_definition = peer
    raise HTTPException(
        status_code=409,
        detail={
            "error": "port_conflict",
            "message": (
                f"machine already has a connected instance of definition "
                f"'{peer_definition.alias}' on port {port}; the admin could "
                "not tell the two apart. Give this provider a distinct "
                "PROVIDER_PORT and publish that same host port."
            ),
            "port": port,
            "conflicting_definition": peer_definition.alias,
            "conflicting_instance_id": str(peer[0].id),
        },
    )


def _schema_gate(
    session: Session,
    provider_type: str,
    schema: dict[str, Any] | None,
    instance_id: uuid.UUID,
) -> tuple[dict[str, Any] | None, str | None]:
    """Run the Phase 12 schema consensus gate.

    Mutates the ProviderType row (create/commit/stage/conflict) in the
    caller's session — the caller commits on both paths. Returns
    ``(None, reported_fingerprint)`` when registration may proceed (type
    bootstrapped, fingerprint matches committed, or this vote completed
    consensus), or ``(409 detail, reported_fingerprint)`` when the
    registration must be refused. The caller persists
    ``reported_fingerprint`` on ProviderInstance.reported_schema_fingerprint.

    TRANSITION (Phase 12): schema-omitted path; remove once all providers
    ship schema.json (Phase D). When ``schema`` is None:
    - unknown type -> bootstrap with the permissive ``{"type":
      "object"}`` schema committed (reported fp = permissive fp), so the
      type exists for definition validation;
    - known type -> the consensus gate is skipped entirely (reported fp =
      None: the instance did not report a schema, and the UI must not
      show it as on-committed when it never proved so).
    """
    if schema is None:
        ptype = session.exec(
            select(ProviderType).where(ProviderType.name == provider_type)
        ).first()
        if ptype is None:
            _bootstrap_type(session, provider_type, PERMISSIVE_SCHEMA)
            return None, PERMISSIVE_FINGERPRINT
        return None, None

    fp = canonical_json_sha256(schema)
    ptype = session.exec(
        select(ProviderType).where(ProviderType.name == provider_type)
    ).first()
    if ptype is None:
        # Bootstrap: first registration of the type commits immediately.
        # L4: a concurrent worker may create the same row between our
        # lookup and the caller's commit; flush inside a SAVEPOINT so an
        # IntegrityError costs only the insert, then re-read and apply
        # the normal consensus logic against the existing row.
        ptype = _bootstrap_type(session, provider_type, schema)
        if ptype.schema_fingerprint == fp:
            return None, fp
        return _gate_existing_type(
            session, ptype, provider_type, schema, instance_id, fp
        ), fp

    return _gate_existing_type(
        session, ptype, provider_type, schema, instance_id, fp
    ), fp


def _bootstrap_type(
    session: Session, provider_type: str, schema: dict[str, Any]
) -> ProviderType:
    """Create the ProviderType row for a first-seen type (L4: savepoint +
    IntegrityError re-read for multi-worker safety)."""
    fp = canonical_json_sha256(schema)
    try:
        with session.begin_nested():
            session.add(
                ProviderType(
                    name=provider_type,
                    schema=schema,
                    schema_fingerprint=fp,
                    status="active",
                )
            )
    except IntegrityError:
        pass  # another worker won the race; fall through to the re-read
    ptype = session.exec(
        select(ProviderType).where(ProviderType.name == provider_type)
    ).first()
    if ptype is None:  # pragma: no cover - only if the row vanished twice
        raise RuntimeError(f"provider type '{provider_type}' bootstrap failed")
    return ptype


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

    # 2. Provider type must match the definition. Phase 14: an untyped
    #    (shell) definition ADOPTS the container's reported type instead —
    #    the container is authoritative for a definition it registers
    #    against first (the registration log + type_adopted response flag
    #    make the adoption visible; an operator PATCH can repair a typo
    #    while no instance is attached).
    type_adopted = False
    if definition.provider_type is None:
        definition.provider_type = body.provider_type
        session.add(definition)
        type_adopted = True
    elif body.provider_type != definition.provider_type:
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

    # 5. Schema must parse as JSON Schema 2020-12 (docs/ws-protocol.md §2
    #    rule 6) — checked before any mutation so a malformed schema never
    #    touches the registry. An omitted schema is the transition path
    #    (see _schema_gate).
    if body.schema is not None:
        try:
            jsonschema.Draft202012Validator.check_schema(body.schema)
        except jsonschema.SchemaError as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"invalid JSON Schema (2020-12) for "
                    f"'{body.provider_type}': {exc.message}"
                ),
            ) from exc

    # 5b. Port clash: two DIFFERENT instances on one machine may not claim
    #     the same port. The admin dials litellm at
    #     http://{machine address}:{instance.port}, so a collision is not a
    #     cosmetic problem — the losing alias silently serves the other
    #     provider's model (a real deployment routed halogen-flash traffic
    #     to the llama.cpp container this way, and every /v1 request "worked"
    #     against the wrong engine). Only a CONNECTED peer holds the port:
    #     refusing on a stale row would make it impossible to bring a
    #     replacement instance online, and the presence sweep clears those
    #     within its TTL. Re-registering the same definition (the
    #     provider.initialize path) is never a clash.
    _reject_port_clash(session, machine.id, definition.id, body.port)

    # 6. Upsert the instance for (machine, definition). The instance id
    #    is assigned here (default_factory=uuid4, known pre-flush) so it
    #    can be recorded as a schema voter even when the gate refuses the
    #    registration below.
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
    # Phase 14: a shell definition has no config yet → no fingerprint
    # (None, not the hash of {}). The config push/heal flows rely on the
    # null-config distinction via models.backend_config_is_authored.
    instance.config_fingerprint = (
        compute_config_fingerprint(definition.backend_config)
        if definition.backend_config is not None
        else None
    )
    session.add(instance)
    # L3: flush so the voter-universe query inside the gate sees this row
    # even with autoflush disabled (the id itself is known pre-flush).
    session.flush()

    # 7. Merge the hardware report into the machine (latest report wins).
    #    M1: done BEFORE the gate and inside the same commit — the box is
    #    real and its hardware report is valid regardless of the schema
    #    consensus outcome, so a refused registration still refreshes it.
    machine.hardware = body.hardware
    reported_total = body.hardware.get("total_vram_bytes")
    if isinstance(reported_total, int):
        machine.total_vram_bytes = reported_total
    session.add(machine)

    # 8. Schema consensus gate (Phase 12). Runs before the secret mint: a
    #    refusal must not complete the registration. All DB side effects
    #    (instance upsert, hardware merge, gate state, reported
    #    fingerprint) commit atomically here, on BOTH the accept and the
    #    refuse path.
    refusal, reported_schema_fingerprint = _schema_gate(
        session, body.provider_type, body.schema, instance.id
    )
    # Reported fingerprint is written on EVERY attempt (success or 409)
    # so the UI can show who is on what schema (None when the provider
    # omitted its schema — transition path).
    instance.reported_schema_fingerprint = reported_schema_fingerprint
    session.add(instance)
    # Single atomic commit of the whole registration side-effect set.
    session.commit()
    session.refresh(instance)

    if refusal is not None:
        logger.info(
            "registration refused by schema gate for instance %s (type %s): %s",
            instance.id,
            body.provider_type,
            refusal["error"],
        )
        raise HTTPException(status_code=409, detail=refusal)

    # 9. Mint a fresh per-instance secret; store in Redis only (never Postgres).
    config_fingerprint = instance.config_fingerprint
    instance_secret = secrets.token_urlsafe(32)
    await redis_client.set(
        redis_keys.secret_key(str(instance.id)), instance_secret, ex=SECRET_TTL_SECONDS
    )

    # 10. Persist the declared machine-metrics categories so the metrics
    #     ownership service can read them back at WS-connect time.
    await redis_client.set(
        redis_keys.metrics_cats_key(str(instance.id)),
        json.dumps(sorted(set(body.metrics_categories))),
    )

    logger.info(
        "registered instance %s for definition %s on machine %s (port %s)%s",
        instance.id,
        definition.alias,
        machine.uid,
        body.port,
        f" (adopted provider type '{body.provider_type}')" if type_adopted else "",
    )

    return {
        "instance_id": str(instance.id),
        "instance_secret": instance_secret,
        "machine": _machine_dict(machine),
        "provider_definition": _definition_dict(definition, config_fingerprint),
        # Phase 14: set when THIS registration set the definition's
        # provider_type from the container's report (shell → typed).
        "type_adopted": type_adopted,
    }

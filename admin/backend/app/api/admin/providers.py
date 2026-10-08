"""Admin provider **agent** registration API (Phase 16).

POST /admin/api/providers/register

A provider agent container calls this at startup. It authenticates with the
shared ``Machine.registration_secret`` (no per-definition token any more) and
identifies itself as a ``(machine_uid, provider_type, agent_id)`` triple. The
admin:

  * validates the machine secret and the exact version match,
  * upserts the ``ProviderAgent`` row (container-level state),
  * runs the Phase 12 schema-consensus gate against the fleet of agents of
    this type,
  * resolves the definitions *placed* on this agent (``agent_placement``:
    every enabled definition of the type for ``any_of_type``, or the
    cherry-picked set for ``specific``) and upserts one ``ProviderInstance``
    per placed definition, assigning each a port ``base_port + offset``,
  * mints a single per-agent WebSocket secret (stored in Redis under
    ``im:ws:secret:{agent_id}``) that authorizes the one socket the agent
    multiplexes all its backends over.

The response returns the agent id/secret plus the list of backends (instance
id + definition + port) the agent is responsible for. See docs/ws-protocol.md.
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
    ProviderAgent,
    ProviderDefinition,
    ProviderInstance,
    ProviderType,
)
from app.services import redis_keys
from app.services.assignments import reconcile_agent_placement
from app.services.hardware import merge_hardware_union, normalize_gpu_uuids
from app.services.hashing import canonical_json_sha256

logger = logging.getLogger("admin.providers")

router = APIRouter(prefix="/admin/api/providers", tags=["admin"])

# Agent secrets live in Redis until the agent connects and rotates; a long TTL
# keeps orphans from accumulating forever.
SECRET_TTL_SECONDS = 30 * 24 * 3600


class RegistrationRequest(BaseModel):
    """Body sent by provider_lib.admin_client.AdminClient.register().

    ``schema`` (Phase 12) is the provider package's committed ``schema.json``
    (JSON Schema 2020-12 for this type's ``backend_config``). The admin
    derives the fingerprint itself; the provider never sends it separately.

    Phase 16: ``machine_secret`` replaces ``registration_token``; ``agent_id``
    is the operator-supplied stable id (container ``AGENT_ID`` env) and
    ``base_port`` is the first backend port (backends get ``base_port + i``).
    """

    machine_uid: str = Field(max_length=255)
    machine_secret: str
    agent_id: str = Field(max_length=255)
    provider_type: str = Field(max_length=64)
    schema: dict[str, Any] | None = None
    version: str
    base_port: int = 8081
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
    return {
        "id": str(definition.id),
        "alias": definition.alias,
        "provider_type": definition.provider_type,
        "modality": definition.modality,
        "backend_config": definition.backend_config,
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

    ``voted`` / ``waiting_on`` values are ProviderAgent id UUID strings.
    """
    return {
        "error": error,
        "provider_type": provider_type,
        "committed_fingerprint": committed_fingerprint,
        "pending_fingerprint": pending_fingerprint,
        "voted": voted,
        "waiting_on": waiting_on,
    }


def type_agents_of(session: Session, provider_type: str) -> list[ProviderAgent]:
    """Voter universe: every ProviderAgent row of this provider type,
    regardless of connection state (docs/ws-protocol.md §2)."""
    return list(
        session.exec(
            select(ProviderAgent).where(ProviderAgent.provider_type == provider_type)
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
    agent_id: uuid.UUID,
    fp: str,
) -> dict[str, Any] | None:
    """Consensus decision for an already-registered type. Mutates ``ptype``
    in the caller's session; returns None to allow the registration or
    the structured 409 detail to refuse it."""
    sid = str(agent_id)

    if fp == ptype.schema_fingerprint:
        # L2: a match against committed resolves any stale conflict as
        # long as nothing is pending (a live pending vote is kept).
        if ptype.pending_fingerprint is None and ptype.status != "active":
            ptype.status = "active"
        return None

    universe = [str(a.id) for a in type_agents_of(session, provider_type)]

    if ptype.pending_fingerprint is None:
        waiting_on = [uid for uid in universe if uid != sid]
        if not waiting_on:
            # L1: this agent is the only member of the voter universe, so
            # its vote is unanimous — commit immediately instead of staging
            # a pending that nobody else can ever complete.
            ptype.schema = schema
            ptype.schema_fingerprint = fp
            ptype.status = "active"
            _apply_max_running_from_schema(ptype, schema)
            _apply_serves_modalities_from_schema(ptype, schema)
            return None
        # Stage the new schema; this agent is the first voter.
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
            _apply_max_running_from_schema(ptype, ptype.schema)
            _apply_serves_modalities_from_schema(ptype, ptype.schema)
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


def _schema_gate(
    session: Session,
    provider_type: str,
    schema: dict[str, Any] | None,
    agent_id: uuid.UUID,
) -> tuple[dict[str, Any] | None, str | None]:
    """Run the Phase 12 schema consensus gate (voters are agents).

    Mutates the ProviderType row (create/commit/stage/conflict) in the
    caller's session — the caller commits on both paths. Returns
    ``(None, reported_fingerprint)`` when registration may proceed, or
    ``(409 detail, reported_fingerprint)`` when it must be refused. The
    caller persists ``reported_fingerprint`` on
    ``ProviderAgent.reported_schema_fingerprint``.
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
            session, ptype, provider_type, schema, agent_id, fp
        ), fp

    return _gate_existing_type(session, ptype, provider_type, schema, agent_id, fp), fp


def _apply_max_running_from_schema(ptype: ProviderType, schema: dict[str, Any]) -> None:
    """Sync ``ProviderType.max_running_backends`` from a committed schema.

    Phase 16 slice 4: the per-agent running cap is declared as a top-level
    ``x-max-running-backends`` key in the type's shipped ``schema.json``
    (default unlimited = 0). The scheduler reads this column to enforce the
    cap (ARCHITECTURE.md §4/§6). Mutates ``ptype`` in the caller's session;
    the caller commits.
    """
    raw = schema.get("x-max-running-backends", 0) if isinstance(schema, dict) else 0
    try:
        ptype.max_running_backends = max(0, int(raw))
    # PEP 758 (Python 3.14): unparenthesized except group is valid; ruff format
    # (target py314) canonicalizes it this way. Not a Python-2 syntax error.
    except TypeError, ValueError:
        ptype.max_running_backends = 0


_KNOWN_MODALITIES = ("llm", "embedding")


def _apply_serves_modalities_from_schema(
    ptype: ProviderType, schema: dict[str, Any]
) -> None:
    """Sync ``ProviderType.serves_modalities`` from a committed schema.

    Phase 18 slice 1: the endpoint kinds a type can host are declared as a
    top-level ``x-serves-modalities`` key (a JSON array of strings) in the
    type's shipped ``schema.json`` (default ``["llm"]``) — the exact mechanism
    already used for ``x-max-running-backends``. Unknown values are dropped;
    if nothing known remains, fall back to ``["llm"]``. The definition CRUD
    validates a definition's ``modality`` against this column (422 otherwise).
    Mutates ``ptype`` in the caller's session; the caller commits.
    """
    raw = (
        schema.get("x-serves-modalities", ["llm"])
        if isinstance(schema, dict)
        else ["llm"]
    )
    if not isinstance(raw, list):
        raw = ["llm"]
    # Dedup while preserving order (e.g. ["llm", "llm"] -> ["llm"]); drop
    # unknown values; fall back to ["llm"] if nothing known remains.
    known: list[str] = []
    for m in raw:
        value = str(m)
        if value in _KNOWN_MODALITIES and value not in known:
            known.append(value)
    ptype.serves_modalities = known or ["llm"]


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
    _apply_max_running_from_schema(ptype, schema)
    _apply_serves_modalities_from_schema(ptype, schema)
    session.add(ptype)
    return ptype


def _placed_definitions(
    session: Session, agent: ProviderAgent
) -> list[ProviderDefinition]:
    """Definitions this agent is responsible for, per placement policy
    (ARCHITECTURE.md §5).

    The placed set is the UNION of:

    * every enabled ``any_of_type`` definition whose ``provider_type``
      matches this agent's type — such a definition targets every agent of
      the type implicitly; and
    * every enabled ``specific`` definition explicitly linked to THIS agent
      via ``definition_agents``.

    A ``specific`` definition placed on some OTHER agent must never leak onto
    this one, and an ``any_of_type`` definition must still apply even when the
    agent also carries ``specific`` links. Sorted by alias so port offsets are
    stable across registrations.
    """
    placed: dict[uuid.UUID, ProviderDefinition] = {}
    # any_of_type defs of this agent's provider_type.
    for d in session.exec(
        select(ProviderDefinition).where(
            ProviderDefinition.provider_type == agent.provider_type,
            ProviderDefinition.agent_placement == "any_of_type",
            ProviderDefinition.enabled == True,  # noqa: E712
        )
    ).all():
        placed[d.id] = d
    # specific defs explicitly linked to this agent.
    for d in agent.definitions:
        if d.enabled and d.agent_placement == "specific":
            placed[d.id] = d
    return sorted(placed.values(), key=lambda d: d.alias)


def prune_unplaced_backends(
    session: Session, agent: ProviderAgent, placed: list[ProviderDefinition]
) -> int:
    """Delete ``ProviderInstance`` rows this agent hosts that are no longer in
    its resolved placement (M2).

    Slice 5: this is now a thin wrapper over the shared
    :func:`app.services.assignments.reconcile_agent_placement` diff (the same
    path the live ``agent.assignments.update`` push uses), so there is exactly
    ONE placement-diff implementation. It runs the prune half only
    (``create_new=False``): a still-``running``/``in_use`` backend is NOT
    deleted (busy-safe — its engine holds VRAM and the scheduler's
    per-booted hold is keyed by the instance id), and is left for a later
    registration/PATCH to retire once it stops. Returns the number of rows
    deleted.
    """
    diff = reconcile_agent_placement(
        session, agent, placed, create_new=False, renumber_ports=False
    )
    return len(diff.removed)


def _reject_port_clash(
    session: Session, agent: ProviderAgent, ports: list[int]
) -> None:
    """409 when a CONNECTED instance on another agent of this machine already
    holds one of the ports this agent is about to claim.

    The admin dials litellm at ``http://{machine address}:{instance.port}``,
    so a collision silently misroutes one alias's traffic into another
    provider's engine. Only a connected peer holds the port: refusing on a
    stale row would make it impossible to bring a replacement online, and the
    presence sweep clears those within its TTL.
    """
    if not ports:
        return
    peer = session.exec(
        select(ProviderInstance, ProviderDefinition)
        .join(
            ProviderDefinition,
            ProviderInstance.provider_definition_id == ProviderDefinition.id,
        )
        .join(ProviderAgent, ProviderInstance.agent_id == ProviderAgent.id)
        .where(
            ProviderAgent.machine_id == agent.machine_id,
            ProviderInstance.agent_id != agent.id,
            ProviderInstance.port.in_(ports),  # type: ignore[attr-defined]
            ProviderAgent.websocket_connected == True,  # noqa: E712
        )
    ).first()
    if peer is None:
        return
    inst, peer_definition = peer
    raise HTTPException(
        status_code=409,
        detail={
            "error": "port_conflict",
            "message": (
                f"machine already has a connected backend of definition "
                f"'{peer_definition.alias}' on port {inst.port}; the admin "
                "could not tell the two apart. Give this agent a distinct "
                "PROVIDER_PORT/base_port and publish that same host port."
            ),
            "port": inst.port,
            "conflicting_definition": peer_definition.alias,
            "conflicting_instance_id": str(inst.id),
        },
    )


@router.post("/register")
async def register_provider(
    body: RegistrationRequest,
    session: Session = Depends(get_session),
    redis_client: aioredis.Redis = Depends(get_redis),
) -> dict[str, Any]:
    """Register (or re-register) a provider agent against its machine."""
    # 1. Machine must be pre-registered and the shared secret must match.
    machine = session.exec(
        select(Machine).where(Machine.uid == body.machine_uid)
    ).first()
    if machine is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown machine_uid '{body.machine_uid}'",
        )
    if not machine.registration_secret or not secrets.compare_digest(
        machine.registration_secret, body.machine_secret
    ):
        raise HTTPException(status_code=401, detail="invalid machine secret")

    # 2. Version gate: exact match required (trusted-LAN, lockstep deploy).
    if body.version != settings.VERSION:
        raise HTTPException(
            status_code=409,
            detail=(
                f"version mismatch: provider reports '{body.version}', "
                f"admin runs '{settings.VERSION}'"
            ),
        )

    # 3. Schema must parse as JSON Schema 2020-12 (docs/ws-protocol.md §2
    #    rule 6) — checked before any mutation so a malformed schema never
    #    touches the registry. An omitted schema is the transition path.
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

    # 4. Upsert the ProviderAgent for (machine, provider_type, agent_id). The
    #    id is assigned here (default_factory=uuid4, known pre-flush) so it can
    #    be recorded as a schema voter even when the gate refuses below.
    agent = session.exec(
        select(ProviderAgent).where(
            ProviderAgent.machine_id == machine.id,
            ProviderAgent.provider_type == body.provider_type,
            ProviderAgent.agent_id == body.agent_id,
        )
    ).first()
    if agent is None:
        agent = ProviderAgent(
            machine_id=machine.id,
            provider_type=body.provider_type,
            agent_id=body.agent_id,
        )
    # Prior attribution (uuid strings) is captured BEFORE overwrite so the
    # hardware union below can drop GPUs this agent stopped reporting.
    prior_assigned_uuids = normalize_gpu_uuids(agent.assigned_gpus)
    agent.base_port = body.base_port
    agent.version = body.version
    agent.agent_status = "registering"
    # assigned_gpus is a list of GPU uuid strings (ARCHITECTURE.md §4):
    # normalize the structured report down to uuids ([] when no gpus list).
    agent.assigned_gpus = normalize_gpu_uuids(body.hardware.get("gpus"))
    session.add(agent)
    # L3: flush so the voter-universe query inside the gate sees this row
    # even with autoflush disabled (the id itself is known pre-flush).
    session.flush()

    # 5. Merge the hardware report into the machine as a per-GPU UNION
    #    (ARCHITECTURE.md §4, Phase 17 B): GPUs are unioned by uuid across every
    #    agent on the box, machine.total_vram_bytes is the auto-sum of the union
    #    (the scheduler's admission budget), and cpu/ram stay last-writer-wins.
    #    Done BEFORE the gate and inside the same commit — the box is real and its
    #    hardware report is valid regardless of the schema consensus outcome, so a
    #    refused registration still refreshes it. A UNION (not a replace) so a
    #    partial report from one agent never erases GPUs another contributed.
    #
    #    Survivor-aware stale-drop (symmetric with the DELETE path): GPUs claimed
    #    by any *other* agent on this machine are never dropped by this agent's
    #    re-registration, even if this agent previously reported them too.
    other_agent_uuids: set[str] = set()
    for other in session.exec(
        select(ProviderAgent).where(
            ProviderAgent.machine_id == machine.id,
            ProviderAgent.id != agent.id,
        )
    ).all():
        other_agent_uuids.update(normalize_gpu_uuids(other.assigned_gpus))
    merged_hardware, new_total = merge_hardware_union(
        machine.hardware, body.hardware, prior_assigned_uuids, sorted(other_agent_uuids)
    )
    machine.hardware = merged_hardware
    if new_total is not None:
        machine.total_vram_bytes = new_total
    session.add(machine)

    # 6. Schema consensus gate (Phase 12). Runs before the secret mint: a
    #    refusal must not complete the registration. All DB side effects
    #    (agent upsert, hardware merge, gate state, reported fingerprint)
    #    commit atomically here, on BOTH the accept and the refuse path.
    refusal, reported_schema_fingerprint = _schema_gate(
        session, body.provider_type, body.schema, agent.id
    )
    agent.reported_schema_fingerprint = reported_schema_fingerprint
    session.add(agent)
    session.commit()
    session.refresh(agent)

    if refusal is not None:
        logger.info(
            "registration refused by schema gate for agent %s (type %s): %s",
            agent.id,
            body.provider_type,
            refusal["error"],
        )
        raise HTTPException(status_code=409, detail=refusal)

    # 7. Resolve placed definitions and reconcile one ProviderInstance each
    #    (create stopped rows at ``base_port + sorted-alias offset``, refresh
    #    fingerprints, and retire de-placed ghosts busy-safe). Slice 5: this is
    #    the SAME shared diff the live ``agent.assignments.update`` push runs,
    #    so registration and placement-change reconciliation never diverge.
    placed = _placed_definitions(session, agent)
    ports = [agent.base_port + i for i in range(len(placed))]
    _reject_port_clash(session, agent, ports)
    diff = reconcile_agent_placement(
        session, agent, placed, create_new=True, renumber_ports=True
    )
    pruned = len(diff.removed)

    # Build the response backends list from the reconciled rows, preserving the
    # registration response shape (instance id + port + definition echo).
    rows = {
        inst.provider_definition_id: inst
        for inst in session.exec(
            select(ProviderInstance).where(ProviderInstance.agent_id == agent.id)
        ).all()
    }
    backends: list[dict[str, Any]] = [
        {
            "instance_id": str(rows[d.id].id),
            "port": rows[d.id].port,
            "definition": _definition_dict(d, rows[d.id].config_fingerprint),
        }
        for d in placed
        if d.id in rows
    ]
    session.commit()

    # 8. Mint a fresh per-agent secret; store in Redis only (never Postgres).
    agent_secret = secrets.token_urlsafe(32)
    await redis_client.set(
        redis_keys.secret_key(str(agent.id)), agent_secret, ex=SECRET_TTL_SECONDS
    )

    # 9. Persist the declared machine-metrics categories so the metrics
    #    ownership service can read them back at WS-connect time.
    await redis_client.set(
        redis_keys.metrics_cats_key(str(agent.id)),
        json.dumps(sorted(set(body.metrics_categories))),
    )

    logger.info(
        "registered agent %s (type %s) on machine %s: %d backend(s), base_port %s"
        + (", pruned %d ghost(s)" if pruned else ""),
        agent.id,
        body.provider_type,
        machine.uid,
        len(backends),
        agent.base_port,
        *([pruned] if pruned else []),
    )

    return {
        "agent_id": str(agent.id),
        "agent_secret": agent_secret,
        "machine": _machine_dict(machine),
        "backends": backends,
    }

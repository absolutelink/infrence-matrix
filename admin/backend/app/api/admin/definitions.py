"""Admin provider-definition CRUD (Phase 9).

``/admin/api/definitions`` — unauthenticated trusted-LAN management of
``ProviderDefinition`` rows (the client-facing "models"). Registration
(POST /admin/api/providers/register) stays where it is; this is the
operator-side create/update/delete the UI (Phase 10) consumes.

Validation boundary (provider/README.md): the admin validates only the
fields **it owns** — non-empty ``alias``/``provider_type``,
``provider_type`` present in the ``ProviderType`` registry (Phase 12),
``capacity >= 1``, ``vram_required_bytes >= 0``,
``idle_timeout_seconds >= 0``, and that ``backend_config`` is a
JSON-object that round-trips through the canonical fingerprint
serializer **and validates against the type's committed JSON Schema
(2020-12)** — per-field errors are returned as a structured 422.
Runtime failure of provider-specific interpretation still surfaces
through the config.update NAK.

Config-change semantics (docs/ws-protocol.md §4):
- A PATCH that changes what the **provider consumes** — ``backend_config``
  (fingerprint differs) or ``capacity`` (enforced at the provider) —
  pushes ``provider.config.update`` to every **websocket-connected**
  instance of the definition and returns the per-instance results in
  the PATCH response body (``config_update_results``). A failed
  provider update does NOT roll back the admin row — it is visible, not
  fatal. A capacity-only change is adopted by the provider without a
  backend restart (see ``provider_lib.config_update``).
  ``idle_timeout_seconds`` is NOT a push trigger: the idle reaper is
  admin-side only (``InferenceScheduler._idle_reaper``); the provider
  carries no idle behavior, so an idle-only PATCH stays admin-side and
  takes effect on the next reaper tick.
- Other non-config PATCHes (idle timeout, enabled, alias, ...) are
  admin-side only: no provider push. ``enabled=false`` excludes the
  definition from scheduling (the scheduler query); providers keep
  running until stopped by other means.
- ``provider_type`` is refused (409) while any instance is attached to
  the definition: existing provider containers are built for the old
  type and would keep running under a definition they no longer match
  (registration cross-checks provider_type, so the binding would be
  silently broken for the next reconnect).
- ``DELETE`` is allowed only when **no attached backend's agent is
  currently websocket_connected**; otherwise 409 with a suggestion to
  disable. When allowed, the definition's *offline* backend rows are
  deleted with it (cascade to non-connected rows only).

Phase 16: ``provider_type`` and ``backend_config`` are required at create
(the Phase 14 shell/``awaiting_config`` machinery is retired) and
``registration_token`` is gone — agents authenticate with the shared
``Machine.registration_secret`` and receive every definition placed on them
(``agent_placement``).
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import jsonschema
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, col, select

from app.api.admin.providers import compute_config_fingerprint
from app.api.admin.serializers import iso_utc
from app.core.db import get_session
from app.models import (
    DefinitionAgent,
    ProviderAgent,
    ProviderDefinition,
    ProviderInstance,
    ProviderType,
)
from app.services import alias_registry, assignments, config_update

logger = logging.getLogger("admin.definitions")

router = APIRouter(prefix="/admin/api/definitions", tags=["admin"])

_VALID_PLACEMENTS = ("any_of_type", "specific")

# Phase 18: modalities the admin accepts. Phase 24 adds the two concrete
# speech modalities (`tts`, `asr`); the Phase 18 reserved `audio` bucket was
# never valid and stays retired. Audio aliases are NOT litellm-registered
# (the /v1/audio/* routes bypass litellm with direct httpx passthrough).
_VALID_MODALITIES = ("llm", "embedding", "tts", "asr")


class DefinitionCreate(BaseModel):
    alias: str = Field(min_length=1, max_length=255)
    # Phase 16: required (no shells). provider_type must be in the registry
    # and backend_config validates against its committed JSON Schema.
    provider_type: str = Field(min_length=1, max_length=64)
    # Phase 18: endpoint kind this definition serves. Must be in the type's
    # serves_modalities list (create/PATCH 422 otherwise).
    modality: str = Field(default="llm", max_length=32)
    backend_config: dict[str, Any] = Field(default_factory=dict)
    vram_required_bytes: int = Field(default=0, ge=0)
    idle_timeout_seconds: int = Field(default=300, ge=0)
    capacity: int = Field(default=1, ge=1)
    enabled: bool = True
    # Phase 16: placement. 'any_of_type' hosts on every agent of the type;
    # 'specific' hosts only on the listed agent ids (ProviderAgent.id).
    agent_placement: str = Field(default="any_of_type", max_length=32)
    agents: list[str] | None = None


class DefinitionPatch(BaseModel):
    alias: str | None = Field(default=None, min_length=1, max_length=255)
    provider_type: str | None = Field(default=None, min_length=1, max_length=64)
    # Phase 18: endpoint kind (see DefinitionCreate). Immutable while backends
    # are attached (409), same gate as provider_type.
    modality: str | None = Field(default=None, max_length=32)
    backend_config: dict[str, Any] | None = None
    vram_required_bytes: int | None = Field(default=None, ge=0)
    idle_timeout_seconds: int | None = Field(default=None, ge=0)
    capacity: int | None = Field(default=None, ge=1)
    enabled: bool | None = None
    # Phase 16: placement (see DefinitionCreate).
    agent_placement: str | None = Field(default=None, max_length=32)
    agents: list[str] | None = None


# Columns that are NOT NULL in the model: an explicit null in a PATCH
# body is a client error, never a silent skip or a DB IntegrityError.
# Phase 16: provider_type/backend_config are NOT NULL again (no shells).
_NON_NULLABLE_FIELDS = frozenset(
    {
        "alias",
        "provider_type",
        "modality",
        "backend_config",
        "vram_required_bytes",
        "idle_timeout_seconds",
        "capacity",
        "enabled",
        "agent_placement",
    }
)


def _ensure_litellm_registered(definition: ProviderDefinition) -> None:
    """Warm the litellm alias registration for litellm-backed modalities only.

    ``llm`` -> ``chat`` mode, ``embedding`` -> ``embedding`` mode. The Phase 24
    audio modalities (``tts``/``asr``) are served by direct httpx passthrough
    (litellm is bypassed — see ARCHITECTURE.md §7 Audio specifics), so their
    aliases must NEVER be registered with litellm: a stale ``chat`` entry would
    mis-declare them and the fake-stream APIError this registry exists to
    prevent would surface on the first (nonexistent) litellm call.
    """
    if definition.modality == "embedding":
        alias_registry.ensure_registered(definition.alias, mode="embedding")
    elif definition.modality == "llm":
        alias_registry.ensure_registered(definition.alias, mode="chat")


def _get_provider_type(session: Session, provider_type: str) -> ProviderType:
    """Resolve a provider_type against the Phase 12 registry (the source
    of truth; the old PROVIDER_TYPES constant is gone). 422 with the
    registered-type list when unknown."""
    ptype = session.exec(
        select(ProviderType).where(ProviderType.name == provider_type)
    ).first()
    if ptype is None:
        known = [
            t.name
            for t in session.exec(
                select(ProviderType).order_by(ProviderType.name)
            ).all()
        ]
        known_desc = ", ".join(known) if known else "(none registered yet)"
        raise HTTPException(
            status_code=422,
            detail=(
                f"unknown provider_type '{provider_type}'; "
                f"registered types: {known_desc}"
            ),
        )
    return ptype


def _validate_backend_config(
    ptype: ProviderType, backend_config: dict[str, Any]
) -> None:
    """Validate a backend_config against the type's committed JSON Schema
    (Phase 12). The caller resolves the ProviderType row once via
    ``_get_provider_type`` and passes it here (no per-validator query)."""
    # Must survive the canonical fingerprint serializer (rejects
    # non-JSON-serializable values sneaking in via odd Pydantic input).
    try:
        compute_config_fingerprint(backend_config)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"backend_config is not JSON-serializable: {exc}",
        ) from exc
    # Phase 12: validate against the type's committed JSON Schema.
    errors = sorted(
        jsonschema.Draft202012Validator(ptype.schema or {}).iter_errors(backend_config),
        # H1: str() each path part — absolute_path mixes str keys and int
        # array indices, which are not order-comparable.
        key=lambda e: [str(p) for p in e.absolute_path],
    )
    if errors:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "backend_config_schema_validation_failed",
                "provider_type": ptype.name,
                "schema_fingerprint": ptype.schema_fingerprint,
                "errors": [
                    {
                        "path": "$"
                        + "".join(
                            f"[{part!r}]" if isinstance(part, int) else f".{part}"
                            for part in err.absolute_path
                        ),
                        "message": err.message,
                    }
                    for err in errors
                ],
            },
        )


def _validate_modality(ptype: ProviderType, modality: str) -> None:
    """Phase 18/24: a definition's modality must be a known value AND be hosted
    by its provider type. Refuses 422 when ``modality`` is not in
    ``{"llm", "embedding", "tts", "asr"}`` (the reserved ``audio`` bucket is
    not accepted) or not in ``ptype.serves_modalities``."""
    if modality not in _VALID_MODALITIES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"modality '{modality}' is not accepted; "
                f"valid values: {list(_VALID_MODALITIES)}"
            ),
        )
    if modality not in (ptype.serves_modalities or []):
        raise HTTPException(
            status_code=422,
            detail=(
                f"provider_type '{ptype.name}' does not serve modality "
                f"'{modality}'; it serves {ptype.serves_modalities}"
            ),
        )


def _resolve_placement_agents(
    session: Session, placement: str, agents: list[str] | None, provider_type: str
) -> list[uuid.UUID]:
    """Validate ``agent_placement`` and resolve ``specific`` agent ids.

    Returns the list of ``ProviderAgent`` ids to link (empty for
    ``any_of_type``). Raises 422 on an unknown placement, an empty/absent
    list for ``specific``, an unknown/malformed agent id, or an agent whose
    ``provider_type`` does not match the definition's (M1: a llama-cpp agent
    cannot host a gufo definition).
    """
    if placement not in _VALID_PLACEMENTS:
        raise HTTPException(
            status_code=422,
            detail=f"agent_placement must be one of {list(_VALID_PLACEMENTS)}",
        )
    if placement != "specific":
        if agents:
            raise HTTPException(
                status_code=422,
                detail="'agents' is only valid with agent_placement='specific'",
            )
        return []
    if not agents:
        raise HTTPException(
            status_code=422,
            detail="agent_placement='specific' requires a non-empty 'agents' list",
        )
    out: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for raw in agents:
        try:
            agent_uuid = uuid.UUID(raw)
        except ValueError:
            raise HTTPException(
                status_code=422, detail=f"invalid agent id '{raw}'"
            ) from None
        agent = session.get(ProviderAgent, agent_uuid)
        if agent is None:
            raise HTTPException(status_code=422, detail=f"unknown agent '{raw}'")
        if agent.provider_type != provider_type:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"agent '{raw}' has provider_type '{agent.provider_type}', "
                    f"which does not match the definition's '{provider_type}'"
                ),
            )
        if agent_uuid not in seen:
            seen.add(agent_uuid)
            out.append(agent_uuid)
    return out


def _write_placement(
    session: Session, definition: ProviderDefinition, agent_ids: list[uuid.UUID]
) -> None:
    """Replace the definition's specific-placement link rows."""
    existing = session.exec(
        select(DefinitionAgent).where(
            DefinitionAgent.provider_definition_id == definition.id
        )
    ).all()
    for row in existing:
        session.delete(row)
    for agent_id in agent_ids:
        session.add(
            DefinitionAgent(provider_definition_id=definition.id, agent_id=agent_id)
        )


def _placement_agent_ids(session: Session, definition: ProviderDefinition) -> list[str]:
    return [
        str(row.agent_id)
        for row in session.exec(
            select(DefinitionAgent).where(
                DefinitionAgent.provider_definition_id == definition.id
            )
        ).all()
    ]


def definition_dict(
    definition: ProviderDefinition,
    *,
    instances: list[ProviderInstance] | None = None,
    config_update_results: list[dict[str, Any]] | None = None,
    placement_agents: list[str] | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": str(definition.id),
        "alias": definition.alias,
        "provider_type": definition.provider_type,
        "modality": definition.modality,
        "backend_config": definition.backend_config,
        "config_fingerprint": compute_config_fingerprint(definition.backend_config),
        "vram_required_bytes": definition.vram_required_bytes,
        "idle_timeout_seconds": definition.idle_timeout_seconds,
        "capacity": definition.capacity,
        "model_metadata": definition.model_metadata,
        "enabled": definition.enabled,
        "status": definition.status,
        # Phase 16: placement.
        "agent_placement": definition.agent_placement,
        "created_at": iso_utc(definition.created_at),
        "updated_at": iso_utc(definition.updated_at),
    }
    if placement_agents is not None:
        d["agents"] = placement_agents
    if instances is not None:
        d["instances"] = [
            {
                "id": str(i.id),
                "agent_id": str(i.agent_id),
                "machine_uid": i.machine.uid if i.machine else None,
                "agent_status": i.agent.agent_status if i.agent else None,
                "backend_status": i.backend_status,
                "websocket_connected": (
                    i.agent.websocket_connected if i.agent else False
                ),
                "config_fingerprint": i.config_fingerprint,
            }
            for i in instances
        ]
        d["connected_instance_count"] = sum(
            1 for i in instances if i.agent is not None and i.agent.websocket_connected
        )
    if config_update_results is not None:
        d["config_update_results"] = config_update_results
    return d


def _get_definition(session: Session, definition_id: str) -> ProviderDefinition:
    try:
        definition = session.get(ProviderDefinition, uuid.UUID(definition_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="definition not found") from None
    if definition is None:
        raise HTTPException(status_code=404, detail="definition not found")
    return definition


@router.post("")
async def create_definition(
    body: DefinitionCreate,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    # Phase 16: provider_type + backend_config are required; validate the
    # config against the type's committed schema before inserting.
    ptype = _get_provider_type(session, body.provider_type)
    _validate_backend_config(ptype, body.backend_config)
    # Phase 18: the modality must be accepted and hosted by the type.
    _validate_modality(ptype, body.modality)
    # Phase 16: validate placement + resolve specific agent ids (422 on an
    # unknown agent) BEFORE inserting, so a bad placement never persists.
    agent_ids = _resolve_placement_agents(
        session, body.agent_placement, body.agents, body.provider_type
    )
    definition = ProviderDefinition(
        **body.model_dump(exclude={"agents"}),
    )
    session.add(definition)
    try:
        # Flush so the definition id exists for the link FK, then write the
        # placement links in the SAME transaction (atomic: never a
        # 'specific' definition with zero links from a mid-way failure).
        session.flush()
        if agent_ids:
            _write_placement(session, definition, agent_ids)
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"alias already exists ('{body.alias}')",
        ) from None
    session.refresh(definition)
    # Warm the litellm alias registration before first use (checklist 4).
    # Audio modalities are bypassed (direct httpx passthrough — see
    # ``_ensure_litellm_registered``).
    _ensure_litellm_registered(definition)
    # Phase 16 slice 5: a new definition is a placement change for every agent
    # of its type — push the updated assignment set to the live ones (an
    # any_of_type def now includes them; a specific def reaches only its links,
    # the others reconcile to a no-op).
    await assignments.push_agent_assignments_for_type(
        definition.provider_type,
        scheduler=getattr(request.app.state, "scheduler", None),
    )
    logger.info("created definition %s (%s)", definition.alias, definition.id)
    return definition_dict(
        definition, instances=[], placement_agents=[str(a) for a in agent_ids]
    )


@router.get("")
def list_definitions(session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    definitions = session.exec(
        select(ProviderDefinition).order_by(ProviderDefinition.alias)
    ).all()
    # `instances` is selectin-loaded on the relationship — no per-row SELECT.
    return [
        definition_dict(
            d,
            instances=list(d.instances),
            placement_agents=_placement_agent_ids(session, d),
        )
        for d in definitions
    ]


@router.get("/{definition_id}")
def get_definition(
    definition_id: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    definition = _get_definition(session, definition_id)
    return definition_dict(
        definition,
        instances=list(definition.instances),
        placement_agents=_placement_agent_ids(session, definition),
    )


@router.patch("/{definition_id}")
async def patch_definition(
    definition_id: str,
    body: DefinitionPatch,
    request: Request,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    definition = _get_definition(session, definition_id)
    changes = body.model_dump(exclude_unset=True)

    # Phase 16: `agents` is a many-to-many relationship, not a column — pull
    # it out of the setattr loop and handle it via the link table after the
    # main commit. `agent_placement` IS a column and stays in `changes`.
    placement_agents_raw = changes.pop("agents", None)
    placement_touched = "agent_placement" in changes or placement_agents_raw is not None
    effective_placement = changes.get("agent_placement", definition.agent_placement)
    effective_type = changes.get("provider_type", definition.provider_type)
    resolved_agent_ids: list[uuid.UUID] | None = None
    if placement_touched:
        if placement_agents_raw is not None or effective_placement == "specific":
            resolved_agent_ids = _resolve_placement_agents(
                session, effective_placement, placement_agents_raw, effective_type
            )
        else:
            resolved_agent_ids = []  # switching to any_of_type: clear links

    # SF-4: explicit null on a non-nullable column is a validation
    # error (422), not a DB IntegrityError dressed up as a 409 conflict.
    # Phase 16: provider_type/backend_config are NOT NULL (no shells), so an
    # explicit null on either is refused here.
    for key in changes:
        if changes[key] is None and key in _NON_NULLABLE_FIELDS:
            raise HTTPException(status_code=422, detail=f"field '{key}' cannot be null")

    # M3: validate the EFFECTIVE config (post-change) against the EFFECTIVE
    # type's committed schema — a retype without a config change must still
    # satisfy the new schema. The type is resolved once up front so an
    # unknown provider_type keeps its original 422-before-409 ordering.
    effective_type_name = changes.get("provider_type", definition.provider_type)
    ptype = _get_provider_type(session, effective_type_name)
    if "backend_config" in changes:
        _validate_backend_config(ptype, changes["backend_config"])
    elif "provider_type" in changes:
        # Retype onto an existing config: validate the stored config against
        # the new type's schema.
        _validate_backend_config(ptype, definition.backend_config)
    if "provider_type" in changes and effective_type_name != definition.provider_type:
        attached = session.exec(
            select(ProviderInstance).where(
                ProviderInstance.provider_definition_id == definition.id
            )
        ).all()
        if attached:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"definition '{definition.alias}' has "
                    f"{len(attached)} instance(s) attached; changing "
                    "provider_type would break their binding — delete "
                    "the instances (or create a new definition) "
                    "instead"
                ),
            )

    # Phase 18: validate the EFFECTIVE modality against the EFFECTIVE type's
    # serves_modalities — but ONLY when the modality or the type actually
    # changes (mirrors the backend_config gating above). An unrelated PATCH
    # (e.g. enabled=false) must stay allowed even if a later schema-consensus
    # change dropped this definition's modality from its type, so the operator
    # can always disable it.
    if "modality" in changes or "provider_type" in changes:
        effective_modality = changes.get("modality", definition.modality)
        _validate_modality(ptype, effective_modality)
    # Phase 18: modality is immutable while backends are attached (same gate as
    # provider_type) — an engine booted for one endpoint kind cannot silently
    # become another.
    if "modality" in changes and changes["modality"] != definition.modality:
        attached = session.exec(
            select(ProviderInstance).where(
                ProviderInstance.provider_definition_id == definition.id
            )
        ).all()
        if attached:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"definition '{definition.alias}' has "
                    f"{len(attached)} instance(s) attached; changing "
                    "modality would break their binding — delete "
                    "the instances (or create a new definition) "
                    "instead"
                ),
            )

    old_fp: str = compute_config_fingerprint(definition.backend_config)
    old_capacity = definition.capacity
    # NOTE (N-6): a rapid double-PATCH can transiently read a stale row
    # here and push the older config; the reconnect/sweep self-heal
    # converges it, so no locking is added.
    for key, value in changes.items():
        setattr(definition, key, value)
    definition.updated_at = datetime.now(UTC)
    session.add(definition)
    # Phase 16: write placement links in the SAME transaction as the column
    # change (atomic). resolved_agent_ids is None when placement is untouched.
    if resolved_agent_ids is not None:
        _write_placement(session, definition, resolved_agent_ids)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="alias already exists",
        ) from None
    session.refresh(definition)
    # Phase 16 slice 5: a placement-affecting change (agent_placement, agents
    # links, enabled, alias) reconciles every agent of this type — creating
    # backend rows on newly-placed connected agents, retiring de-placed ghosts
    # busy-safe, and pushing agent.assignments.update to the live sockets. This
    # replaces the slice-3 interim prune (which only ran on the next
    # registration). A pure config/capacity change is NOT a placement trigger —
    # it rides provider.config.update below; a newly-placed backend already
    # carries the current config in its assignment entry.
    placement_changed = (
        resolved_agent_ids is not None
        or "enabled" in changes
        or "alias" in changes
        or "provider_type" in changes  # L1: a retype (only allowed with zero
        # attached rows) changes which agents should host it — push the new type.
    )
    if placement_changed:
        provider_type = definition.provider_type
        # End the route transaction before awaiting the (possibly slow) socket
        # fan-out — same N-8 discipline as the config push below (never hold an
        # idle-in-transaction connection across a network await).
        session.commit()
        await assignments.push_agent_assignments_for_type(
            provider_type,
            scheduler=getattr(request.app.state, "scheduler", None),
        )
    _ensure_litellm_registered(definition)

    # The provider-visible state: the config (fingerprint) or a capacity
    # change. A type-only change with the SAME config is not a push trigger
    # (fingerprint identical).
    new_fp: str = compute_config_fingerprint(definition.backend_config)
    # SF-2: push when the provider-visible fields changed — the
    # backend_config fingerprint (restart-worthy) or capacity (adopted
    # at the provider without restart). idle_timeout_seconds is not a
    # trigger (admin-side reaper only; see module docstring).
    provider_fields_changed = new_fp != old_fp or definition.capacity != old_capacity
    results: list[dict[str, Any]] = []
    if provider_fields_changed:
        reasons = []
        if new_fp != old_fp:
            reasons.append("backend_config")
        if definition.capacity != old_capacity:
            reasons.append("capacity")
        # N-8: the push can legitimately take minutes (drain + artifact
        # download + boot; command timeout 300s, retries on top). The
        # route session would otherwise sit idle-in-transaction across
        # that await and Postgres's
        # idle_in_transaction_session_timeout (15s, app/core/db.py) kills
        # the connection — the final re-read here would then 500 even
        # though the admin row and the push both succeeded. COMMIT
        # (already done above) and end the transaction, then re-bind the
        # row after the await: a committed session opens its NEXT
        # transaction lazily on the following statement, so nothing sits
        # open during the minutes-long push. The push resolves the
        # definition in its own session (push_config_update_by_id).
        definition_id = definition.id
        alias = definition.alias
        session.expire_all()
        session.commit()

        pushed = await config_update.push_config_update_by_id(definition_id)
        results = [r.to_dict() for r in pushed]
        logger.info(
            "definition %s %s changed (%s -> %s); pushed to %d connected backend(s)",
            alias,
            "+".join(reasons),
            str(old_fp)[:8],
            str(new_fp)[:8],
            len(results),
        )
        # Re-bind the row post-push (model_metadata may have been
        # discovered during the apply; _record_success wrote it in
        # another session).
        definition = session.get(ProviderDefinition, definition_id)
        if definition is None:  # pragma: no cover - deleted mid-push
            return {
                "id": str(definition_id),
                "alias": alias,
                "deleted_during_push": True,
                "config_update_results": results,
            }

    instances = session.exec(
        select(ProviderInstance).where(
            ProviderInstance.provider_definition_id == definition.id
        )
    ).all()
    return definition_dict(
        definition,
        instances=list(instances),
        config_update_results=results,
        placement_agents=_placement_agent_ids(session, definition),
    )


@router.delete("/{definition_id}")
def delete_definition(
    definition_id: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    # Phase 16 slice 5: DELETE needs no live agent.assignments.update push —
    # the guard below refuses deletion while any hosting agent is connected, so
    # at this point every remaining backend row belongs to a DISCONNECTED agent.
    # Those rows cascade-delete with the definition; the (already offline) agent
    # re-resolves an empty placement on its next registration. The live teardown
    # path for a running backend is `enabled=false` (a PATCH), which pushes the
    # removal to the connected agents.
    definition = _get_definition(session, definition_id)
    connected = session.exec(
        select(ProviderInstance)
        .join(ProviderAgent, col(ProviderInstance.agent_id) == col(ProviderAgent.id))
        .where(
            ProviderInstance.provider_definition_id == definition.id,
            ProviderAgent.websocket_connected == True,  # noqa: E712
        )
    ).all()
    if connected:
        raise HTTPException(
            status_code=409,
            detail=(
                f"definition '{definition.alias}' has "
                f"{len(connected)} connected instance(s); disconnect them or "
                "set enabled=false instead of deleting"
            ),
        )
    # Delete the offline instance rows first: the ORM relationship would
    # otherwise try to null their non-nullable FK during flush.
    remaining = session.exec(
        select(ProviderInstance).where(
            ProviderInstance.provider_definition_id == definition.id
        )
    ).all()
    for inst in remaining:
        session.delete(inst)
    session.delete(definition)
    session.commit()
    logger.info("deleted definition %s (%s)", definition.alias, definition_id)
    return {"ok": True, "id": definition_id}

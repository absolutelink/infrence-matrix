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
- ``DELETE`` is allowed only when **no instance is currently
  websocket_connected**; otherwise 409 with a suggestion to disable.
  When allowed, the definition's *offline* instance rows are deleted
  with it (cascade to non-connected rows only).

Shell definitions (Phase 14):
- Create may omit ``provider_type`` AND ``backend_config`` (a "shell").
  A shell holds the alias + token + scheduler hints only; its type is
  ADOPTED from the container's first registration and its config is
  authored through a later PATCH (validated against the adopted type's
  committed schema, then pushed via the Phase 9 flow). A shell is
  never schedulable and never pushed — the scheduler, /v1/models and
  the alias registry all exclude it (``backend_config_is_authored``).
- Rules: a shell (type null) MUST NOT carry ``backend_config`` (422 —
  there is no schema to validate against yet, so accepting it would
  store unvalidatable config). A typed definition behaves exactly as
  before (type resolved + config validated on every affected write).
- Once a type/config has been set, explicit nulls are refused (422):
  a definition cannot go back to being a shell. Setting the type on a
  shell is allowed (operator repair path) subject to the usual
  attached-instances refusal.
"""

import logging
import secrets
import uuid
from datetime import UTC, datetime
from typing import Any

import jsonschema
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.api.admin.providers import compute_config_fingerprint
from app.api.admin.serializers import iso_utc
from app.core.db import get_session
from app.models import (
    DefinitionAgent,
    ProviderAgent,
    ProviderDefinition,
    ProviderInstance,
    ProviderType,
    backend_config_is_authored,
)
from app.services import alias_registry, config_update

logger = logging.getLogger("admin.definitions")

router = APIRouter(prefix="/admin/api/definitions", tags=["admin"])

_VALID_PLACEMENTS = ("any_of_type", "specific")


class DefinitionCreate(BaseModel):
    alias: str = Field(min_length=1, max_length=255)
    # Phase 14: None = shell definition (type adopted at first registration).
    provider_type: str | None = Field(default=None, min_length=1, max_length=64)
    backend_config: dict[str, Any] | None = None
    vram_required_bytes: int = Field(default=0, ge=0)
    idle_timeout_seconds: int = Field(default=300, ge=0)
    capacity: int = Field(default=1, ge=1)
    # Auto-generated when omitted (trusted-LAN: the operator copies the
    # issued token into the provider container env).
    registration_token: str | None = Field(default=None, max_length=255)
    enabled: bool = True
    # Phase 16: placement. 'any_of_type' hosts on every agent of the type;
    # 'specific' hosts only on the listed agent ids (ProviderAgent.id).
    agent_placement: str = Field(default="any_of_type", max_length=32)
    agents: list[str] | None = None


class DefinitionPatch(BaseModel):
    alias: str | None = Field(default=None, min_length=1, max_length=255)
    provider_type: str | None = Field(default=None, min_length=1, max_length=64)
    backend_config: dict[str, Any] | None = None
    vram_required_bytes: int | None = Field(default=None, ge=0)
    idle_timeout_seconds: int | None = Field(default=None, ge=0)
    capacity: int | None = Field(default=None, ge=1)
    registration_token: str | None = Field(default=None, min_length=1, max_length=255)
    enabled: bool | None = None
    # Phase 16: placement (see DefinitionCreate).
    agent_placement: str | None = Field(default=None, max_length=32)
    agents: list[str] | None = None


# Columns that are NOT NULL in the model: an explicit null in a PATCH
# body is a client error, never a silent skip or a DB IntegrityError.
# Phase 14: provider_type/backend_config leave this list (nullable on a
# shell) but are instead guarded as "cannot return to null once set" —
# see _check_no_unshell in patch_definition.
_NON_NULLABLE_FIELDS = frozenset(
    {
        "alias",
        "vram_required_bytes",
        "idle_timeout_seconds",
        "capacity",
        "registration_token",
        "enabled",
        "agent_placement",
    }
)


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


def _resolve_placement_agents(
    session: Session, placement: str, agents: list[str] | None
) -> list[uuid.UUID]:
    """Validate ``agent_placement`` and resolve ``specific`` agent ids.

    Returns the list of ``ProviderAgent`` ids to link (empty for
    ``any_of_type``). Raises 422 on an unknown placement, an empty/absent
    list for ``specific``, or an unknown/malformed agent id.
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
        if session.get(ProviderAgent, agent_uuid) is None:
            raise HTTPException(status_code=422, detail=f"unknown agent '{raw}'")
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
    # Phase 14: a shell definition reports null type/config/fingerprint —
    # an empty config must never masquerade as authored (the fingerprint
    # of {} would look identical to a real permissive-schema config).
    authored = backend_config_is_authored(definition)
    d: dict[str, Any] = {
        "id": str(definition.id),
        "alias": definition.alias,
        "provider_type": definition.provider_type,
        "backend_config": definition.backend_config if authored else None,
        "config_fingerprint": (
            compute_config_fingerprint(definition.backend_config or {})
            if authored
            else None
        ),
        "vram_required_bytes": definition.vram_required_bytes,
        "idle_timeout_seconds": definition.idle_timeout_seconds,
        "capacity": definition.capacity,
        "registration_token": definition.registration_token,
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
                "machine_uid": i.machine.uid if i.machine else None,
                "instance_status": i.instance_status,
                "backend_status": i.backend_status,
                "websocket_connected": i.websocket_connected,
                "config_fingerprint": i.config_fingerprint,
                "port": i.port,
                "version": i.version,
            }
            for i in instances
        ]
        d["connected_instance_count"] = sum(
            1 for i in instances if i.websocket_connected
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
    body: DefinitionCreate, session: Session = Depends(get_session)
) -> dict[str, Any]:
    # Phase 14: a Shell definition (provider_type null) must not carry a
    # backend_config — there is no committed schema to validate it against.
    if body.provider_type is None:
        if body.backend_config is not None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "backend_config cannot be set on a definition without "
                    "provider_type: create the shell and let the provider "
                    "registration adopt the type first"
                ),
            )
        ptype = None
    else:
        ptype = _get_provider_type(session, body.provider_type)
        _validate_backend_config(
            ptype, body.backend_config if body.backend_config is not None else {}
        )
    # Phase 16: validate placement + resolve specific agent ids (422 on an
    # unknown agent) BEFORE inserting, so a bad placement never persists.
    agent_ids = _resolve_placement_agents(session, body.agent_placement, body.agents)
    definition = ProviderDefinition(
        **body.model_dump(exclude={"registration_token", "agents"}),
        registration_token=body.registration_token or secrets.token_urlsafe(24),
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
            detail=(f"alias or registration_token already exists ('{body.alias}')"),
        ) from None
    session.refresh(definition)
    # Warm the litellm alias registration before first use (checklist 4) —
    # only for definitions that can actually serve (a shell never
    # schedules; it is registered when the config lands instead).
    if backend_config_is_authored(definition):
        alias_registry.ensure_registered(definition.alias)
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
    resolved_agent_ids: list[uuid.UUID] | None = None
    if placement_touched:
        if placement_agents_raw is not None or effective_placement == "specific":
            resolved_agent_ids = _resolve_placement_agents(
                session, effective_placement, placement_agents_raw
            )
        else:
            resolved_agent_ids = []  # switching to any_of_type: clear links

    # SF-4: explicit null on a non-nullable column is a validation
    # error (422), not a DB IntegrityError dressed up as a 409 conflict.
    for key in changes:
        if changes[key] is None and key in _NON_NULLABLE_FIELDS:
            raise HTTPException(status_code=422, detail=f"field '{key}' cannot be null")

    # Phase 14: a definition can never go BACK to being a shell. Explicit
    # null on provider_type/backend_config *present in the body* is
    # refused 422 (absent keys mean "unchanged").
    for key in ("provider_type", "backend_config"):
        if key in changes and changes[key] is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"field '{key}' cannot be unset: a definition never "
                    "returns to the shell (provider_type/backend_config "
                    "null) state"
                ),
            )

    # M3: validate the EFFECTIVE config (post-change) against the
    # EFFECTIVE type's committed schema — a retype without a config
    # change must still satisfy the new schema. The type is resolved once
    # up front so an unknown provider_type keeps its original 422-before-
    # 409 ordering.
    effective_type_name = changes.get("provider_type", definition.provider_type)
    # Phase 14 shell rule: backend_config is settable only against a type.
    # "backend_config" arriving on a still-untyped definition (type not
    # touched by this PATCH and definition untyped) is rejected up front;
    # _get_provider_type above would NPE on None otherwise.
    if effective_type_name is None:
        if "backend_config" in changes:
            raise HTTPException(
                status_code=422,
                detail=(
                    "backend_config cannot be set while the definition has "
                    "no provider_type — set provider_type first"
                ),
            )
        ptype = None
    else:
        ptype = _get_provider_type(session, effective_type_name)
        if "backend_config" in changes:
            _validate_backend_config(ptype, changes["backend_config"])
        elif "provider_type" in changes and backend_config_is_authored(definition):
            # Retype onto an existing config: validate the stored config
            # against the new type's schema.
            _validate_backend_config(ptype, definition.backend_config or {})
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
    # Shell → typed transition with no config change is admin-side only
    # (no push): the type follows the already-registered container.

    old_fp: str | None = (
        compute_config_fingerprint(definition.backend_config or {})
        if backend_config_is_authored(definition)
        else None
    )
    old_capacity = definition.capacity
    was_shell = not backend_config_is_authored(definition)
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
            detail="alias or registration_token already exists",
        ) from None
    session.refresh(definition)
    # Phase 14: a shell is never litellm-registered — the alias warms
    # exactly when the config is authored (possibly in this PATCH).
    if backend_config_is_authored(definition):
        alias_registry.ensure_registered(definition.alias)

    # The provider-visible state: an authored config (fingerprint) or a
    # capacity change. A type-only change on a typed definition with the
    # SAME config is not a push trigger (fingerprint identical); a type
    # adoption on a shell never pushes (nothing to apply yet).
    new_fp: str | None = (
        compute_config_fingerprint(definition.backend_config or {})
        if backend_config_is_authored(definition)
        else None
    )
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
            "definition %s %s changed (%s -> %s; shell→configured=%s); "
            "pushed to %d connected instance(s)",
            alias,
            "+".join(reasons),
            str(old_fp)[:8],
            str(new_fp)[:8],
            was_shell,
            len(results),
        )
        # Re-bind the row post-push (model_metadata may have been
        # discovered during the apply; _record_success wrote it in
        # another session).
        definition = session.get(ProviderDefinition, definition_id)
        if definition is None:  # pragma: no cover - deleted mid-push
            instances = []
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
    definition = _get_definition(session, definition_id)
    connected = session.exec(
        select(ProviderInstance).where(
            ProviderInstance.provider_definition_id == definition.id,
            ProviderInstance.websocket_connected == True,  # noqa: E712
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

"""Admin provider-definition CRUD (Phase 9).

``/admin/api/definitions`` — unauthenticated trusted-LAN management of
``ProviderDefinition`` rows (the client-facing "models"). Registration
(POST /admin/api/providers/register) stays where it is; this is the
operator-side create/update/delete the UI (Phase 10) consumes.

Validation boundary (provider/README.md): the admin validates only the
fields **it owns** — non-empty ``alias``/``provider_type``,
``provider_type`` in ``PROVIDER_TYPES``, ``capacity >= 1``,
``vram_required_bytes >= 0``, ``idle_timeout_seconds >= 0``, and that
``backend_config`` is a JSON-object that round-trips through the
canonical fingerprint serializer. Deep per-provider-type
``backend_config`` validation stays provider-side (the driver raises at
start and the failure surfaces through the config.update NAK).

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
"""

import logging
import secrets
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.api.admin.providers import compute_config_fingerprint
from app.api.admin.serializers import iso_utc
from app.core.db import get_session
from app.models import PROVIDER_TYPES, ProviderDefinition, ProviderInstance
from app.services import alias_registry, config_update

logger = logging.getLogger("admin.definitions")

router = APIRouter(prefix="/admin/api/definitions", tags=["admin"])


class DefinitionCreate(BaseModel):
    alias: str = Field(min_length=1, max_length=255)
    provider_type: str = Field(min_length=1, max_length=64)
    backend_config: dict[str, Any] = Field(default_factory=dict)
    vram_required_bytes: int = Field(default=0, ge=0)
    idle_timeout_seconds: int = Field(default=300, ge=0)
    capacity: int = Field(default=1, ge=1)
    # Auto-generated when omitted (trusted-LAN: the operator copies the
    # issued token into the provider container env).
    registration_token: str | None = Field(default=None, max_length=255)
    enabled: bool = True


class DefinitionPatch(BaseModel):
    alias: str | None = Field(default=None, min_length=1, max_length=255)
    provider_type: str | None = Field(default=None, min_length=1, max_length=64)
    backend_config: dict[str, Any] | None = None
    vram_required_bytes: int | None = Field(default=None, ge=0)
    idle_timeout_seconds: int | None = Field(default=None, ge=0)
    capacity: int | None = Field(default=None, ge=1)
    registration_token: str | None = Field(default=None, min_length=1, max_length=255)
    enabled: bool | None = None


# Columns that are NOT NULL in the model: an explicit null in a PATCH
# body is a client error, never a silent skip or a DB IntegrityError.
_NON_NULLABLE_FIELDS = frozenset(
    {
        "alias",
        "provider_type",
        "backend_config",
        "vram_required_bytes",
        "idle_timeout_seconds",
        "capacity",
        "registration_token",
        "enabled",
    }
)


def _validate_provider_type(provider_type: str) -> None:
    if provider_type not in PROVIDER_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"unknown provider_type '{provider_type}'; "
                f"must be one of {', '.join(PROVIDER_TYPES)}"
            ),
        )


def _validate_backend_config(backend_config: dict[str, Any]) -> None:
    # Must survive the canonical fingerprint serializer (rejects
    # non-JSON-serializable values sneaking in via odd Pydantic input).
    try:
        compute_config_fingerprint(backend_config)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"backend_config is not JSON-serializable: {exc}",
        ) from exc


def definition_dict(
    definition: ProviderDefinition,
    *,
    instances: list[ProviderInstance] | None = None,
    config_update_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": str(definition.id),
        "alias": definition.alias,
        "provider_type": definition.provider_type,
        "backend_config": definition.backend_config,
        "config_fingerprint": compute_config_fingerprint(
            definition.backend_config or {}
        ),
        "vram_required_bytes": definition.vram_required_bytes,
        "idle_timeout_seconds": definition.idle_timeout_seconds,
        "capacity": definition.capacity,
        "registration_token": definition.registration_token,
        "model_metadata": definition.model_metadata,
        "enabled": definition.enabled,
        "status": definition.status,
        "created_at": iso_utc(definition.created_at),
        "updated_at": iso_utc(definition.updated_at),
    }
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
    _validate_provider_type(body.provider_type)
    _validate_backend_config(body.backend_config)
    definition = ProviderDefinition(
        **body.model_dump(exclude={"registration_token"}),
        registration_token=body.registration_token or secrets.token_urlsafe(24),
    )
    session.add(definition)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail=(f"alias or registration_token already exists ('{body.alias}')"),
        ) from None
    session.refresh(definition)
    # Warm the litellm alias registration before first use (checklist 4).
    alias_registry.ensure_registered(definition.alias)
    logger.info("created definition %s (%s)", definition.alias, definition.id)
    return definition_dict(definition, instances=[])


@router.get("")
def list_definitions(session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    definitions = session.exec(
        select(ProviderDefinition).order_by(ProviderDefinition.alias)
    ).all()
    # `instances` is selectin-loaded on the relationship — no per-row SELECT.
    return [definition_dict(d, instances=list(d.instances)) for d in definitions]


@router.get("/{definition_id}")
def get_definition(
    definition_id: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    definition = _get_definition(session, definition_id)
    return definition_dict(definition, instances=list(definition.instances))


@router.patch("/{definition_id}")
async def patch_definition(
    definition_id: str,
    body: DefinitionPatch,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    definition = _get_definition(session, definition_id)
    changes = body.model_dump(exclude_unset=True)

    # SF-4: explicit null on a non-nullable column is a validation
    # error (422), not a DB IntegrityError dressed up as a 409 conflict.
    for key in changes:
        if changes[key] is None and key in _NON_NULLABLE_FIELDS:
            raise HTTPException(status_code=422, detail=f"field '{key}' cannot be null")

    if "provider_type" in changes:
        _validate_provider_type(changes["provider_type"])
        if changes["provider_type"] != definition.provider_type:
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
    if "backend_config" in changes:
        _validate_backend_config(changes["backend_config"])

    old_fp = compute_config_fingerprint(definition.backend_config or {})
    old_capacity = definition.capacity
    # NOTE (N-6): a rapid double-PATCH can transiently read a stale row
    # here and push the older config; the reconnect/sweep self-heal
    # converges it, so no locking is added.
    for key, value in changes.items():
        setattr(definition, key, value)
    definition.updated_at = datetime.now(UTC)
    session.add(definition)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail="alias or registration_token already exists",
        ) from None
    session.refresh(definition)
    alias_registry.ensure_registered(definition.alias)

    new_fp = compute_config_fingerprint(definition.backend_config or {})
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
        pushed = await config_update.push_config_update(definition)
        results = [r.to_dict() for r in pushed]
        logger.info(
            "definition %s %s changed (%s -> %s); pushed to %d connected instance(s)",
            definition.alias,
            "+".join(reasons),
            old_fp[:8],
            new_fp[:8],
            len(results),
        )
        # The push commits instance fingerprints / model_metadata in a
        # separate session; drop the cached relationship + row so the
        # response reflects the post-push state.
        session.expire_all()
        session.refresh(definition)

    instances = session.exec(
        select(ProviderInstance).where(
            ProviderInstance.provider_definition_id == definition.id
        )
    ).all()
    return definition_dict(
        definition, instances=list(instances), config_update_results=results
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

"""Canonical served-model resolver (Phase 25 / docs/multi-model-definitions.md).

A ``ProviderDefinition`` may expose a **set** of served models (multi-model,
``ProviderType.multi_model``) or just its own ``alias``/``modality``/
``backend_config`` (single-model). Every consumer — the scheduler,
``/v1/models``, assignment payloads, v1 endpoint gates, litellm registration —
routes name resolution through this module so the two shapes behave
identically and single-model definitions stay byte-for-byte unchanged.

The agent-side mirror of this shape is ``provider_lib.models.ModelSpec`` /
``models_from_entry`` (S1); the wire key is ``served_models``. This module is
the admin-side source of truth (Postgres), not a wire parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlmodel import Session, col, select

from app.models import ProviderDefinition, ProviderModel


@dataclass
class ModelSpec:
    """One client-facing model served by a definition's backend process."""

    name: str
    modality: str
    backend_config: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


def _spec_from_row(row: ProviderModel) -> ModelSpec:
    return ModelSpec(
        name=row.name,
        modality=row.modality,
        backend_config=dict(row.backend_config or {}),
        enabled=bool(row.enabled),
    )


def _synth_spec(definition: ProviderDefinition) -> ModelSpec:
    """The single synthesized spec for a definition with no ProviderModel rows."""
    return ModelSpec(
        name=definition.alias,
        modality=definition.modality,
        backend_config=dict(definition.backend_config or {}),
        enabled=True,
    )


def resolve_models(session: Session, definition: ProviderDefinition) -> list[ModelSpec]:
    """The definition's served-model specs.

    Its ``ProviderModel`` rows when any exist — **including disabled ones** so
    callers can gate on ``enabled`` — else the synthesized single entry
    ``[{alias, modality, backend_config, enabled=True}]``. Order is
    deterministic (by name) so assignment payloads and fingerprints are stable.
    """
    rows = session.exec(
        select(ProviderModel)
        .where(ProviderModel.definition_id == definition.id)
        .order_by(col(ProviderModel.name).asc())
    ).all()
    if rows:
        return [_spec_from_row(row) for row in rows]
    return [_synth_spec(definition)]


def resolve_served(
    session: Session, name: str
) -> tuple[ProviderDefinition, ModelSpec] | None:
    """Resolve a client-facing ``name`` to ``(owning definition, matching spec)``.

    The name matches a ``ProviderModel`` row first (multi-model served name);
    otherwise it falls back to a single-model definition's ``alias``. Returns
    ``None`` when the name is unknown **or** refers to a disabled
    ``ProviderModel`` (a disabled served name 404s like a disabled definition —
    spec §3). The owning definition's own ``enabled`` flag is NOT consulted
    here; callers gate on it so they can distinguish "definition disabled" from
    "unknown name" in their error messages.
    """
    row = session.exec(select(ProviderModel).where(ProviderModel.name == name)).first()
    if row is not None:
        definition = session.get(ProviderDefinition, row.definition_id)
        if definition is None:  # pragma: no cover - FK guarantees presence
            return None
        spec = _spec_from_row(row)
        if not spec.enabled:
            return None
        return definition, spec
    definition = session.exec(
        select(ProviderDefinition).where(ProviderDefinition.alias == name)
    ).first()
    if definition is None:
        return None
    return definition, _synth_spec(definition)


def served_models_payload(
    session: Session, definition: ProviderDefinition
) -> list[dict[str, Any]]:
    """The canonical ``served_models`` wire list for an assignment/registration
    entry (always the full list — single-model definitions send one entry).

    Each item is ``{name, modality, backend_config, enabled}`` — the shape
    ``provider_lib.models.models_from_entry`` prefers on the agent side.
    """
    return [
        {
            "name": spec.name,
            "modality": spec.modality,
            "backend_config": spec.backend_config,
            "enabled": spec.enabled,
        }
        for spec in resolve_models(session, definition)
    ]


__all__ = [
    "ModelSpec",
    "resolve_models",
    "resolve_served",
    "served_models_payload",
]

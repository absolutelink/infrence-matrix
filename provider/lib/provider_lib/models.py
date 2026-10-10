"""Multi-model served-model specs (Phase 25 S1).

A single backend process may serve several client-facing model names
(``docs/multi-model-definitions.md``). ``ModelSpec`` is the agent-side view of
one served name; :func:`models_from_entry` parses the wire shape — a
registration ``definition`` dict or an ``agent.assignments.update`` entry —
into a list of specs.

Parsing policy (locked for S1):

* Prefer ``entry["served_models"]`` (a list of ``{name, modality,
  backend_config, enabled}``). Each item must be an object with a non-empty
  string ``name``; **invalid items are skipped with a warning**.
* When ``served_models`` is absent, not a list, or yields no valid items, fall
  back to the legacy single-model trio (``alias`` / ``modality`` /
  ``backend_config``) as one enabled entry — old admins keep working unchanged.
* If neither a valid ``served_models`` list nor a usable ``alias`` is present,
  the result is an empty list (a nameless placeholder handle is unroutable).

This module is dependency-free (imports nothing from ``provider_lib``) so both
``backend`` (the driver hook type hint) and ``registry`` can import it without a
cycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("provider.models")

# Endpoint kind assumed when a wire entry omits ``modality`` (matches the
# ``BackendDriver.modality`` default).
DEFAULT_MODALITY = "llm"


@dataclass
class ModelSpec:
    """One client-facing model served by a backend process."""

    name: str
    modality: str = DEFAULT_MODALITY
    backend_config: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


def _spec_from_item(item: Any) -> ModelSpec | None:
    """Build a :class:`ModelSpec` from one ``served_models`` entry.

    Returns ``None`` (and logs a warning) for a non-object entry or one whose
    ``name`` is missing / not a non-empty string. ``modality`` /
    ``backend_config`` / ``enabled`` degrade to safe defaults.
    """
    if not isinstance(item, dict):
        logger.warning("served_models: skipping non-object entry %r", item)
        return None
    name = item.get("name")
    if not isinstance(name, str) or not name:
        logger.warning(
            "served_models: skipping entry with missing/invalid name: %r", item
        )
        return None
    modality = item.get("modality")
    if not isinstance(modality, str) or not modality:
        modality = DEFAULT_MODALITY
    backend_config = item.get("backend_config")
    if not isinstance(backend_config, dict):
        backend_config = {}
    return ModelSpec(
        name=name,
        modality=modality,
        backend_config=backend_config,
        enabled=bool(item.get("enabled", True)),
    )


def _legacy_specs(entry: dict[str, Any]) -> list[ModelSpec]:
    """Synthesize the single-model spec list from the legacy trio."""
    alias = entry.get("alias")
    if not isinstance(alias, str) or not alias:
        return []
    modality = entry.get("modality")
    if not isinstance(modality, str) or not modality:
        modality = DEFAULT_MODALITY
    backend_config = entry.get("backend_config")
    if not isinstance(backend_config, dict):
        backend_config = {}
    return [
        ModelSpec(
            name=alias,
            modality=modality,
            backend_config=backend_config,
            enabled=True,
        )
    ]


def models_from_entry(entry: dict[str, Any] | None) -> list[ModelSpec]:
    """Parse a registration ``definition`` or assignment ``entry`` into specs.

    Prefers ``served_models`` (skipping invalid entries with a warning); falls
    back to the legacy ``alias``/``modality``/``backend_config`` trio when
    ``served_models`` is absent or yields nothing valid.
    """
    if not isinstance(entry, dict):
        return []
    served = entry.get("served_models")
    if isinstance(served, list):
        specs = [
            spec
            for spec in (_spec_from_item(item) for item in served)
            if spec is not None
        ]
        if specs:
            return specs
        logger.warning(
            "served_models present but yielded no valid entries; "
            "falling back to legacy alias/modality/backend_config"
        )
    return _legacy_specs(entry)


__all__ = ["ModelSpec", "models_from_entry", "DEFAULT_MODALITY"]

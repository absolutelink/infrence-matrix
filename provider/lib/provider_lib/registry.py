"""Per-backend dispatch registry (Phase 16, H3).

An agent owns 1..N backends (``ProviderInstance`` rows), each with its own
``BackendLifecycle`` + ``ConfigState``. Admin commands that target a specific
backend (``backend.start`` / ``backend.stop`` / ``backend.restart`` /
``provider.initialize`` / ``provider.config.update`` / ``cache.clear`` /
``storage.prune_unused``) carry the target ``instance_id`` in their payload.

The registry maps ``instance_id -> BackendHandle`` so the command handlers
installed by :func:`provider_lib.ops.install_backend_ops` and
:func:`provider_lib.config_update.install_config_handlers` route each frame to
the correct hosted lifecycle instead of blindly driving the first one. A frame
naming an ``instance_id`` this agent does not host is NAK'd ``unknown_instance``
rather than silently mis-served.

Single-backend providers (every package today) simply register one handle; the
dispatch layer is a transparent pass-through for them.
"""

from __future__ import annotations

import logging
from typing import Any

from provider_lib.backend import BackendLifecycle
from provider_lib.models import ModelSpec

logger = logging.getLogger("provider.registry")


class BackendHandle:
    """One hosted backend: its lifecycle + applied-config state + identity.

    ``models`` is the list of client-facing served models this backend exposes
    (Phase 25 multi-model). It is always non-empty for a handle built from a real
    registration/assignment entry (a single-model definition carries exactly one
    synthesized spec); a nameless placeholder handle (built before an alias is
    known) carries an empty list and is unroutable. ``alias`` is kept as the
    first enabled served name for display / back-compat (the ``/health`` summary
    and older packages read it).

    The agent's single ``/v1`` surface routes an inbound request to this handle
    by matching the request's ``model`` field against ANY enabled served name
    (see :meth:`BackendRegistry.resolve_by_model`).
    """

    def __init__(
        self,
        instance_id: str,
        lifecycle: BackendLifecycle,
        config_state: Any = None,
        alias: str | None = None,
        models: list[ModelSpec] | None = None,
    ) -> None:
        self.instance_id = instance_id
        self.lifecycle = lifecycle
        self.config_state = config_state
        self.alias = alias
        if models is None:
            # Synthesize the single-model spec from the legacy alias so every
            # existing construction site (which passes only ``alias``) carries a
            # one-entry model list; a nameless placeholder stays empty.
            models = [ModelSpec(name=alias)] if isinstance(alias, str) and alias else []
        self.models = models

    @property
    def served_names(self) -> list[str]:
        """Client-facing names this handle routes on (enabled served models).

        There is deliberately **no alias fallback**: the constructor already
        synthesizes a one-entry ``models`` list from a legacy ``alias`` (see
        :meth:`__init__`), so a single-model handle routes on ``[alias]`` through
        ``models`` itself. Falling back to ``alias`` here would resurrect a stale
        name whenever ``models`` is empty — breaking both the all-disabled toggle
        (spec §3/§5: disabled -> 404) and a ``set_models([])`` on a handle that
        lost a name to a collision (it would steal the name back). A nameless
        placeholder (``models == []``, ``alias is None``) correctly routes on
        nothing.
        """
        return [m.name for m in self.models if m.enabled]


class BackendRegistry:
    """``instance_id -> BackendHandle`` with single-backend convenience.

    Also maintains a ``model name -> BackendHandle`` index over every handle's
    **enabled served names** so the agent's single ``/v1`` surface can resolve
    the target backend from a request's ``model`` field (port model overhaul +
    Phase 25 multi-model). The index is kept in sync by :meth:`add` /
    :meth:`remove` / :meth:`set_models`.
    """

    def __init__(self) -> None:
        self._handles: dict[str, BackendHandle] = {}
        self._by_model: dict[str, BackendHandle] = {}

    def _index(self, name: str, handle: BackendHandle) -> None:
        """Point a served name at ``handle``, warning on a cross-handle clash.

        Last-write-wins is preserved (the newest handle owns the name), but a
        collision across two DIFFERENT handles is almost always a
        misconfiguration (two definitions claiming one name), so it is logged
        loudly rather than silently shadowing the previous owner.
        """
        existing = self._by_model.get(name)
        if existing is not None and existing is not handle:
            logger.warning(
                "model name %r already served by another handle (%s); "
                "overwriting (last-write-wins)",
                name,
                existing.instance_id,
            )
        self._by_model[name] = handle

    def add(self, handle: BackendHandle) -> None:
        self._handles[handle.instance_id] = handle
        for name in handle.served_names:
            self._index(name, handle)

    def remove(self, instance_id: str) -> BackendHandle | None:
        """Drop a hosted handle by its key (slice 5 assignments reconcile).

        Returns the removed handle (or ``None`` when no handle is keyed by
        ``instance_id``). The caller is responsible for having stopped the
        lifecycle first (busy-safe removal is enforced upstream).
        """
        handle = self._handles.pop(str(instance_id), None)
        if handle is not None:
            # Only clear an indexed name if it still points at this handle (a
            # later add/set_models of the same name would have replaced it).
            for name in handle.served_names:
                if self._by_model.get(name) is handle:
                    del self._by_model[name]
        return handle

    def set_models(self, handle: BackendHandle, models: list[ModelSpec]) -> None:
        """Re-apply a handle's served-model list in place (Phase 25).

        Updates ``handle.models``, re-derives ``alias`` (first enabled name) for
        display/back-compat, re-indexes the ``model -> handle`` map, and pushes
        the list to the driver via ``driver.set_models`` so a driver can rebuild
        its internal name->engine map **without a restart**. The lifecycle is
        never touched (no stop/boot), which is exactly what makes a live add of a
        served name safe on a running backend.

        The re-index uses the explicit enabled-name set (never the alias
        fallback), so toggling the sole name off — or an empty list — removes the
        handle from the index instead of resurrecting a stale alias.
        """
        models = list(models)
        # Drop the names this handle currently owns (guarded: only remove an
        # entry that still points at THIS handle, so a name another handle won
        # via a collision is left intact when this one moves away).
        for name in handle.served_names:
            if self._by_model.get(name) is handle:
                del self._by_model[name]
        handle.models = models
        enabled = [m.name for m in models if m.enabled]
        handle.alias = enabled[0] if enabled else handle.alias
        for name in enabled:
            self._index(name, handle)
        driver = getattr(handle.lifecycle, "driver", None)
        setter = getattr(driver, "set_models", None) if driver is not None else None
        if callable(setter):
            setter(models)

    def get(self, instance_id: str) -> BackendHandle | None:
        return self._handles.get(instance_id)

    def resolve(self, instance_id: str | None) -> BackendHandle | None:
        """Resolve the target handle for a command payload.

        ``instance_id`` present -> that exact handle (or ``None`` if this agent
        does not host it, so the caller NAKs). ``instance_id`` absent -> the
        sole handle when the agent hosts exactly one backend (legacy /
        single-backend convenience); otherwise ambiguous -> ``None``.
        """
        if instance_id is not None:
            return self._handles.get(instance_id)
        if len(self._handles) == 1:
            return next(iter(self._handles.values()))
        return None

    def resolve_by_model(self, model: str | None) -> BackendHandle | None:
        """Resolve the handle that serves an inbound request's ``model`` field
        (the port model overhaul: one agent ``/v1`` surface routes by model).

        Matches ANY **enabled** served name on a handle (Phase 25 multi-model):
        a disabled or unknown name is not indexed and resolves to ``None``.
        ``None`` model, a non-string model, or an empty string -> ``None`` (the
        caller answers 404). The non-string guard keeps a malformed request
        (e.g. ``model`` as a list/int) from raising inside ``dict.get`` — it
        degrades to a clean 404 instead of a 500.
        """
        if not isinstance(model, str) or not model:
            return None
        return self._by_model.get(model)

    def resolve_target(self, instance_id: str | None) -> BackendHandle | None:
        """Resolve the handle a per-backend command targets, honoring a
        single-backend agent whose registry key predates registration.

        A single-backend package installs its handlers before ``register()``
        returns, so its handle is keyed by the placeholder ``""`` and only
        learns its real ``instance_id`` when ``apply_registration`` stamps
        ``lifecycle.instance_id``. This resolver therefore matches a named
        ``instance_id`` against BOTH the handle key and the handle's live
        ``lifecycle.instance_id``.

        Rules (H3 hardening):
        * ``instance_id`` absent -> the sole handle if exactly one is hosted,
          else ``None`` (ambiguous).
        * ``instance_id`` present -> the handle that owns it (by key or by its
          lifecycle's current id). A foreign id on a single-handle agent is
          NOT served — it returns ``None`` so the caller NAKs
          ``unknown_instance`` rather than mis-booting the one backend.
        """
        if instance_id is None:
            handles = self.handles()
            return handles[0] if len(handles) == 1 else None
        iid = str(instance_id)
        handle = self._handles.get(iid)
        if handle is not None:
            return handle
        for h in self._handles.values():
            live = h.lifecycle.instance_id
            if live is not None and str(live) == iid:
                return h
        return None

    def handles(self) -> list[BackendHandle]:
        return list(self._handles.values())

    def __len__(self) -> int:
        return len(self._handles)

    def __contains__(self, instance_id: object) -> bool:
        return instance_id in self._handles


def registry_from_lifecycle(
    lifecycle: BackendLifecycle, config_state: Any = None
) -> BackendRegistry:
    """Build a one-handle registry from a single lifecycle (backward-compat
    path for providers that host exactly one backend)."""
    reg = BackendRegistry()
    reg.add(
        BackendHandle(
            str(lifecycle.instance_id) if lifecycle.instance_id is not None else "",
            lifecycle,
            config_state,
        )
    )
    return reg


__all__ = ["BackendHandle", "BackendRegistry", "registry_from_lifecycle", "ModelSpec"]

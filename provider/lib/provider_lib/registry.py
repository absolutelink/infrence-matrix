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

from typing import Any

from provider_lib.backend import BackendLifecycle


class BackendHandle:
    """One hosted backend: its lifecycle + applied-config state."""

    def __init__(
        self,
        instance_id: str,
        lifecycle: BackendLifecycle,
        config_state: Any = None,
    ) -> None:
        self.instance_id = instance_id
        self.lifecycle = lifecycle
        self.config_state = config_state


class BackendRegistry:
    """``instance_id -> BackendHandle`` with single-backend convenience."""

    def __init__(self) -> None:
        self._handles: dict[str, BackendHandle] = {}

    def add(self, handle: BackendHandle) -> None:
        self._handles[handle.instance_id] = handle

    def remove(self, instance_id: str) -> BackendHandle | None:
        """Drop a hosted handle by its key (slice 5 assignments reconcile).

        Returns the removed handle (or ``None`` when no handle is keyed by
        ``instance_id``). The caller is responsible for having stopped the
        lifecycle first (busy-safe removal is enforced upstream).
        """
        return self._handles.pop(str(instance_id), None)

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


__all__ = ["BackendHandle", "BackendRegistry", "registry_from_lifecycle"]

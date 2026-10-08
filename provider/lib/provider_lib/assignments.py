"""Provider-side ``agent.assignments.update`` reconcile (Phase 16 slice 5).

The admin pushes an agent's **full** current assignment set whenever a
definition's placement changes (added to / removed from this agent) so the
agent can spin up or tear down the corresponding backend slots over its single
socket — without a full re-registration. This handler reconciles the agent's
:class:`~provider_lib.registry.BackendRegistry` to match the pushed set:

* **add** — for each assignment whose ``instance_id`` this agent does not yet
  host, build a fresh ``BackendHandle`` via the package-supplied ``make_handle``
  factory (lifecycle + applied-config state seeded from the entry) and register
  it. Since slice 6 every provider package (mock + all four real engines) supplies
  a ``make_handle`` and hosts N backends per process, so adds are normally served.
  The factory is optional only as a defensive fallback: an agent that passes none
  **refuses** an add of a backend it does not already host (reported in
  ``refused``) rather than crashing; the admin leaves the row and the backend
  simply never boots.
* **remove** — for each hosted handle no longer in the set, stop the engine if
  idle and drop the handle. A **busy** backend (live slots held) is refused and
  kept until it frees — the exact mirror of the admin's busy-safe prune, so a
  de-placed-but-running backend is never cut mid-stream.

Ack payload (ARCHITECTURE.md §5): ``{ok, added, removed, refused}`` where
``added``/``removed`` are ``instance_id`` lists and ``refused`` is a list of
``{instance_id, reason}``. The reconcile is pure bookkeeping (no engine boot),
so it always acks well inside the command window.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from provider_lib.admin_client import AdminClient
from provider_lib.backend import BackendBusy
from provider_lib.registry import BackendHandle, BackendRegistry
from provider_lib.wire import Frame, FrameKind

logger = logging.getLogger("provider.assignments")

# Builds a hosted handle from one assignment entry. Supplied by every provider
# package since slice 6 (mock + all four real engines host N backends per
# process); when omitted the agent defensively refuses adds it cannot serve.
HandleFactory = Callable[[dict[str, Any]], BackendHandle]


def install_assignment_handler(
    client: AdminClient,
    registry: BackendRegistry,
    *,
    make_handle: HandleFactory | None = None,
    on_max_running: Callable[[int], None] | None = None,
) -> None:
    """Install the ``agent.assignments.update`` command handler on ``client``.

    ``make_handle`` (optional) turns an assignment entry into a
    :class:`BackendHandle`. Every provider package supplies one since slice 6;
    when absent the agent defensively refuses any add of a backend it does not
    already host rather than crashing.
    ``on_max_running`` (optional) lets a package cache the agent's
    ``max_running_backends`` view (the admin scheduler is the authority; this
    is informational).
    """

    async def on_assignments_update(frame: Frame) -> dict[str, Any]:
        payload = frame.payload or {}
        assignments = payload.get("assignments")
        if not isinstance(assignments, list):
            return {
                "ok": False,
                "error": "invalid payload: assignments (list) required",
                "detail": {"step": "validate"},
            }
        max_running = payload.get("max_running_backends")
        if isinstance(max_running, int) and on_max_running is not None:
            on_max_running(max_running)

        desired: dict[str, dict[str, Any]] = {}
        for entry in assignments:
            if not isinstance(entry, dict):
                continue
            iid = entry.get("instance_id")
            if iid is not None:
                desired[str(iid)] = entry

        added: list[str] = []
        removed: list[str] = []
        refused: list[dict[str, str]] = []

        # Add backends the agent does not yet host.
        for iid, entry in desired.items():
            if registry.resolve_target(iid) is not None:
                # Already hosted. NOTE (L2): an already-hosted backend's alias/
                # metadata is NOT refreshed here — recreating the lifecycle would
                # drop its running state. A renamed definition reaches a live
                # backend via the next provider.config.update (which carries the
                # new config) or its next re-registration; assignments.update
                # only governs which backends exist, not their live contents.
                continue
            if make_handle is None:
                # Defensive: every provider package supplies a make_handle
                # factory since slice 6, so this path is not hit in practice.
                # An agent built without one cannot spawn a new lifecycle, so
                # refuse the add rather than crash.
                refused.append(
                    {
                        "instance_id": iid,
                        "reason": "no_handle_factory: agent cannot spawn a new backend",
                    }
                )
                continue
            try:
                handle = make_handle(entry)
            except Exception as exc:  # noqa: BLE001 - reported, never fatal
                logger.exception("assignments: add failed for %s", iid)
                refused.append({"instance_id": iid, "reason": f"add_failed: {exc}"})
                continue
            # Phase 18 slice 3: thread the entry's endpoint kind onto the new
            # handle's driver so every provider package gets it without each
            # factory re-reading the field. Coerce to str; absent/unknown ->
            # "llm" (the BackendDriver default). Also mirror it onto the
            # handle's applied-config state (when present) so a package that
            # relies solely on this shared stamp keeps config_state consistent.
            raw_modality = entry.get("modality")
            modality = (
                raw_modality
                if isinstance(raw_modality, str) and raw_modality
                else "llm"
            )
            handle.lifecycle.driver.modality = modality
            cfg_state = getattr(handle, "config_state", None)
            if cfg_state is not None:
                cfg_state.modality = modality
            registry.add(handle)
            added.append(iid)

        # Retire hosted backends no longer assigned, busy-safe. Resolve each
        # handle's identity the SAME way resolve_target does — the live
        # ``lifecycle.instance_id`` first, falling back to the registry key —
        # so a single-backend agent whose handle is still keyed by the ""
        # placeholder (its real instance_id is stamped only after
        # apply_registration) is NOT torn down when the admin pushes that real
        # id (B1). Removal is by the actual registry key.
        for handle in list(registry.handles()):
            iid = str(handle.lifecycle.instance_id or handle.instance_id)
            if iid in desired:
                continue
            try:
                await handle.lifecycle.stop_if_idle()
            except BackendBusy:
                refused.append({"instance_id": iid, "reason": "backend_in_use"})
                continue
            registry.remove(handle.instance_id)
            removed.append(iid)

        # Slice 6: a reconcile that changed the hosted set drives the multi-port
        # server to bind/unbind the affected backends' HTTP listeners. Fired
        # only on an actual change so the initial build (before the server
        # exists) and no-op pushes stay silent.
        if added or removed:
            await registry.notify_changed()

        logger.info(
            "assignments reconciled: added=%s removed=%s refused=%s",
            added,
            removed,
            refused,
        )
        return {"ok": True, "added": added, "removed": removed, "refused": refused}

    client.on_command(FrameKind.AGENT_ASSIGNMENTS_UPDATE, on_assignments_update)


__all__ = ["install_assignment_handler", "HandleFactory"]

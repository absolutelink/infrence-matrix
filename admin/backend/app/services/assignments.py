"""Phase 16 slice 5: event-driven placement reconciliation + ``agent.assignments.update``.

When a ``ProviderDefinition``'s placement changes (created / PATCHed /
disabled / re-linked), the admin must propagate the new assignment set to
every **live** provider agent over its single WebSocket — *without* forcing a
full re-registration. This module is the **one** placement-diff path:
:func:`reconcile_agent_placement` diffs an agent's ``ProviderInstance`` rows
against its resolved placement (create new backends, retire de-placed ones
busy-safe) and is shared by registration (``providers.py``) and the live push
(:func:`push_agent_assignments`), replacing the slice-3 "prune only on next
registration" interim.

Frame (ARCHITECTURE.md §5): admin → provider ``agent.assignments.update``
carries the agent's **full** current assignment list plus its
``max_running_backends``; the provider reconciles its ``BackendRegistry`` and
acks ``{ok, added, removed, refused}``. Busy-safe removal is enforced on BOTH
sides: the admin never deletes a ``running``/``in_use`` row (leaves it for a
later registration/PATCH to prune once it stops), and the provider never stops
a busy handle (reports it in ``refused`` and keeps it until it frees).

If the agent is disconnected the DB rows are still reconciled (so a de-placed
ghost cannot linger as a schedulable row) but no frame is sent — the next
registration re-resolves placement authoritatively. On a successful push the
scheduler is asked to proactively warm the agent (the slice-4 seam) so newly
added backends boot within the VRAM + ``max_running_backends`` budget.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlmodel import Session, select

from app.core.db import engine
from app.models import (
    ProviderAgent,
    ProviderDefinition,
    ProviderInstance,
    ProviderType,
)
from app.services.connection_manager import manager
from app.services.wire import BackendStatusValue, FrameKind

logger = logging.getLogger("admin.assignments")

# The provider's assignment reconcile is pure bookkeeping (spawn/retire
# lifecycle handles; no engine boot), so the default command ack window is
# plenty. Warm-up runs separately as a bounded background pass.
ASSIGNMENTS_TIMEOUT_SECONDS = 30.0

# Strong references to fire-and-forget warm-up tasks (asyncio only weakly
# references running tasks; a boot can await for a long time). Mirrors the WS
# accept path's ``_spawn_background`` discipline (app/api/ws.py).
_background_tasks: set[asyncio.Task[None]] = set()  # type: ignore[type-arg]

_LIVE_BACKEND_STATUSES = (BackendStatusValue.RUNNING, BackendStatusValue.IN_USE)


@dataclass
class PlacementDiff:
    """Outcome of reconciling an agent's backend rows against its placement."""

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    refused: list[dict[str, str]] = field(default_factory=list)
    assignments: list[dict[str, Any]] = field(default_factory=list)


def build_assignment_entry(
    session: Session, instance: ProviderInstance, definition: ProviderDefinition
) -> dict[str, Any]:
    """One ``agent.assignments.update`` entry (ARCHITECTURE.md §5 payload).

    Phase 25: carries the canonical ``served_models`` list (always present —
    single-model definitions send one synthesized entry) and a fingerprint that
    covers the served list so a per-model edit is detected by the provider.
    """
    from app.api.admin.providers import compute_definition_fingerprint
    from app.services.served_models import served_models_payload

    return {
        "instance_id": str(instance.id),
        "provider_definition_id": str(definition.id),
        "alias": definition.alias,
        "modality": definition.modality,
        "backend_config": definition.backend_config,
        "config_fingerprint": compute_definition_fingerprint(session, definition),
        "capacity": definition.capacity,
        "idle_timeout_seconds": definition.idle_timeout_seconds,
        "vram_required_bytes": definition.vram_required_bytes,
        "served_models": served_models_payload(session, definition),
    }


def reconcile_agent_placement(
    session: Session,
    agent: ProviderAgent,
    placed: list[ProviderDefinition],
    *,
    create_new: bool,
) -> PlacementDiff:
    """Diff ``agent``'s ``ProviderInstance`` rows against ``placed`` (the one
    placement-diff path shared by registration and the live push).

    * ``create_new`` — create a stopped row for each placed definition the
      agent does not yet host. Registration always creates (the agent is
      declaring itself); the live push creates only for a **connected** agent
      (a disconnected one materializes its rows on its next registration).
    * Always: refresh the ``config_fingerprint`` of existing placed rows, and
      delete rows for definitions no longer placed **unless** the backend is
      ``running``/``in_use`` (busy-safe — dropping a live row would leak the
      scheduler's per-booted VRAM hold and orphan the engine). Busy ghosts are
      reported in ``refused`` and left for a later prune.

    Backends carry no admin-facing port (the agent publishes a single env
    ``PROVIDER_PORT`` and routes by model), so there is nothing to allocate or
    renumber here.

    ``placed`` must be sorted by alias (as ``_placed_definitions`` returns) so
    the assignment list is deterministic. Mutates the caller's session; the
    caller commits.
    """
    from app.api.admin.providers import compute_definition_fingerprint

    placed_ids = {d.id for d in placed}
    existing = {
        inst.provider_definition_id: inst
        for inst in session.exec(
            select(ProviderInstance).where(ProviderInstance.agent_id == agent.id)
        ).all()
    }

    added: list[str] = []
    removed: list[str] = []
    refused: list[dict[str, str]] = []
    row_for_def: dict[uuid.UUID, ProviderInstance] = {}

    for definition in placed:
        fp = compute_definition_fingerprint(session, definition)
        inst = existing.get(definition.id)
        if inst is None:
            if not create_new:
                continue
            inst = ProviderInstance(
                agent_id=agent.id,
                provider_definition_id=definition.id,
                config_fingerprint=fp,
            )
            session.add(inst)
            session.flush()
            added.append(str(inst.id))
        else:
            inst.config_fingerprint = fp
            session.add(inst)
        row_for_def[definition.id] = inst

    for def_id, inst in existing.items():
        if def_id in placed_ids:
            continue
        if inst.backend_status in _LIVE_BACKEND_STATUSES:
            refused.append({"instance_id": str(inst.id), "reason": "backend_busy"})
            continue
        session.delete(inst)
        removed.append(str(inst.id))

    assignments = [
        build_assignment_entry(session, row_for_def[d.id], d)
        for d in placed
        if d.id in row_for_def
    ]
    return PlacementDiff(
        added=added, removed=removed, refused=refused, assignments=assignments
    )


def _schedule_warm_up(scheduler: Any, agent_id: str) -> None:
    """Fire-and-forget proactive warm-up (the slice-4 seam). Never blocks the
    caller on a slow engine boot; failures are logged inside warm_up_agent."""
    task = asyncio.create_task(scheduler.warm_up_agent(agent_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def push_agent_assignments(
    agent_id: str, *, scheduler: Any = None
) -> PlacementDiff | None:
    """Reconcile one agent's backend rows to its current placement and, if it
    is connected, push ``agent.assignments.update`` over its socket.

    Returns the :class:`PlacementDiff` (DB outcome), or ``None`` when the agent
    no longer exists. A disconnected agent still gets its rows reconciled
    (ghosts pruned) but no frame is sent; a live socket that races away
    mid-push is treated the same (the next registration re-resolves). On a
    successful push that added backends, the scheduler is asked to warm the
    agent (bounded by VRAM + ``max_running_backends``).
    """
    from app.api.admin.providers import _placed_definitions

    agent_uuid = uuid.UUID(str(agent_id))
    with Session(engine) as session:
        agent = session.get(ProviderAgent, agent_uuid)
        if agent is None:
            return None
        placed = _placed_definitions(session, agent)
        connected = agent.websocket_connected
        diff = reconcile_agent_placement(
            session,
            agent,
            placed,
            create_new=connected,
        )
        ptype = session.exec(
            select(ProviderType).where(ProviderType.name == agent.provider_type)
        ).first()
        max_running = ptype.max_running_backends if ptype is not None else 0
        session.commit()
        assignments = diff.assignments

    if not connected:
        if diff.removed or diff.refused:
            logger.info(
                "reconciled placement for disconnected agent %s: removed=%d refused=%d",
                agent_id,
                len(diff.removed),
                len(diff.refused),
            )
        return diff

    payload = {"assignments": assignments, "max_running_backends": max_running}
    try:
        reply = await manager.send_command(
            str(agent_uuid),
            FrameKind.AGENT_ASSIGNMENTS_UPDATE,
            payload,
            timeout=ASSIGNMENTS_TIMEOUT_SECONDS,
        )
    except (ConnectionError, TimeoutError) as exc:
        # L3: a dead socket (ConnectionError) or a 30s stall (TimeoutError) both
        # mean "no ack" — skip cleanly; the rows are already reconciled and the
        # next registration/connect re-resolves.
        logger.info(
            "assignments push skipped for agent %s (no ack): %s",
            agent_id,
            exc,
        )
        return diff

    ok = bool(reply.payload.get("ok"))
    ack_added = list(reply.payload.get("added") or [])
    ack_removed = list(reply.payload.get("removed") or [])
    ack_refused = list(reply.payload.get("refused") or [])
    logger.info(
        "assignments.update pushed to agent %s: admin added=%d removed=%d refused=%d "
        "| provider ack ok=%s added=%s removed=%s refused=%s",
        agent_id,
        len(diff.added),
        len(diff.removed),
        len(diff.refused),
        ok,
        ack_added,
        ack_removed,
        ack_refused,
    )
    # M2: warm only backends the provider ACTUALLY added (ack_added ∩ diff.added).
    # A single-backend agent refuses adds it cannot host (ack_added empty) — no
    # warm-up, so no futile backend.start NAK churn for an unhostable id.
    ack_added_set = set(ack_added)
    confirmed_added = [i for i in diff.added if i in ack_added_set]
    if ok and confirmed_added and scheduler is not None:
        _schedule_warm_up(scheduler, str(agent_uuid))
    return diff


def _agent_ids_of_type(session: Session, provider_type: str) -> list[str]:
    return [
        str(a.id)
        for a in session.exec(
            select(ProviderAgent).where(ProviderAgent.provider_type == provider_type)
        ).all()
    ]


async def push_agent_assignments_for_type(
    provider_type: str, *, scheduler: Any = None
) -> None:
    """Push a placement refresh to every agent of ``provider_type``.

    Used after a definition create/PATCH/disable: the affected set is every
    agent whose type matches (``any_of_type`` defs target them all; a
    ``specific`` def only targets its links, but recomputing on the others is
    a correct no-op/prune — "keep it correct over clever"). Each agent is
    reconciled independently; one failure never aborts the rest.
    """
    with Session(engine) as session:
        agent_ids = _agent_ids_of_type(session, provider_type)
    for agent_id in agent_ids:
        try:
            await push_agent_assignments(agent_id, scheduler=scheduler)
        except Exception:  # noqa: BLE001 - one bad agent must not abort the fan-out
            logger.exception(
                "assignments push failed for agent %s (type %s)",
                agent_id,
                provider_type,
            )


async def push_agent_assignments_for_definition(
    definition_id: Any, *, scheduler: Any = None
) -> None:
    """Resolve a definition's type and refresh every agent of that type."""
    with Session(engine) as session:
        definition = session.get(ProviderDefinition, definition_id)
        if definition is None:
            return
        provider_type = definition.provider_type
    await push_agent_assignments_for_type(provider_type, scheduler=scheduler)


__all__ = [
    "PlacementDiff",
    "build_assignment_entry",
    "reconcile_agent_placement",
    "push_agent_assignments",
    "push_agent_assignments_for_type",
    "push_agent_assignments_for_definition",
]

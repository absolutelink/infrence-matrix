"""Phase 9 config-update flow: push ``provider.config.update`` to
connected provider instances and self-heal stale fingerprints.

When a ``ProviderDefinition``'s ``backend_config`` changes (admin PATCH)
— or when an instance reconnects carrying an older
``config_fingerprint`` than the definition's current one — the admin
pushes the new config over the provider WS and waits for the provider to
apply it (drain → stop → cache clear → apply → start → metadata
scrape; see ``provider_lib.config_update`` and docs/ws-protocol.md).

Retry policy (documented choice):
- The command uses a generous timeout (``CONFIG_UPDATE_TIMEOUT_SECONDS``,
  default 300s) because a backend drain + artifact download + boot can
  take minutes. Instances are pushed concurrently (``asyncio.gather``).
- A NAK with ``error == "backend_in_use"`` (drain refused) is retried up
  to ``CONFIG_UPDATE_RETRIES`` total attempts with
  ``CONFIG_UPDATE_RETRY_DELAY`` seconds between them. Any other failure
  (or timeout) is reported per-instance immediately — it is visible in
  the PATCH response but not fatal to the admin row; the operator can
  re-PATCH or let the reconnect self-heal retry.

The instance's ``config_fingerprint`` in Postgres is updated only when
the provider acks ok and echoes the new fingerprint back in its ack
detail, so the DB never claims a config the provider didn't apply.

Two distinct entry points:

- :func:`push_config_update` (PATCH-driven, operator changed the
  definition): fans out to **every** connected instance of the
  definition — each instance needs the new config.
- :func:`heal_stale_fingerprint` (self-heal, connect path + presence
  sweep): pushes to the **single stale instance only**, never the whole
  definition fan-out (a stale sibling must not disturb current ones).

Per-instance in-flight guard: a module-level ``set`` of instance ids
currently being pushed. The admin runs a single uvicorn worker, so an
in-process set is sufficient (and simpler than a Redis NX lease); it is
always cleared in a ``finally``, even on exception/cancel. A push that
arrives while another for the same instance is in flight is skipped
(heal returns False; the sweep retries on the next pass), which stops a
slow (300s) apply from being re-pushed every 30s sweep into the same
provider's command queue.
"""

import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlmodel import Session, col, select

from app.core.config import settings
from app.core.db import engine
from app.models import (
    ProviderDefinition,
    ProviderInstance,
    backend_config_is_authored,
)
from app.services.connection_manager import manager

logger = logging.getLogger("admin.config_update")

# Generous per-instance command timeout: backend drain + model download +
# boot can take minutes on cold hardware.
CONFIG_UPDATE_TIMEOUT_SECONDS = settings.CONFIG_UPDATE_TIMEOUT_SECONDS
# Drain-refused retry discipline (see module docstring).
CONFIG_UPDATE_RETRIES = settings.CONFIG_UPDATE_RETRIES
CONFIG_UPDATE_RETRY_DELAY = settings.CONFIG_UPDATE_RETRY_DELAY_SECONDS

# Per-instance in-flight push guard (see module docstring). The admin is
# a single uvicorn worker, so an in-process set is authoritative here.
_push_in_flight: set[str] = set()


def _build_payload(definition: ProviderDefinition, new_fp: str) -> dict[str, Any]:
    """The provider.config.update payload derived from a definition.

    Includes the fields the provider actually consumes:
    ``backend_config`` + ``config_fingerprint`` (restart-worthy change)
    and ``capacity`` (adopted at the provider without restart).
    ``idle_timeout_seconds`` is carried for observability only — the idle
    reaper is admin-side (``InferenceScheduler._idle_reaper``); the
    provider does not adopt it.

    Phase 14: never call this for a shell definition — callers must gate
    (a shell would otherwise push a fabricated ``{}``-config with the
    hash of empty space as its fingerprint).
    """
    return {
        "backend_config": definition.backend_config or {},
        "config_fingerprint": new_fp,
        "idle_timeout_seconds": definition.idle_timeout_seconds,
        "capacity": definition.capacity,
    }


@dataclass
class UpdateResult:
    """Per-instance outcome of one config.update push."""

    instance_id: str
    ok: bool
    noop: bool = False
    error: str | None = None
    step: str | None = None
    config_fingerprint: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "ok": self.ok,
            "noop": self.noop,
            "error": self.error,
            "step": self.step,
            "config_fingerprint": self.config_fingerprint,
        }


def _instance_fingerprint(definition: ProviderDefinition) -> str:
    from app.api.admin.providers import compute_config_fingerprint

    return compute_config_fingerprint(definition.backend_config or {})


def _connected_instances(
    session: Session, definition_id: Any
) -> list[ProviderInstance]:
    rows = session.exec(
        select(ProviderInstance).where(
            col(ProviderInstance.provider_definition_id) == definition_id,
            col(ProviderInstance.websocket_connected) == True,  # noqa: E712
        )
    ).all()
    result = []
    for row in rows:
        session.expunge(row)
        result.append(row)
    return result


async def _push_one(instance_id: str, payload: dict[str, Any]) -> UpdateResult:
    """Send config.update to one instance with the drain-refused retry loop."""
    last: UpdateResult | None = None
    for attempt in range(1, CONFIG_UPDATE_RETRIES + 1):
        try:
            reply = await manager.send_command(
                instance_id,
                "provider.config.update",
                payload,
                timeout=CONFIG_UPDATE_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - delivery failure
            logger.warning(
                "config.update delivery failed for instance %s: %s", instance_id, exc
            )
            return UpdateResult(
                instance_id=instance_id, ok=False, error=f"delivery failed: {exc}"
            )
        detail = reply.payload.get("detail") or {}
        if reply.payload.get("ok"):
            return UpdateResult(
                instance_id=instance_id,
                ok=True,
                noop=bool(detail.get("noop")),
                config_fingerprint=detail.get("config_fingerprint"),
                detail=detail,
            )
        error = reply.payload.get("error") or "provider refused"
        last = UpdateResult(
            instance_id=instance_id,
            ok=False,
            error=error,
            step=detail.get("step"),
            detail=detail,
        )
        if error != "backend_in_use" or attempt >= CONFIG_UPDATE_RETRIES:
            return last
        logger.info(
            "config.update drain refused for instance %s; retry %d/%d in %.0fs",
            instance_id,
            attempt + 1,
            CONFIG_UPDATE_RETRIES,
            CONFIG_UPDATE_RETRY_DELAY,
        )
        await asyncio.sleep(CONFIG_UPDATE_RETRY_DELAY)
    assert last is not None  # for type-checkers; loop always returns or sets
    return last


def _record_success(definition_id: Any, results: list[UpdateResult]) -> None:
    """Persist echoed fingerprints + discovered metadata after pushes."""
    with Session(engine) as session:
        definition = session.get(ProviderDefinition, definition_id)
        if definition is None:
            return
        for r in results:
            if not r.ok or not r.config_fingerprint:
                continue
            inst = session.get(ProviderInstance, uuid.UUID(r.instance_id))
            if inst is not None:
                inst.config_fingerprint = r.config_fingerprint
                session.add(inst)
        # Persist scraped model_metadata when the update carried any and
        # it differs from the stored value (registration does not
        # discover metadata; this is the place). The ack detail is the
        # driver's list_models() array; the DB column is a dict, so the
        # discovered objects are stored under the "models" key.
        for r in results:
            discovered = r.detail.get("model_metadata")
            if isinstance(discovered, list) and discovered:
                normalized = {"models": discovered}
                if definition.model_metadata != normalized:
                    definition.model_metadata = normalized
                    session.add(definition)
                    break
        session.commit()


async def _push_guarded(
    instance_id: str, payload: dict[str, Any]
) -> UpdateResult | None:
    """Push config.update to one instance under the in-flight guard.

    Returns ``None`` when a push to this instance is already in flight
    (the caller skips — never duplicates a push onto the same socket).
    The guard is always released, even on exception/cancellation.
    """
    if instance_id in _push_in_flight:
        logger.info(
            "config.update push already in flight for instance %s; skipping",
            instance_id,
        )
        return None
    _push_in_flight.add(instance_id)
    try:
        return await _push_one(instance_id, payload)
    finally:
        _push_in_flight.discard(instance_id)


async def push_config_update(definition: ProviderDefinition) -> list[UpdateResult]:
    """Push the definition's current backend_config to every connected
    instance; returns per-instance results (never raises for a failed
    provider — failures are reported, not fatal).

    This is the PATCH-driven fan-out entry point. Instances with another
    push already in flight are skipped (omitted from the results) rather
    than queued behind it.
    """
    from app.api.admin.providers import compute_config_fingerprint

    # Phase 14: shells are never pushed (nothing authored to apply).
    if not backend_config_is_authored(definition):
        return []
    new_fp = compute_config_fingerprint(definition.backend_config or {})
    with Session(engine) as session:
        instances = _connected_instances(session, definition.id)

    if not instances:
        return []

    payload = _build_payload(definition, new_fp)
    pushed = await asyncio.gather(
        *(_push_guarded(str(inst.id), payload) for inst in instances)
    )
    results = [r for r in pushed if r is not None]
    _record_success(definition.id, results)
    return results


async def heal_stale_fingerprint(instance_id: str) -> bool:
    """Self-heal one (re)connected instance whose stored fingerprint is
    stale w.r.t. its definition's current backend_config.

    Pushes to the **stale instance only** (never the definition-wide
    fan-out — sibling instances with a current fingerprint must not get
    churned by a heal). Cheap: one DB read; the push only happens when
    connected and mismatched. Returns True when a config.update was
    pushed (not when it was a no-op fingerprint match or a push to this
    instance was already in flight). Used from the WS connect path
    (background task) and the presence sweep.
    """

    with Session(engine) as session:
        inst = session.get(ProviderInstance, uuid.UUID(str(instance_id)))
        if inst is None or not inst.websocket_connected:
            return False
        definition = session.get(ProviderDefinition, inst.provider_definition_id)
        if definition is None:
            return False
        # Phase 14: a shell definition never heals — there is no config
        # whose fingerprint could be stale (and fabricating `{}`'s hash
        # would push an empty config that the provider must not adopt).
        if not backend_config_is_authored(definition):
            return False
        current_fp = _instance_fingerprint(definition)
        if inst.config_fingerprint == current_fp:
            return False
        payload = _build_payload(definition, current_fp)

    logger.info(
        "stale fingerprint for instance %s (%s != %s); pushing config.update",
        instance_id,
        inst.config_fingerprint,
        current_fp,
    )
    result = await _push_guarded(instance_id, payload)
    if result is None:
        return False
    _record_success(definition.id, [result])
    return True


async def heal_stale_fingerprint_safe(instance_id: str) -> None:
    """Fire-and-forget wrapper: never let a heal break the connect path."""
    with contextlib.suppress(Exception):
        await heal_stale_fingerprint(instance_id)

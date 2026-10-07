"""Admin provider-backend (instance) reads + actions (Phase 9 / Phase 10 UI).

Phase 16: a ``ProviderInstance`` is one backend hosted by a ``ProviderAgent``.
The agent holds the single WebSocket, so every per-backend action addresses
the owning agent's socket and carries the target ``instance_id`` in its
payload. Container-level state (liveness, version, agent_status, epoch,
reported schema fingerprint) lives on the agent and is denormalized here for
the tables.

Reads (Phase 10 UI):

- ``GET /admin/api/instances`` — every backend with its machine uid and
  definition alias denormalized for tables.
- ``GET /admin/api/instances/{id}`` — single backend.

Actions over the provider WS:

- ``POST /admin/api/instances/{id}/cache/clear`` — send ``cache.clear``
  (prompt-cache dirs only, never model files; see
  ``provider_lib.config_update`` and provider/README.md).
- ``POST /admin/api/instances/{id}/storage/prune`` — send
  ``storage.prune_unused`` (delete MODELS_DIR files not referenced by
  the driver's resolved artifact set; ``{"dry_run": true}` supported).
- ``POST /admin/api/instances/{id}/backend/start|stop|restart`` and
  ``/initialize`` — manual backend control (``provider_lib.ops``).
  Start and restart default to **fire-and-forget** (202): the provider
  acks "accepted" and the boot runs in the background, because a cold
  halogen-flash boot downloads its checkpoint and companions first —
  tens of GB, tens of minutes — and no HTTP request should sit open for
  that. Watch ``backend_status`` on the instance (``initializing``
  heartbeats + ``running``/``error`` land over the WS) and the log tail.
  Pass ``{"wait_for_running": true}`` to block until the backend reports
  running (bounded by ``BACKEND_BOOT_TIMEOUT_SECONDS``). ``stop`` always
  awaits — it is quick. ``initialize`` re-registers (fresh agent secret +
  definition config), then reboots and re-publishes model metadata.

All actions require the backend's agent WebSocket to be connected; otherwise
409. The provider's ack detail is returned verbatim (deleted paths, bytes
freed, capacity, ports, accepted/backend_status).
"""

import logging
import uuid
from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.admin.serializers import iso_utc
from app.core.config import settings
from app.core.db import get_session
from app.core.redis import get_redis
from app.models import ProviderInstance
from app.services import log_store
from app.services.connection_manager import manager

logger = logging.getLogger("admin.instances")

router = APIRouter(prefix="/admin/api/instances", tags=["admin"])

# Cache clears are local file deletes; prune may stat/delete a large
# models dir over slow storage, so it gets the longer budget (N-8).
ACTION_TIMEOUT_SECONDS = 60.0
PRUNE_TIMEOUT_SECONDS = 300.0


class ActionBody(BaseModel):
    dry_run: bool = False
    # cache.clear only: override the backend_in_use refusal (clearing
    # engine caches while requests are live may cause I/O errors).
    force: bool = False


class BackendActionBody(BaseModel):
    """`wait_for_running`: hold the ack until the backend reports running.

    Default false (202-style): the provider starts the boot in the
    background and the caller polls the instance. True mirrors the
    scheduler's contract and is bounded by
    ``settings.BACKEND_BOOT_TIMEOUT_SECONDS`` — long enough for a cold
    engine-side download, but it does keep the request open that whole time.
    """

    wait_for_running: bool = False


def instance_dict(inst: ProviderInstance) -> dict[str, Any]:
    agent = inst.agent
    machine = inst.machine
    return {
        "id": str(inst.id),
        "agent_id": str(inst.agent_id),
        "machine_id": str(agent.machine_id) if agent else None,
        "machine_uid": machine.uid if machine else None,
        "machine_name": machine.name if machine else None,
        "provider_definition_id": str(inst.provider_definition_id),
        "alias": (inst.provider_definition.alias if inst.provider_definition else None),
        "provider_type": (
            inst.provider_definition.provider_type if inst.provider_definition else None
        ),
        "port": inst.port,
        # Container-level state (denormalized from the owning agent).
        "version": agent.version if agent else None,
        "agent_status": agent.agent_status if agent else None,
        "websocket_connected": agent.websocket_connected if agent else False,
        "epoch": agent.epoch if agent else 0,
        "last_seen": iso_utc(agent.last_seen) if agent else None,
        "reported_schema_fingerprint": (
            agent.reported_schema_fingerprint if agent else None
        ),
        "backend_status": inst.backend_status,
        "last_request_at": iso_utc(inst.last_request_at),
        # Idle-reaper load clock: when this backend entered running/in_use.
        "backend_loaded_at": iso_utc(inst.backend_loaded_at),
        "config_fingerprint": inst.config_fingerprint,
        "assigned_gpus": inst.assigned_gpus,
        "error_message": inst.error_message,
        "created_at": iso_utc(inst.created_at),
    }


@router.get("")
def list_instances(session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    instances = session.exec(
        select(ProviderInstance).order_by(ProviderInstance.created_at)
    ).all()
    return [instance_dict(i) for i in instances]


@router.get("/{instance_id}")
def get_instance(
    instance_id: str, session: Session = Depends(get_session)
) -> dict[str, Any]:
    try:
        inst = session.get(ProviderInstance, uuid.UUID(instance_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="instance not found") from None
    if inst is None:
        raise HTTPException(status_code=404, detail="instance not found")
    return instance_dict(inst)


def _get_connected_instance(session: Session, instance_id: str) -> ProviderInstance:
    try:
        inst = session.get(ProviderInstance, uuid.UUID(instance_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="instance not found") from None
    if inst is None:
        raise HTTPException(status_code=404, detail="instance not found")
    if inst.agent is None or not inst.agent.websocket_connected:
        raise HTTPException(
            status_code=409,
            detail=f"instance {instance_id} has no live websocket connection",
        )
    return inst


async def _send_action(
    inst: ProviderInstance,
    command: str,
    payload: dict[str, Any],
    timeout: float = ACTION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Send a per-backend command over the owning agent's socket, tagging the
    payload with the target ``instance_id``."""
    body = {**payload, "instance_id": str(inst.id)}
    try:
        reply = await manager.send_command(
            str(inst.agent_id), command, body, timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"{command} delivery failed: {exc}"
        ) from exc
    ok = bool(reply.payload.get("ok"))
    detail = reply.payload.get("detail") or {}
    if not ok:
        error = {
            "error": reply.payload.get("error") or "provider refused",
            "step": detail.get("step"),
        }
        # backend_in_use NAKs carry retry_after (docs/ws-protocol.md §4);
        # surface it so the UI can show "retry in Ns".
        if detail.get("retry_after") is not None:
            error["retry_after"] = detail["retry_after"]
        raise HTTPException(status_code=502, detail=error)
    return detail


@router.post("/{instance_id}/cache/clear")
async def clear_cache(
    instance_id: str,
    body: ActionBody | None = None,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    inst = _get_connected_instance(session, instance_id)
    dry_run = bool(body and body.dry_run)
    force = bool(body and body.force)
    detail = await _send_action(
        inst, "cache.clear", {"dry_run": dry_run, "force": force}
    )
    logger.info(
        "cache.clear on instance %s: freed %s bytes (dry_run=%s)",
        instance_id,
        detail.get("bytes_freed"),
        dry_run,
    )
    return {"ok": True, "instance_id": instance_id, **detail}


@router.post("/{instance_id}/storage/prune")
async def prune_storage(
    instance_id: str,
    body: ActionBody | None = None,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    inst = _get_connected_instance(session, instance_id)
    dry_run = bool(body and body.dry_run)
    detail = await _send_action(
        inst,
        "storage.prune_unused",
        {"dry_run": dry_run},
        timeout=PRUNE_TIMEOUT_SECONDS,
    )
    logger.info(
        "storage.prune_unused on instance %s: freed %s bytes (dry_run=%s)",
        instance_id,
        detail.get("bytes_freed"),
        dry_run,
    )
    return {"ok": True, "instance_id": instance_id, **detail}


@router.post("/{instance_id}/backend/start", status_code=202)
async def start_backend(
    instance_id: str,
    body: BackendActionBody | None = None,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Boot this backend by hand (no inference request needed).

    202 by default: the provider acks "accepted" and boots in the
    background, heartbeating ``backend.status`` (`initializing` while the
    engine downloads/loads, then `running` or `error`). Poll
    ``GET /admin/api/instances/{id}`` — or pass ``wait_for_running: true``
    to block until it is up.
    """
    inst = _get_connected_instance(session, instance_id)
    wait = bool(body and body.wait_for_running)
    detail = await _send_action(
        inst,
        "backend.start",
        {"wait_for_running": wait},
        timeout=(
            settings.BACKEND_BOOT_TIMEOUT_SECONDS if wait else ACTION_TIMEOUT_SECONDS
        ),
    )
    logger.info(
        "manual backend.start on instance %s (wait_for_running=%s): %s",
        instance_id,
        wait,
        detail.get("backend_status"),
    )
    return {"ok": True, "instance_id": instance_id, **detail}


@router.post("/{instance_id}/backend/stop")
async def stop_backend(
    instance_id: str,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Unload the backend (always awaited — a stop is quick).

    Refused (502 with ``retry_after``) while live requests hold slots: the
    provider does not cancel in-flight streams. Unload during a long
    download is fine and is the escape hatch for a boot that will not
    finish.
    """
    inst = _get_connected_instance(session, instance_id)
    detail = await _send_action(
        inst,
        "backend.stop",
        {},
        timeout=settings.BACKEND_STOP_TIMEOUT_SECONDS,
    )
    logger.info("manual backend.stop on instance %s: %s", instance_id, detail)
    return {"ok": True, "instance_id": instance_id, **detail}


@router.post("/{instance_id}/backend/restart", status_code=202)
async def restart_backend(
    instance_id: str,
    body: BackendActionBody | None = None,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Stop then start the backend (config re-applied by the driver)."""
    inst = _get_connected_instance(session, instance_id)
    wait = bool(body and body.wait_for_running)
    detail = await _send_action(
        inst,
        "backend.restart",
        {"wait_for_running": wait},
        timeout=(
            settings.BACKEND_BOOT_TIMEOUT_SECONDS if wait else ACTION_TIMEOUT_SECONDS
        ),
    )
    logger.info(
        "manual backend.restart on instance %s (wait_for_running=%s): %s",
        instance_id,
        wait,
        detail.get("backend_status"),
    )
    return {"ok": True, "instance_id": instance_id, **detail}


@router.post("/{instance_id}/initialize", status_code=202)
async def initialize_instance(
    instance_id: str,
    body: BackendActionBody | None = None,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Re-run the whole init lifecycle: re-register, adopt, boot, re-scrape.

    ``provider.initialize`` POSTs a fresh registration (re-checking the
    version + schema gates, re-adopting capacity/``backend_config``, minting
    a new agent secret for future reconnects and rewriting
    ``provider_config.json``), then drain-stops, boots and publishes
    ``backend.metadata`` from the running engine. The boot is asynchronous
    (202): a cold halogen-flash backend downloads its checkpoint plus
    companions before the API answers, which is far beyond an HTTP request's
    patience. ``wait_for_running: true`` waits for the boot instead.
    """
    inst = _get_connected_instance(session, instance_id)
    wait = bool(body and body.wait_for_running)
    detail = await _send_action(
        inst,
        "provider.initialize",
        {"wait_for_running": wait},
        timeout=(
            settings.BACKEND_BOOT_TIMEOUT_SECONDS if wait else ACTION_TIMEOUT_SECONDS
        ),
    )
    logger.info(
        "provider.initialize on instance %s (wait_for_running=%s): %s",
        instance_id,
        wait,
        detail.get("backend_status"),
    )
    return {"ok": True, "instance_id": instance_id, **detail}


@router.get("/{instance_id}/logs")
async def get_instance_logs(
    instance_id: str,
    kind: str = Query("backend", pattern="^(backend|provider|all)$"),
    since: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
    session: Session = Depends(get_session),
    redis_client: aioredis.Redis = Depends(get_redis),
) -> dict[str, Any]:
    """Phase 13 log tail (Redis-backed, cursor-based).

    Entries newest-first; pass the returned ``cursor`` back as
    ``since=`` to tail. ``kind=all`` merges backend + provider by the
    ingest-assigned monotonic ``seq``. Empty (not an error) when Redis
    holds nothing yet; 404 only when the instance is unknown.
    """
    try:
        inst = session.get(ProviderInstance, uuid.UUID(instance_id))
    except ValueError:
        inst = None
    if inst is None:
        raise HTTPException(status_code=404, detail="instance not found")
    return await log_store.read_logs(
        redis_client,
        instance_id,
        kind=kind,
        since=since,
        limit=limit,
        provider_id=str(inst.agent_id),
    )

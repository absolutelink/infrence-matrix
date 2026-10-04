"""Admin provider-instance actions (Phase 9).

Small operator/UI-facing pushes over the provider WS:

- ``POST /admin/api/instances/{id}/cache/clear`` — send ``cache.clear``
  (prompt-cache dirs only, never model files; see
  ``provider_lib.config_update`` and provider/README.md).
- ``POST /admin/api/instances/{id}/storage/prune`` — send
  ``storage.prune_unused`` (delete MODELS_DIR files not referenced by
  the driver's resolved artifact set; ``{"dry_run": true}` supported).

Both require the instance's WebSocket to be connected; otherwise 409.
The provider's ack detail is returned verbatim (deleted paths, bytes
freed, kept list).
"""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from app.core.db import get_session
from app.models import ProviderInstance
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


def _get_connected_instance(session: Session, instance_id: str) -> ProviderInstance:
    try:
        inst = session.get(ProviderInstance, uuid.UUID(instance_id))
    except ValueError:
        raise HTTPException(status_code=404, detail="instance not found") from None
    if inst is None:
        raise HTTPException(status_code=404, detail="instance not found")
    if not inst.websocket_connected:
        raise HTTPException(
            status_code=409,
            detail=f"instance {instance_id} has no live websocket connection",
        )
    return inst


async def _send_action(
    instance_id: str,
    command: str,
    payload: dict[str, Any],
    timeout: float = ACTION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    try:
        reply = await manager.send_command(
            instance_id, command, payload, timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"{command} delivery failed: {exc}"
        ) from exc
    ok = bool(reply.payload.get("ok"))
    detail = reply.payload.get("detail") or {}
    if not ok:
        raise HTTPException(
            status_code=502,
            detail={
                "error": reply.payload.get("error") or "provider refused",
                "step": detail.get("step"),
            },
        )
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
        str(inst.id), "cache.clear", {"dry_run": dry_run, "force": force}
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
        str(inst.id),
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

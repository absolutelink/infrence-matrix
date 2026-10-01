"""Proxy endpoints for llama.cpp API."""

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.routes.servers import server_manager
from app.core.config import settings
from app.services.inference_operations import (
    cancel_operation,
    get_operation,
    list_operations,
    operation_exists,
)
from app.services.proxy import ProxyOperationExists, ServerProxy

router = APIRouter(prefix="/proxy", tags=["proxy"])

# One shared proxy (and therefore one bounded connection pool) for the whole
# agent process. Per-request clients leaked a socket per call, and halogen-flash
# keeps HTTP/1.1 connections alive, so the leaked sockets eventually starved
# new inference dispatch of a descriptor against the single upstream listener.
proxy = ServerProxy(server_manager)


async def close_proxy() -> None:
    """Release the shared proxy's pooled sockets at shutdown."""
    if not proxy.client.is_closed:
        await proxy.aclose()


# llama.cpp error bodies carry their own {"error": {...}} envelope; return
# them with the upstream status so clients can distinguish failures instead
# of parsing an error object as a completion (which yields empty output).
STATUS_ERRORS = (httpx.HTTPStatusError,)
HALOGEN_FLASH_PATHS = {
    "health",
    "metrics",
    "cache",
    "v1/models",
    "v1/completions",
    "v1/chat/completions",
}
GUFO_PATHS = {
    "health",
    "ready",
    "metrics",
    "v1/models",
    "v1/completions",
    "v1/chat/completions",
    "v1/responses",
}


def _is_supported_halogen_flash_path(path: str) -> bool:
    return path in HALOGEN_FLASH_PATHS


def _is_supported_gufo_path(path: str) -> bool:
    return path in GUFO_PATHS


@router.get("/{server_id}/operations/{request_id}")
async def get_inference_operation(server_id: str, request_id: str) -> dict:
    operation = get_operation(server_manager, request_id, server_id)
    if operation is None:
        return {"request_id": request_id, "status": "unknown"}
    return operation


@router.get("/{server_id}/operations")
async def list_inference_operations(server_id: str) -> dict:
    return {
        "complete": True,
        "operations": list_operations(server_manager, server_id),
    }


@router.post("/{server_id}/operations/{request_id}/cancel")
async def cancel_inference_operation(
    server_id: str,
    request_id: str,
    reservation_id: str | None = Header(
        default=None, alias="X-Inference-Reservation-ID"
    ),
) -> dict:
    operation = get_operation(server_manager, request_id, server_id)
    stored_reservation = (
        operation.get("reservation_id") if operation is not None else None
    )
    # A reservation_id mismatch is only authoritative when the operation was
    # actually created through the reservation protocol. The plain proxy dispatch
    # path (_connection_started -> activate_operation) never stores a reservation
    # token, so a running ``active`` op there has no ``reservation_id`` at all.
    # Rejecting those cancels (``None != <uuid>``) left the bound llama.cpp stream
    # draining to completion after the downstream client was gone, holding the
    # inference slot. The lease owner is the only caller that knows the request_id,
    # so it must be able to cancel a token-less running operation.
    if stored_reservation is not None and stored_reservation != reservation_id:
        return {
            "request_id": request_id,
            "status": operation["status"],
            "cancelled": False,
        }
    if operation is not None and operation["status"] in {"reserved", "active"}:
        accepted = await proxy.release_pending_reservation(
            server_id, request_id, operation["slot_generation"], reservation_id
        )
        if accepted:
            return {"request_id": request_id, "status": "cancelled", "cancelled": True}
    accepted = cancel_operation(server_manager, request_id, server_id)
    operation = get_operation(server_manager, request_id, server_id)
    return {
        "request_id": request_id,
        "status": operation["status"] if operation else "unknown",
        "cancelled": accepted,
    }


@router.post("/{server_id}/reservations/{request_id}")
async def reserve_inference_operation(
    server_id: str,
    request_id: str,
    slot_generation: int = Header(alias="X-Inference-Slot-Generation"),
    reservation_id: str = Header(alias="X-Inference-Reservation-ID"),
) -> dict:
    if server_id not in server_manager.servers:
        raise HTTPException(404, "Server not found")
    try:
        accepted = await proxy.try_reserve_inference_slot(
            server_id, request_id, slot_generation, reservation_id
        )
    except ProxyOperationExists as exc:
        raise HTTPException(409, "Reservation already exists") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(409, "Server generation changed") from exc
    return {"accepted": accepted, "request_id": request_id}


@router.api_route(
    "/{server_id}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    response_model=None,
)
async def proxy_request(
    server_id: str,
    path: str,
    request: Request,
    slot_generation: int | None = Header(
        default=None, alias="X-Inference-Slot-Generation"
    ),
    inference_request_id: str | None = Header(
        default=None, alias="X-Inference-Request-ID"
    ),
    inference_reservation: str | None = Header(
        default=None, alias="X-Inference-Reservation"
    ),
    reservation_id: str | None = Header(
        default=None, alias="X-Inference-Reservation-ID"
    ),
) -> StreamingResponse | dict:
    """Proxy request to llama.cpp server."""
    if server_id not in server_manager.servers:
        raise HTTPException(404, "Server not found")
    live_generation = server_manager.configs[server_id].slot_generation
    if slot_generation is not None and slot_generation != live_generation:
        raise HTTPException(409, "Inference slot generation does not match live server")

    if settings.AGENT_PLATFORM == "halogen-flash":
        if not _is_supported_halogen_flash_path(path):
            raise HTTPException(
                404,
                "Endpoint is not supported by halogen-flash-server",
            )

    if settings.AGENT_PLATFORM == "gufo":
        if not _is_supported_gufo_path(path):
            raise HTTPException(
                404,
                "Endpoint is not supported by gufo",
            )

    inference_request = path in {
        "v1/chat/completions",
        "v1/completions",
        "v1/embeddings",
    }
    if (
        inference_request
        and inference_request_id is not None
        and inference_reservation is None
        and operation_exists(server_manager, inference_request_id)
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Inference operation {inference_request_id} already exists",
        )

    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.json()

    if inference_reservation is not None:
        if (
            not inference_request
            or inference_request_id != inference_reservation
            or slot_generation is None
            or reservation_id is None
            or not await proxy.consume_inference_slot(
                server_id, inference_reservation, slot_generation, reservation_id
            )
        ):
            raise HTTPException(409, "Invalid or expired inference reservation")

    if path.startswith("stream") or (body and body.get("stream")):
        return StreamingResponse(
            proxy.proxy_stream_background(
                server_id,
                request.method,
                f"/{path}",
                body,
                expected_slot_generation=slot_generation,
                enforce_capacity=inference_request,
                capacity_reserved=inference_reservation is not None,
                operation_id=inference_request_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        try:
            response = await proxy.proxy_request(
                server_id,
                request.method,
                f"/{path}",
                dict(request.headers),
                body,
                expected_slot_generation=slot_generation,
                enforce_capacity=inference_request,
                capacity_reserved=inference_reservation is not None,
                operation_id=inference_request_id,
            )
        except ProxyOperationExists as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                raise HTTPException(409, "Inference slot generation changed") from exc
            raise
        if response.is_error:
            # Relay the upstream status + error body (llama.cpp {"error"})
            detail: dict | str
            try:
                detail = response.json()
            except Exception:
                detail = response.text or "llama-server error"
            raise HTTPException(status_code=response.status_code, detail=detail)
        if path == "metrics":
            return {"metrics": response.text}
        return response.json()

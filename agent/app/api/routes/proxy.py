"""Proxy endpoints for llama.cpp API."""

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.routes.servers import server_manager
from app.services.proxy import ServerProxy

router = APIRouter(prefix="/proxy", tags=["proxy"])

# llama.cpp error bodies carry their own {"error": {...}} envelope; return
# them with the upstream status so clients can distinguish failures instead
# of parsing an error object as a completion (which yields empty output).
STATUS_ERRORS = (httpx.HTTPStatusError,)


@router.api_route(
    "/{server_id}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    response_model=None,
)
async def proxy_request(
    server_id: str, path: str, request: Request
) -> StreamingResponse | dict:
    """Proxy request to llama.cpp server."""
    if server_id not in server_manager.servers:
        raise HTTPException(404, "Server not found")

    proxy = ServerProxy(server_manager)

    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.json()

    if path.startswith("stream") or (body and body.get("stream")):
        return StreamingResponse(
            proxy.proxy_stream(server_id, request.method, f"/{path}", body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        response = await proxy.proxy_request(
            server_id, request.method, f"/{path}", dict(request.headers), body
        )
        if response.is_error:
            # Relay the upstream status + error body (llama.cpp {"error"})
            detail: dict | str
            try:
                detail = response.json()
            except Exception:
                detail = response.text or "llama-server error"
            raise HTTPException(status_code=response.status_code, detail=detail)
        return response.json()

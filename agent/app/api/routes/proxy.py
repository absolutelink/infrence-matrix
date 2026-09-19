"""Proxy endpoints for llama.cpp API."""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import httpx

from app.api.routes.servers import server_manager
from app.services.proxy import LlamaCppProxy

router = APIRouter(prefix="/proxy", tags=["proxy"])


@router.api_route("/{server_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_request(
    server_id: str,
    path: str,
    request: Request
) -> StreamingResponse | dict:
    """Proxy request to llama.cpp server."""
    if server_id not in server_manager.servers:
        raise HTTPException(404, "Server not found")

    proxy = LlamaCppProxy(server_manager)

    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.json()

    if path.startswith("stream") or (body and body.get("stream")):
        return StreamingResponse(
            proxy.proxy_stream(
                server_id,
                request.method,
                f"/{path}",
                body
            ),
            media_type="text/event-stream"
        )
    else:
        response = await proxy.proxy_request(
            server_id,
            request.method,
            f"/{path}",
            dict(request.headers),
            body
        )
        return response.json()

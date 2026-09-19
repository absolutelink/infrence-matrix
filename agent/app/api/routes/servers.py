"""Server management endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid

from app.services.llama_server import LlamaServerManager, ServerConfig

router = APIRouter(prefix="/servers", tags=["servers"])
server_manager = LlamaServerManager()


class StartServerRequest(BaseModel):
    model_id: str
    model_path: str
    config: dict


class StartServerResponse(BaseModel):
    server_id: str
    status: str
    proxy_url: str


@router.post("/start", response_model=StartServerResponse)
async def start_server(request: StartServerRequest) -> StartServerResponse:
    """Start a llama.cpp server."""
    server_id = str(uuid.uuid4())

    config = ServerConfig(
        model_path=request.model_path,
        port=8081,
        **request.config
    )

    success = await server_manager.start_server(server_id, config)

    if not success:
        raise HTTPException(400, "Server already running")

    return StartServerResponse(
        server_id=server_id,
        status="running",
        proxy_url=f"http://localhost:{config.port}"
    )


@router.post("/{server_id}/stop")
async def stop_server(server_id: str, force: bool = False) -> dict:
    """Stop a llama.cpp server."""
    success = await server_manager.stop_server(server_id, force)

    if not success:
        raise HTTPException(404, "Server not found")

    return {"status": "stopped"}


@router.get("")
async def list_servers() -> dict:
    """List all running servers."""
    servers = server_manager.list_servers()
    return {"servers": servers}

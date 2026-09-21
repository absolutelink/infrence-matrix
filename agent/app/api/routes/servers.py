"""Server management API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from app.services.llama_server import llama_server_manager
from app.services.model_manager import model_manager

router = APIRouter(prefix="/servers", tags=["servers"])


class ServerConfig(BaseModel):
    id: str
    model_path: str
    port: int
    gpu_layers: int = 35
    context_size: int = 4096
    batch_size: int = 512
    cache_prompt: bool = True
    flash_attn: bool = True


class ServerStartRequest(BaseModel):
    config: ServerConfig


class ServerStopRequest(BaseModel):
    server_id: str


@router.post("/start")
async def start_server(request: ServerStartRequest) -> dict:
    """Start a llama.cpp server."""
    try:
        # Validate model exists
        if not model_manager.model_exists(request.config.model_path):
            raise HTTPException(status_code=404, detail="Model not found")

        # Start the server
        config = llama_server_manager.ServerConfig(
            model_path=request.config.model_path,
            port=request.config.port,
            gpu_layers=request.config.gpu_layers,
            context_size=request.config.context_size,
            batch_size=request.config.batch_size,
            cache_prompt=request.config.cache_prompt,
            flash_attn=request.config.flash_attn
        )
        
        success = await llama_server_manager.start_server(request.config.id, config)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start server")

        return {"status": "started", "server_id": request.config.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_server(request: ServerStopRequest) -> dict:
    """Stop a llama.cpp server."""
    try:
        success = await llama_server_manager.stop_server(request.server_id)
        if not success:
            raise HTTPException(status_code=404, detail="Server not found")

        return {"status": "stopped", "server_id": request.server_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_servers() -> dict:
    """List all running servers."""
    servers = llama_server_manager.list_servers()
    return {"servers": servers}


@router.get("/status/{server_id}")
async def get_server_status(server_id: str) -> dict:
    """Get status of a specific server."""
    servers = llama_server_manager.list_servers()
    for server in servers:
        if server["server_id"] == server_id:
            return {"status": "running", "server": server}
    
    return {"status": "stopped", "server_id": server_id}
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
import asyncio
import subprocess
import signal
import time

from app.core.config import settings
from app.core.logging import logger

router = APIRouter(prefix="/servers", tags=["servers"])


class ServerConfig(BaseModel):
    model_path: str
    gpu_layers: int = settings.DEFAULT_GPU_LAYERS
    context_size: int = settings.DEFAULT_CONTEXT_SIZE
    batch_size: int = settings.DEFAULT_BATCH_SIZE
    cache_prompt: bool = True


class StartServerRequest(BaseModel):
    model_id: str
    model_path: str
    config: ServerConfig


class StartServerResponse(BaseModel):
    server_id: str
    status: str
    proxy_url: str


# In-memory server tracking
running_servers: dict[str, dict] = {}


@router.post("/start", response_model=StartServerResponse)
async def start_server(request: StartServerRequest) -> StartServerResponse:
    """Start a llama.cpp server."""
    server_id = str(uuid.uuid4())

    logger.info(f"Starting server {server_id} for model {request.model_path}")

    # Build command
    cmd = [
        settings.LLAMA_SERVER_PATH,
        "--model", request.model_path,
        "--port", "8081",
        "--n-gpu-layers", str(request.config.gpu_layers),
        "--ctx-size", str(request.config.context_size),
        "--batch-size", str(request.config.batch_size),
    ]

    if request.config.cache_prompt:
        cmd.extend(["--prompt-cache", f"{settings.CACHE_PATH}/{server_id}.cache"])

    # Start subprocess
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    await asyncio.sleep(5)

    # Check if server is healthy
    # TODO: Implement health check

    running_servers[server_id] = {
        "model_id": request.model_id,
        "model_path": request.model_path,
        "port": 8081,
        "process": proc,
        "started_at": time.time(),
        "config": request.config.dict(),
    }

    logger.info(f"Server {server_id} started on port 8081")

    return StartServerResponse(
        server_id=server_id,
        status="running",
        proxy_url=f"http://localhost:8080/proxy/{server_id}",
    )


@router.post("/{server_id}/stop")
async def stop_server(server_id: str, force: bool = False) -> dict:
    """Stop a llama.cpp server."""
    if server_id not in running_servers:
        raise HTTPException(404, "Server not found")

    server_info = running_servers[server_id]
    proc = server_info["process"]

    logger.info(f"Stopping server {server_id}")

    if force:
        proc.kill()
    else:
        proc.send_signal(signal.SIGTERM)

    proc.wait(timeout=30)

    del running_servers[server_id]

    logger.info(f"Server {server_id} stopped")

    return {"status": "stopped"}


@router.get("")
async def list_servers() -> dict:
    """List all running servers."""
    servers = []

    for server_id, info in running_servers.items():
        uptime = time.time() - info["started_at"]
        servers.append({
            "server_id": server_id,
            "model_id": info["model_id"],
            "model_path": info["model_path"],
            "port": info["port"],
            "status": "running",
            "uptime_seconds": int(uptime),
        })

    return {"servers": servers}

"""Server management API endpoints."""

import socket
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.logging import logger
from app.services.llama_server import ServerConfig, llama_server_manager
from app.services.model_manager import model_manager

router = APIRouter(prefix="/servers", tags=["servers"])

server_manager = llama_server_manager

# Range for ephemeral llama-server ports when the caller doesn't pin one
EPHEMERAL_PORT_START = 8090
EPHEMERAL_PORT_END = 8190


class ServerSpec(BaseModel):
    id: str
    model_path: str
    # Omit to let the agent allocate a free random port
    port: int | None = None
    gpu_layers: int = 35
    context_size: int = 4096
    batch_size: int = 512
    cache_prompt: bool = True
    # None lets llama.cpp decide; True/False maps to --flash-attn on/off
    flash_attn: bool | None = True
    # Jinja chat templates (required for tool calling); default on
    jinja: bool = True
    # Multimodal projector (vision GGUF); resolved like model_path
    mmproj_path: str | None = None


class ServerStartRequest(BaseModel):
    config: ServerSpec
    # If provided and the model file is missing locally, the agent downloads
    # the file from the source repo before starting the server.
    source: dict[str, str] | None = None
    # Same shape as source, for the mmproj projector (downloaded on demand)
    mmproj_source: dict[str, str] | None = None


class ServerStopRequest(BaseModel):
    server_id: str


def _allocate_port() -> int:
    """Pick a random free port in the ephemeral llama-server range."""
    import random

    candidates = list(range(EPHEMERAL_PORT_START, EPHEMERAL_PORT_END))
    random.shuffle(candidates)
    for port in candidates:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("0.0.0.0", port))
            return port
        except OSError:
            continue
    raise HTTPException(
        status_code=503,
        detail=f"No free ports in {EPHEMERAL_PORT_START}-{EPHEMERAL_PORT_END}",
    )


def _resolve_model_path(model_path: str) -> str:
    """Accept absolute paths or bare filenames under MODELS_PATH."""
    if model_path.startswith("/"):
        return model_path
    return str(model_manager.models_path / model_path)


def _model_candidates(model_path: str, filename: str) -> list[Path]:
    """Possible local locations for a model file.

    Models registered from HuggingFace use paths like
    ``/models/{repo}/{filename}`` while plain downloads land directly in
    MODELS_PATH, so check both locations.
    """
    candidates = []
    if model_path.startswith("/"):
        candidates.append(Path(model_path))
    candidates.append(model_manager.models_path / model_path.lstrip("/"))
    candidates.append(model_manager.models_path / filename)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            unique.append(c)
    return unique


async def _ensure_model(
    model_path: str,
    source: dict[str, str] | None,
) -> str:
    """Ensure the model file exists locally, downloading it if needed.

    Returns the absolute path to the model file.
    """
    path = _resolve_model_path(model_path)
    filename = path.rsplit("/", 1)[-1]

    candidates = _model_candidates(model_path, filename)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Not on disk: download it via hf_hub (lands under MODELS_PATH).
    if not source or not source.get("repo_id") or not source.get("filename"):
        raise HTTPException(
            status_code=404,
            detail=f"Model file {filename} not found on agent and no download source provided",
        )

    logger.info(f"Model {filename} missing; downloading from {source['repo_id']}")
    downloaded = await model_manager.download_model(
        repo_id=source["repo_id"],
        filename=source["filename"],
        source=source.get("source", "huggingface"),
        job_id=source.get("job_id"),
    )
    return downloaded


@router.post("/start")
async def start_server(request: ServerStartRequest) -> dict:
    """Start a llama.cpp server, auto-downloading the model if needed."""
    try:
        model_path = await _ensure_model(request.config.model_path, request.source)

        mmproj_path: str | None = None
        if request.config.mmproj_path:
            mmproj_path = await _ensure_model(
                request.config.mmproj_path, request.mmproj_source or request.source
            )

        # The agent owns port allocation: pick a free random port unless the
        # caller pinned one explicitly.
        port = request.config.port or _allocate_port()

        config = ServerConfig(
            model_path=model_path,
            port=port,
            gpu_layers=request.config.gpu_layers,
            context_size=request.config.context_size,
            batch_size=request.config.batch_size,
            cache_prompt=request.config.cache_prompt,
            flash_attn=request.config.flash_attn,
            jinja=request.config.jinja,
            mmproj_path=mmproj_path,
        )

        success = await llama_server_manager.start_server(request.config.id, config)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start server")

        return {
            "status": "started",
            "server_id": request.config.id,
            "model_path": model_path,
            "port": port,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/stop")
async def stop_server(request: ServerStopRequest) -> dict:
    """Stop a llama.cpp server (idempotent: already-stopped is success)."""
    try:
        success = await llama_server_manager.stop_server(request.server_id)
        if not success:
            # Already stopped/unknown — treat as success so callers can
            # stop-then-start without racing the process table.
            return {
                "status": "stopped",
                "server_id": request.server_id,
                "already_stopped": True,
            }

        return {"status": "stopped", "server_id": request.server_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/list")
async def list_servers() -> dict:
    """List all running servers."""
    servers = llama_server_manager.list_servers()
    return {"servers": servers}


@router.get("/status/{server_id}")
async def get_server_status(server_id: str) -> dict:
    """Get status of a specific server."""
    status = await llama_server_manager.get_server_status(server_id)
    return status


@router.get("/logs/{server_id}")
async def get_server_logs(server_id: str, lines: int = 100) -> dict:
    """Get recent logs from a llama.cpp server."""
    return llama_server_manager.get_server_logs(server_id, lines)

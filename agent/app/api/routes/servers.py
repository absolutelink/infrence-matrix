"""Server management API endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.logging import logger
from app.services.llama_server import ServerConfig, llama_server_manager
from app.services.model_manager import model_manager

router = APIRouter(prefix="/servers", tags=["servers"])

server_manager = llama_server_manager


class ServerSpec(BaseModel):
    id: str
    model_path: str
    port: int
    gpu_layers: int = 35
    context_size: int = 4096
    batch_size: int = 512
    cache_prompt: bool = True
    flash_attn: bool = True


class ServerStartRequest(BaseModel):
    config: ServerSpec
    # If provided and the model file is missing locally, the agent downloads
    # the file from the source repo before starting the server.
    source: dict[str, str] | None = None


class ServerStopRequest(BaseModel):
    server_id: str


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

        config = ServerConfig(
            model_path=model_path,
            port=request.config.port,
            gpu_layers=request.config.gpu_layers,
            context_size=request.config.context_size,
            batch_size=request.config.batch_size,
            cache_prompt=request.config.cache_prompt,
            flash_attn=request.config.flash_attn,
        )

        success = await llama_server_manager.start_server(request.config.id, config)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start server")

        return {
            "status": "started",
            "server_id": request.config.id,
            "model_path": model_path,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/stop")
async def stop_server(request: ServerStopRequest) -> dict:
    """Stop a llama.cpp server."""
    try:
        success = await llama_server_manager.stop_server(request.server_id)
        if not success:
            raise HTTPException(status_code=404, detail="Server not found")

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
    servers = llama_server_manager.list_servers()
    for server in servers:
        if server["server_id"] == server_id:
            return {"status": "running", "server": server}

    return {"status": "stopped", "server_id": server_id}


@router.get("/logs/{server_id}")
async def get_server_logs(server_id: str, lines: int = 100) -> dict:
    """Get recent logs from a llama.cpp server."""
    return llama_server_manager.get_server_logs(server_id, lines)

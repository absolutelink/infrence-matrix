"""Server management API endpoints."""

import socket
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import logger
from app.services.event_bus import publish_event
from app.services.halogen_flash_server import (
    HALOGEN_FLASH_CHECKPOINT,
    HALOGEN_FLASH_REPO_ID,
    HALOGEN_FLASH_TOKENIZER,
    HalogenFlashServerConfig,
)
from app.services.halogen_server import (
    HALOGEN_CHECKPOINT,
    HALOGEN_REPO_ID,
    HALOGEN_TOKENIZER,
)
from app.services.llama_server import ServerConfig
from app.services.model_manager import model_manager
from app.services.server_manager import server_manager

router = APIRouter(prefix="/servers", tags=["servers"])


# Range for ephemeral llama-server ports when the caller doesn't pin one
EPHEMERAL_PORT_START = 8090
EPHEMERAL_PORT_END = 8190


class ServerSpec(BaseModel):
    id: str
    model_path: str
    engine: str = "llamacpp"
    engine_options: dict = Field(default_factory=dict)
    # Omit to let the agent allocate a free random port
    port: int | None = None
    gpu_layers: int = 35
    context_size: int = 4096
    batch_size: int = 512
    cache_prompt: bool = True
    # None lets llama.cpp decide; True/False maps to --flash-attn on/off
    flash_attn: bool | None = True
    # MTP Draft N-Max; values greater than zero enable draft MTP flags
    mtp_draft_max: int | None = None
    # Jinja chat templates (required for tool calling); default on
    jinja: bool = True
    # Multimodal projector (vision GGUF); resolved like model_path
    mmproj_path: str | None = None
    draft_model_path: str | None = None
    options: dict = Field(default_factory=dict)


class ServerStartRequest(BaseModel):
    config: ServerSpec
    # If provided and the model file is missing locally, the agent downloads
    # the file from the source repo before starting the server.
    source: dict[str, str] | None = None
    # Same shape as source, for the mmproj projector (downloaded on demand)
    mmproj_source: dict[str, str] | None = None
    # Same shape as source, for the dflash draft model
    draft_source: dict[str, str] | None = None


def _validate_engine_platform(engine: str) -> None:
    """Reject engine requests that do not match this agent image."""
    platform = settings.AGENT_PLATFORM
    if platform in ("halogen", "halogen-flash") and engine != platform:
        raise HTTPException(
            status_code=400,
            detail=f"Agent platform={platform} only supports engine={platform}",
        )
    if platform not in ("halogen", "halogen-flash") and engine != "llamacpp":
        raise HTTPException(
            status_code=400,
            detail=f"Agent platform={platform} only supports engine=llamacpp",
        )


@router.post("/prepare")
async def prepare_server(request: ServerStartRequest) -> dict:
    """Download the model and projector without starting llama-server."""
    try:
        _validate_engine_platform(request.config.engine)
        if request.config.engine == "halogen":
            await model_manager.download_repository(
                HALOGEN_REPO_ID, job_id=f"halogen-{request.config.id}"
            )
            return {
                "status": "prepared",
                "server_id": request.config.id,
                "model_path": HALOGEN_CHECKPOINT,
                "tokenizer_path": HALOGEN_TOKENIZER,
            }
        if request.config.engine == "halogen-flash":
            await model_manager.download_repository(
                HALOGEN_FLASH_REPO_ID, job_id=f"halogen-flash-{request.config.id}"
            )
            return {
                "status": "prepared",
                "server_id": request.config.id,
                "model_path": HALOGEN_FLASH_CHECKPOINT,
                "tokenizer_path": HALOGEN_FLASH_TOKENIZER,
            }
        model_path = await _ensure_model(request.config.model_path, request.source)
        mmproj_path = None
        if request.config.mmproj_path:
            if not request.mmproj_source:
                raise HTTPException(
                    status_code=422,
                    detail="mmproj_path set but mmproj_source missing",
                )
            mmproj_path = await _ensure_model(
                request.config.mmproj_path, request.mmproj_source
            )
        draft_model_path = None
        if request.config.draft_model_path:
            draft_model_path = await _ensure_model(
                request.config.draft_model_path, request.draft_source
            )
        return {
            "status": "prepared",
            "server_id": request.config.id,
            "model_path": model_path,
            "mmproj_path": mmproj_path,
            "draft_model_path": draft_model_path,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


class ServerStopRequest(BaseModel):
    server_id: str


class ServerDeleteRequest(BaseModel):
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

    Returns the absolute path to the model file. Emits download.* events
    so the backend/UI can follow the download phase of a server start.
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

    job_id = source.get("job_id") or f"server-file-{filename}"
    logger.info(f"Model {filename} missing; downloading from {source['repo_id']}")
    publish_event(
        "download.started",
        {
            "job_id": job_id,
            "filename": filename,
            "repo_id": source["repo_id"],
        },
    )
    try:
        downloaded = await model_manager.download_model(
            repo_id=source["repo_id"],
            filename=source["filename"],
            source=source.get("source", "huggingface"),
            job_id=job_id,
        )
    except Exception as e:
        publish_event(
            "download.failed",
            {
                "job_id": job_id,
                "filename": filename,
                "error": str(e),
            },
        )
        raise
    return downloaded


@router.post("/start")
async def start_server(request: ServerStartRequest) -> dict:
    """Start a llama.cpp server, auto-downloading the model if needed.

    Downloads (main model + optional projector) happen before the
    llama-server spawn; download.started/progress/completed events keep
    the backend's instance row in "starting" for the whole phase.
    """
    try:
        _validate_engine_platform(request.config.engine)
        # Duplicate start for a running instance is a no-op success (the
        # 60s registration cycle re-dispatches starts the agent already
        # fulfilled — spawning a second process would be wrong).
        if request.config.id in server_manager.servers:
            existing = server_manager.configs[request.config.id]
            return {
                "status": "started",
                "server_id": request.config.id,
                "model_path": existing.model_path,
                "port": existing.port,
                "already_running": True,
            }

        if request.config.engine == "halogen":
            await model_manager.download_repository(
                HALOGEN_REPO_ID, job_id=f"halogen-{request.config.id}"
            )
            model_path = HALOGEN_CHECKPOINT
        elif request.config.engine == "halogen-flash":
            await model_manager.download_repository(
                HALOGEN_FLASH_REPO_ID, job_id=f"halogen-flash-{request.config.id}"
            )
            model_path = HALOGEN_FLASH_CHECKPOINT
        else:
            model_path = await _ensure_model(request.config.model_path, request.source)

        mmproj_path: str | None = None
        if request.config.mmproj_path:
            # The projector's source must describe the projector file itself.
            # A missing mmproj_source means the backend payload is stale —
            # resolving via the main model's source would return the main
            # GGUF as "projector" (llama-server fails to load it as CLIP).
            if not request.mmproj_source:
                raise HTTPException(
                    status_code=422,
                    detail="mmproj_path set but mmproj_source missing — "
                    "cannot resolve the projector file safely",
                )
            mmproj_path = await _ensure_model(
                request.config.mmproj_path, request.mmproj_source
            )

        draft_model_path: str | None = None
        if request.config.draft_model_path:
            draft_model_path = await _ensure_model(
                request.config.draft_model_path, request.draft_source
            )

        # The agent owns port allocation: pick a free random port unless the
        # caller pinned one explicitly.
        port = request.config.port or _allocate_port()

        if request.config.engine == "halogen":
            from app.services.halogen_server import HalogenServerConfig

            config = HalogenServerConfig(
                model_path=HALOGEN_CHECKPOINT,
                port=port,
                api_port=port,
                engine_port=_allocate_port(),
                options=request.config.engine_options,
            )
        elif request.config.engine == "halogen-flash":
            config = HalogenFlashServerConfig(
                model_path=HALOGEN_FLASH_CHECKPOINT,
                port=port,
                api_port=port,
                engine_port=_allocate_port(),
                options=request.config.engine_options,
            )
        else:
            config = ServerConfig(
                model_path=model_path,
                port=port,
                gpu_layers=request.config.gpu_layers,
                context_size=request.config.context_size,
                batch_size=request.config.batch_size,
                cache_prompt=request.config.cache_prompt,
                flash_attn=request.config.flash_attn,
                mtp_draft_max=request.config.mtp_draft_max,
                jinja=request.config.jinja,
                mmproj_path=mmproj_path,
                draft_model_path=draft_model_path,
                options=request.config.options,
            )

        success = await server_manager.start_server(request.config.id, config)
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
        success = await server_manager.stop_server(request.server_id)
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
    servers = server_manager.list_servers()
    return {"servers": servers}


@router.get("/status/{server_id}")
async def get_server_status(server_id: str) -> dict:
    """Get status of a specific server."""
    status = await server_manager.get_server_status(server_id)
    return status


@router.get("/logs/{server_id}")
async def get_server_logs(server_id: str, lines: int = 100) -> dict:
    """Get recent logs from a llama.cpp server."""
    return server_manager.get_server_logs(server_id, lines)


@router.post("/delete")
async def delete_server(request: ServerDeleteRequest) -> dict:
    """Remove server state and its per-server cache directory."""
    await server_manager.stop_server(request.server_id)
    if hasattr(server_manager, "delete_server"):
        server_manager.delete_server(request.server_id)
    return {"status": "deleted", "server_id": request.server_id}

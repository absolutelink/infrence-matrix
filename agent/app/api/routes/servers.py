"""Server management API endpoints."""

import socket
from pathlib import Path
from typing import Any

import httpx
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
    slot_generation: int = Field(default=0, ge=0)


class ServerStartRequest(BaseModel):
    config: ServerSpec
    # If provided and the model file is missing locally, the agent downloads
    # the file from the source repo before starting the server. May carry a
    # "filenames" list of split GGUF parts alongside the primary "filename".
    source: dict[str, Any] | None = None
    # Same shape as source, for the mmproj projector (downloaded on demand)
    mmproj_source: dict[str, Any] | None = None
    # Same shape as source, for the dflash draft model
    draft_source: dict[str, Any] | None = None
    # Gufo engine_options aux files (mmproj / dflash / dspark / mtp), each
    # resolved to a download source plus the exact "path" the binary uses.
    aux_sources: list[dict[str, Any]] | None = None


STRICT_PLATFORMS = ("halogen", "halogen-flash", "gufo")


def _validate_engine_platform(engine: str) -> None:
    """Reject engine requests that do not match this agent image."""
    platform = settings.AGENT_PLATFORM
    if platform in STRICT_PLATFORMS and engine != platform:
        raise HTTPException(
            status_code=400,
            detail=f"Agent platform={platform} only supports engine={platform}",
        )
    if platform not in STRICT_PLATFORMS and engine != "llamacpp":
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
                HALOGEN_REPO_ID,
                job_id=f"halogen-{request.config.id}",
                server_id=request.config.id,
            )
            return {
                "status": "prepared",
                "server_id": request.config.id,
                "model_path": HALOGEN_CHECKPOINT,
                "tokenizer_path": HALOGEN_TOKENIZER,
            }
        if request.config.engine == "halogen-flash":
            await model_manager.download_repository(
                HALOGEN_FLASH_REPO_ID,
                job_id=f"halogen-flash-{request.config.id}",
                server_id=request.config.id,
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
        for aux in request.aux_sources or []:
            if aux.get("path"):
                resolved = await _ensure_model(aux["path"], aux)
                # The file may live in a repo SUBFOLDER (e.g. the Qwen3.8-Flash-Next
                # MTP sidecar under ``mtp/``), so its on-disk path differs from the
                # library ``Model.path`` recorded in engine_options. Rewrite the
                # option to the actual resolved path so the binary reads the file
                # where it was downloaded, not the repo-root path.
                key = aux.get("key")
                if key:
                    request.config.engine_options[key] = resolved
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
    slot_generation: int | None = Field(default=None, ge=0)


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
    source: dict[str, Any] | None,
) -> str:
    """Ensure the model file (and every split part) exists locally.

    Returns the absolute path to the primary file (the ``-00001-of-`` part
    for split GGUFs, which is what llama-server is pointed at). Emits
    download.* events so the backend/UI can follow the download phase of a
    server start.
    """
    from app.services.model_manager import pick_primary_filename

    path = _resolve_model_path(model_path)

    filenames = (source or {}).get("filenames") or (
        [source["filename"]] if source and source.get("filename") else []
    )
    primary = pick_primary_filename(filenames) if filenames else path.rsplit("/", 1)[-1]

    def exists(fn: str) -> bool:
        repo = (source or {}).get("repo_id")
        candidates: list[Path] = []
        if repo:
            candidates.append(model_manager.storage_path(repo, fn))
        candidates.append(model_manager.models_path / fn)
        if fn == primary:
            # Preserve single-file resolution: repo-style absolute path and the
            # original model_path candidates (flat-under-MODELS_PATH etc.).
            candidates.extend(_model_candidates(model_path, fn))
        return any(c.exists() for c in candidates)

    # If every part is already on disk, return the primary's resolved path.
    if filenames and all(exists(fn) for fn in filenames):
        repo = (source or {}).get("repo_id")
        if repo and model_manager.storage_path(repo, primary).exists():
            return str(model_manager.storage_path(repo, primary))
        if (model_manager.models_path / primary).exists():
            return str(model_manager.models_path / primary)
        for candidate in _model_candidates(model_path, primary):
            if candidate.exists():
                return str(candidate)
        return path

    # Need to download: require a source repo.
    if not source or not source.get("repo_id"):
        raise HTTPException(
            status_code=404,
            detail=f"Model file {primary} not found on agent and no download source provided",
        )

    job_id = source.get("job_id") or f"server-file-{primary}"
    server_id = source.get("server_id")
    logger.info(
        f"Model {primary} (or a split part) missing; downloading from {source['repo_id']}"
    )
    # download_model_parts drives per-file download.started/progress/completed
    # events (via download_model); emit a top-level failed event on error to
    # preserve the existing event contract for the primary file.
    try:
        downloaded = await model_manager.download_model_parts(
            repo_id=source["repo_id"],
            filenames=filenames or [primary],
            source=source.get("source", "huggingface"),
            job_id=job_id,
            server_id=server_id,
        )
    except Exception as e:
        publish_event(
            "download.failed",
            {
                "job_id": job_id,
                "filename": primary,
                "server_id": server_id or "",
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
        latest_generation = server_manager.slot_generations.get(request.config.id, -1)
        generation_is_fenced = (
            latest_generation > 0 or request.config.slot_generation > 0
        )
        if (
            request.config.id not in server_manager.servers
            and generation_is_fenced
            and request.config.slot_generation <= latest_generation
        ):
            raise HTTPException(status_code=409, detail="Stale slot generation")
        # Duplicate start for a running instance is a no-op success (the
        # 60s registration cycle re-dispatches starts the agent already
        # fulfilled — spawning a second process would be wrong).
        if request.config.id in server_manager.servers:
            existing = server_manager.configs[request.config.id]
            if request.config.slot_generation < existing.slot_generation:
                raise HTTPException(status_code=409, detail="Stale slot generation")
            if request.config.slot_generation > existing.slot_generation:
                await server_manager.stop_server(
                    request.config.id,
                    slot_generation=existing.slot_generation,
                )
            else:
                return {
                    "status": "started",
                    "server_id": request.config.id,
                    "model_path": existing.model_path,
                    "port": existing.port,
                    "slot_generation": existing.slot_generation,
                    "effective_capacity": server_manager.get_effective_capacity(
                        request.config.id
                    ),
                    "already_running": True,
                }

        if request.config.engine == "halogen":
            await model_manager.download_repository(
                HALOGEN_REPO_ID,
                job_id=f"halogen-{request.config.id}",
                server_id=request.config.id,
            )
            model_path = HALOGEN_CHECKPOINT
        elif request.config.engine == "halogen-flash":
            await model_manager.download_repository(
                HALOGEN_FLASH_REPO_ID,
                job_id=f"halogen-flash-{request.config.id}",
                server_id=request.config.id,
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

        # Gufo aux files (mmproj / dflash / dspark / mtp) are referenced by
        # engine_options path and must exist at that exact path before spawn.
        for aux in request.aux_sources or []:
            if aux.get("path"):
                resolved = await _ensure_model(aux["path"], aux)
                # The file may live in a repo SUBFOLDER (e.g. the Qwen3.8-Flash-Next
                # MTP sidecar under ``mtp/``), so its on-disk path differs from the
                # library ``Model.path`` recorded in engine_options. Rewrite the
                # option to the actual resolved path so the binary reads the file
                # where it was downloaded, not the repo-root path. This must happen
                # before GufoServerConfig is built from engine_options below.
                key = aux.get("key")
                if key:
                    request.config.engine_options[key] = resolved

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
                slot_generation=request.config.slot_generation,
            )
        elif request.config.engine == "halogen-flash":
            config = HalogenFlashServerConfig(
                model_path=HALOGEN_FLASH_CHECKPOINT,
                port=port,
                api_port=port,
                engine_port=_allocate_port(),
                options=request.config.engine_options,
                slot_generation=request.config.slot_generation,
            )
        elif request.config.engine == "gufo":
            from app.services.gufo_server import GufoServerConfig

            config = GufoServerConfig(
                model_path=model_path,
                port=port,
                options=request.config.engine_options,
                slot_generation=request.config.slot_generation,
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
                slot_generation=request.config.slot_generation,
            )

        success = await server_manager.start_server(request.config.id, config)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start server")

        return {
            "status": "started",
            "server_id": request.config.id,
            "model_path": model_path,
            "port": port,
            "slot_generation": request.config.slot_generation,
            "effective_capacity": server_manager.get_effective_capacity(
                request.config.id
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/stop")
async def stop_server(request: ServerStopRequest) -> dict:
    """Stop a llama.cpp server (idempotent: already-stopped is success)."""
    try:
        config = server_manager.configs.get(request.server_id)
        if (
            config is not None
            and request.slot_generation is not None
            and request.slot_generation != config.slot_generation
        ):
            raise HTTPException(status_code=409, detail="Stale slot generation")
        success = await server_manager.stop_server(
            request.server_id, slot_generation=request.slot_generation
        )
        if not success:
            # Already stopped/unknown — treat as success so callers can
            # stop-then-start without racing the process table.
            return {
                "status": "stopped",
                "server_id": request.server_id,
                "already_stopped": True,
            }

        return {"status": "stopped", "server_id": request.server_id}
    except HTTPException:
        raise
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


@router.get("/metadata/{server_id}")
async def get_server_metadata(server_id: str) -> dict:
    """Return the OpenAI model list from a healthy managed server."""
    config = server_manager.configs.get(server_id)
    if config is None or server_id not in server_manager.servers:
        raise HTTPException(status_code=404, detail="Server is not running")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"http://127.0.0.1:{config.port}/v1/models")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise HTTPException(status_code=502, detail="Invalid /v1/models response")
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get("/logs/{server_id}")
async def get_server_logs(
    server_id: str, lines: int = 100, after: int | None = None
) -> dict:
    """Get recent logs from a managed server.

    ``after`` is a cursor (from ``next_cursor`` or a log.lines ``seq_end``);
    when provided, only newer lines are returned.
    """
    return server_manager.get_server_logs(server_id, lines=lines, after=after)


@router.post("/delete")
async def delete_server(request: ServerDeleteRequest) -> dict:
    """Remove server state and its per-server cache directory."""
    await server_manager.stop_server(request.server_id)
    if hasattr(server_manager, "delete_server"):
        server_manager.delete_server(request.server_id)
    return {"status": "deleted", "server_id": request.server_id}

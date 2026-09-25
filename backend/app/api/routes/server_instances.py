"""Server instances management endpoints."""

import asyncio
import logging
import uuid as uuid_module
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from app.db.session import AsyncSessionMaker
from app.models import Agent, Model, ServerInstance
from app.services.agent_manager import agent_manager
from app.services.benchmark import is_benchmark_blocking
from app.services.server_options import (
    ServerOptions,
    validate_halogen_options,
    validate_server_options,
)
from app.services.server_startup import (
    build_start_payload as _build_start_payload,
)
from app.services.server_startup import dispatch_prepare as _dispatch_prepare
from app.services.server_startup import (
    dispatch_start as _dispatch_start,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/server-instances", tags=["server-instances"])

HALOGEN_MODEL_NAME = "halogen-qwen3.8-27b"
HALOGEN_MODEL_PATH = "/models/peonist-ai/halogen-qwen3.8-27b/qwen3.8-27b-p1w4d-d2.hgn"


async def _resolve_mmproj_model(
    session: AsyncSession, mmproj_model_id: str | None
) -> Model | None:
    """Fetch and validate the selected mmproj model (None = no projector)."""
    if not mmproj_model_id:
        return None
    found = await session.get(Model, uuid_module.UUID(mmproj_model_id))
    if not found:
        raise HTTPException(status_code=404, detail="mmproj model not found")
    if found.model_type != "mmproj":
        raise HTTPException(
            status_code=400, detail="Selected model is not an mmproj projector"
        )
    return found


async def _resolve_dflash_model(
    session: AsyncSession, dflash_model_id: str | None
) -> Model | None:
    """Fetch and validate the selected dflash draft model."""
    if not dflash_model_id:
        return None
    found = await session.get(Model, uuid_module.UUID(dflash_model_id))
    if not found:
        raise HTTPException(status_code=404, detail="dflash model not found")
    if found.model_type != "dflash":
        raise HTTPException(status_code=400, detail="Selected model is not dflash")
    return found


class ServerInstanceResponse(BaseModel):
    id: str
    model_id: str
    model_name: str | None = None
    mmproj_model_id: str | None = None
    mmproj_model_name: str | None = None
    dflash_model_id: str | None = None
    dflash_model_name: str | None = None
    alias: str
    engine: Literal["llamacpp", "halogen"] = "llamacpp"
    engine_options: dict = Field(default_factory=dict)
    status: str
    health_status: str
    error_message: str | None = None
    agent_id: str
    agent_name: str | None = None
    agent_host: str | None = None
    agent_port: int | None = None
    proxy_url: str | None = None
    started_at: str | None = None
    last_health_check: str | None = None
    total_requests: int
    cpu_usage_percent: float | None = None
    ram_usage_bytes: int | None = None
    vram_usage_bytes: int | None = None
    gpu_layers: int = 35
    context_size: int = 4096
    flash_attn: bool = True
    mtp_draft_max: int | None = None
    inactivity_timeout_seconds: int = 300
    server_options: dict = Field(default_factory=dict)


class ServerInstanceListResponse(BaseModel):
    server_instances: list[ServerInstanceResponse]


class StartServerRequest(BaseModel):
    model_id: str | None = None
    agent_id: str | None = None
    gpu_layers: int = 35
    context_size: int = 4096
    # Optional multimodal projector; omitted from server flags when None
    mmproj_model_id: str | None = None
    dflash_model_id: str | None = None
    # MTP Draft N-Max value; when 0 no MTP flags are added, when > 0 add --spec-type draft-mtp --spec-draft-n-max <value>
    mtp_draft_max: int | None = None
    server_options: ServerOptions = Field(default_factory=ServerOptions)
    # Public name clients use in OpenAI-compatible requests
    alias: str = Field(..., min_length=1, max_length=255)
    engine: Literal["llamacpp", "halogen"] = "llamacpp"
    engine_options: dict = Field(default_factory=dict)


class UpdateServerRequest(BaseModel):
    """Editable server settings. All fields optional."""

    alias: str | None = Field(None, min_length=1, max_length=255)
    engine: Literal["llamacpp", "halogen"] | None = None
    engine_options: dict | None = None
    # Swap the model this server serves (requires stop + restart when running)
    model_id: str | None = None
    gpu_layers: int | None = Field(None, ge=0, le=1000)
    context_size: int | None = Field(None, ge=256, le=1_048_576)
    flash_attn: bool | None = None
    mtp_draft_max: int | None = None
    inactivity_timeout_seconds: int | None = Field(None, ge=0, le=86400)
    # Multimodal projector; None = no change, "" = clear selection
    mmproj_model_id: str | None = None
    # Dflash draft model; None = no change, "" = clear selection
    dflash_model_id: str | None = None
    server_options: ServerOptions | None = None
    restart: bool = True  # restart immediately if the server is running


@router.get("", response_model=ServerInstanceListResponse)
async def list_server_instances() -> ServerInstanceListResponse:
    """List all server instances."""
    async with AsyncSessionMaker() as session:
        result = await session.execute(
            select(ServerInstance)
            .join(Model, col(ServerInstance.model_id) == col(Model.id))
            .join(Agent, col(ServerInstance.agent_id) == col(Agent.id))
        )
        instances = result.scalars().all()

        server_list = []
        for instance in instances:
            server_list.append(
                ServerInstanceResponse(
                    id=str(instance.id),
                    model_id=str(instance.model_id),
                    model_name=instance.model.name if instance.model else None,
                    mmproj_model_id=str(instance.mmproj_model_id)
                    if instance.mmproj_model_id
                    else None,
                    mmproj_model_name=instance.mmproj_model.name
                    if instance.mmproj_model
                    else None,
                    dflash_model_id=str(instance.dflash_model_id)
                    if instance.dflash_model_id
                    else None,
                    dflash_model_name=instance.dflash_model.name
                    if instance.dflash_model
                    else None,
                    alias=instance.alias,
                    engine=instance.engine,
                    engine_options=instance.engine_options or {},
                    status=instance.status,
                    health_status=instance.health_status,
                    error_message=instance.error_message,
                    agent_id=str(instance.agent_id),
                    agent_name=instance.agent.name if instance.agent else None,
                    agent_host=instance.agent.host if instance.agent else None,
                    agent_port=instance.agent.port if instance.agent else None,
                    proxy_url=instance.proxy_url,
                    started_at=instance.started_at.isoformat()
                    if instance.started_at
                    else None,
                    last_health_check=instance.last_health_check.isoformat()
                    if instance.last_health_check
                    else None,
                    total_requests=instance.total_requests,
                    cpu_usage_percent=instance.cpu_usage_percent,
                    ram_usage_bytes=instance.ram_usage_bytes,
                    vram_usage_bytes=instance.vram_usage_bytes,
                    gpu_layers=instance.gpu_layers,
                    context_size=instance.context_size,
                    flash_attn=instance.flash_attn,
                    mtp_draft_max=instance.mtp_draft_max,
                    inactivity_timeout_seconds=instance.inactivity_timeout_seconds,
                    server_options=instance.server_options or {},
                )
            )

        return ServerInstanceListResponse(server_instances=server_list)


@router.get("/{server_id}")
async def get_server_instance(server_id: str) -> ServerInstanceResponse:
    """Get a specific server instance."""
    async with AsyncSessionMaker() as session:
        result = await session.execute(
            select(ServerInstance)
            .where(col(ServerInstance.id) == server_id)
            .join(Model, col(ServerInstance.model_id) == col(Model.id))
            .join(Agent, col(ServerInstance.agent_id) == col(Agent.id))
        )
        instance = result.scalar_one_or_none()

        if not instance:
            raise HTTPException(status_code=404, detail="Server instance not found")

        return ServerInstanceResponse(
            id=str(instance.id),
            model_id=str(instance.model_id),
            model_name=instance.model.name if instance.model else None,
            mmproj_model_id=str(instance.mmproj_model_id)
            if instance.mmproj_model_id
            else None,
            mmproj_model_name=instance.mmproj_model.name
            if instance.mmproj_model
            else None,
            dflash_model_id=str(instance.dflash_model_id)
            if instance.dflash_model_id
            else None,
            dflash_model_name=instance.dflash_model.name
            if instance.dflash_model
            else None,
            alias=instance.alias,
            engine=instance.engine,
            engine_options=instance.engine_options or {},
            status=instance.status,
            health_status=instance.health_status,
            error_message=instance.error_message,
            agent_id=str(instance.agent_id),
            agent_name=instance.agent.name if instance.agent else None,
            agent_host=instance.agent.host if instance.agent else None,
            agent_port=instance.agent.port if instance.agent else None,
            proxy_url=instance.proxy_url,
            started_at=instance.started_at.isoformat() if instance.started_at else None,
            last_health_check=(
                instance.last_health_check.isoformat()
                if instance.last_health_check
                else None
            ),
            total_requests=instance.total_requests,
            cpu_usage_percent=instance.cpu_usage_percent,
            ram_usage_bytes=instance.ram_usage_bytes,
            vram_usage_bytes=instance.vram_usage_bytes,
            gpu_layers=instance.gpu_layers,
            context_size=instance.context_size,
            flash_attn=instance.flash_attn,
            mtp_draft_max=instance.mtp_draft_max,
            inactivity_timeout_seconds=instance.inactivity_timeout_seconds,
            server_options=instance.server_options or {},
        )


@router.post("/start")
async def start_server(request: StartServerRequest) -> dict[str, Any]:
    """Start a new server instance."""
    if await is_benchmark_blocking():
        raise HTTPException(
            status_code=409,
            detail="Servers cannot be started while a benchmark is running",
        )
    async with AsyncSessionMaker() as session:
        if request.engine == "halogen":
            if request.model_id is not None:
                raise HTTPException(
                    status_code=400,
                    detail="Halogen model selection is managed by the agent",
                )
            model_result = await session.execute(
                select(Model).where(col(Model.name) == HALOGEN_MODEL_NAME)
            )
            model = model_result.scalar_one_or_none()
            if model is None:
                model = Model(
                    name=HALOGEN_MODEL_NAME,
                    path=HALOGEN_MODEL_PATH,
                    size_bytes=0,
                    architecture="qwen3.8",
                    model_type="llm",
                    quantization="hgn",
                    source="huggingface",
                    source_repo_id="peonist-ai/halogen-qwen3.8-27b",
                    source_file="qwen3.8-27b-p1w4d-d2.hgn",
                    description="Agent-managed Halogen checkpoint",
                )
                session.add(model)
                await session.flush()
        elif request.model_id is None:
            raise HTTPException(status_code=422, detail="model_id is required")
        else:
            model = await session.get(Model, uuid_module.UUID(request.model_id))
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        agent = None
        if request.agent_id:
            agent = await session.get(Agent, uuid_module.UUID(request.agent_id))
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
        else:
            agent_query = select(Agent).where(col(Agent.status) == "online")
            if request.engine == "halogen":
                agent_query = agent_query.where(col(Agent.platform) == "halogen")
            result = await session.execute(agent_query.limit(1))
            agent = result.scalar_one_or_none()

        if not agent:
            raise HTTPException(
                status_code=503, detail="No online agent available to start server"
            )

        if request.engine == "halogen":
            if agent.platform != "halogen" or agent.type != "rocm":
                raise HTTPException(
                    status_code=400,
                    detail="Halogen servers require an agent with platform=halogen and type=rocm",
                )
            if request.server_options.model_dump(exclude_none=True):
                raise HTTPException(
                    status_code=400,
                    detail="llama.cpp server_options are not valid for Halogen",
                )
            engine_options = validate_halogen_options(request.engine_options)
        else:
            # Preserve compatibility with existing custom llama.cpp platform
            # labels; only Halogen is a distinct engine contract.
            if agent.platform == "halogen":
                raise HTTPException(
                    status_code=400,
                    detail="Halogen agents require engine=halogen",
                )
            engine_options = {}

        # Reject duplicate aliases up front for a clean 4xx (the DB unique
        # constraint stays as the race backstop).
        existing_alias = await session.execute(
            select(col(ServerInstance.id)).where(
                col(ServerInstance.alias) == request.alias
            )
        )
        if existing_alias.scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=409, detail=f"Alias '{request.alias}' already exists"
            )

        # The agent allocates a random free port per start; the port is not
        # persisted. Create the instance row up front so the UI can track it.
        if request.engine == "halogen" and (
            request.mmproj_model_id or request.dflash_model_id
        ):
            raise HTTPException(
                status_code=400,
                detail="Halogen servers do not support mmproj or dflash models",
            )
        mmproj_model = await _resolve_mmproj_model(session, request.mmproj_model_id)
        dflash_model = await _resolve_dflash_model(session, request.dflash_model_id)
        options = validate_server_options(request.server_options)
        server = ServerInstance(
            model_id=model.id,
            agent_id=agent.id,
            alias=request.alias,
            process_command=f"{model.source_file or model.path}",
            engine=request.engine,
            engine_options=engine_options,
            gpu_layers=request.gpu_layers,
            context_size=request.context_size,
            flash_attn=True,
            mtp_draft_max=request.mtp_draft_max,
            mmproj_model_id=mmproj_model.id if mmproj_model else None,
            dflash_model_id=dflash_model.id if dflash_model else None,
            server_options=options,
            status="stopped",
            health_status="unknown",
            inactivity_timeout_seconds=300,
        )
        session.add(server)
        await session.commit()
        await session.refresh(server)

        server_id = str(server.id)

        payload = _build_start_payload(server, model)

        # Dispatch to the agent in the background: the download can take a
        # long time and the HTTP request from the UI must not block on it.
        agent_id = str(agent.id)
        asyncio.create_task(_dispatch_prepare(agent_id, server_id, payload))

        return {
            "status": "stopped",
            "server_id": server_id,
            "model_id": str(model.id),
            "agent_id": agent_id,
            "message": "Server created; model preparation requested",
        }


@router.put("/{server_id}")
async def update_server(server_id: str, request: UpdateServerRequest) -> dict[str, Any]:
    """Update server settings; restart the server if it is running."""
    if await is_benchmark_blocking():
        raise HTTPException(
            status_code=409,
            detail="Server changes are disabled while a benchmark is running",
        )
    async with AsyncSessionMaker() as session:
        instance = await session.get(ServerInstance, uuid_module.UUID(server_id))
        if not instance:
            raise HTTPException(status_code=404, detail="Server instance not found")

        if instance.status == "starting":
            raise HTTPException(
                status_code=409,
                detail="Server is starting; wait for it to finish before editing",
            )

        was_running = instance.status == "running"
        model_changed = False
        mmproj_changed = False
        dflash_changed = False

        if request.alias is not None and request.alias != instance.alias:
            existing_alias = await session.execute(
                select(col(ServerInstance.id)).where(
                    col(ServerInstance.alias) == request.alias,
                    col(ServerInstance.id) != instance.id,
                )
            )
            if existing_alias.scalar_one_or_none() is not None:
                raise HTTPException(
                    status_code=409, detail=f"Alias '{request.alias}' already exists"
                )
            instance.alias = request.alias

        if request.engine is not None and request.engine != instance.engine:
            raise HTTPException(
                status_code=400,
                detail="Changing a server engine is not supported; create a new server",
            )
        if request.engine_options is not None:
            instance.engine_options = (
                validate_halogen_options(request.engine_options)
                if instance.engine == "halogen"
                else {}
            )

        if request.model_id is not None and request.model_id != str(instance.model_id):
            new_model = await session.get(Model, uuid_module.UUID(request.model_id))
            if not new_model:
                raise HTTPException(status_code=404, detail="Model not found")
            instance.model_id = new_model.id
            instance.process_command = new_model.source_file or new_model.path
            model_changed = True

        # "" clears the projector selection; None leaves it unchanged
        if request.mmproj_model_id is not None:
            mmproj_model = await _resolve_mmproj_model(
                session, request.mmproj_model_id or None
            )
            new_mmproj_id = mmproj_model.id if mmproj_model else None
            if new_mmproj_id != instance.mmproj_model_id:
                instance.mmproj_model_id = new_mmproj_id
                mmproj_changed = True

        if request.dflash_model_id is not None:
            dflash_model = await _resolve_dflash_model(
                session, request.dflash_model_id or None
            )
            new_dflash_id = dflash_model.id if dflash_model else None
            if new_dflash_id != instance.dflash_model_id:
                instance.dflash_model_id = new_dflash_id
                dflash_changed = True

        if request.gpu_layers is not None:
            instance.gpu_layers = request.gpu_layers
        if request.context_size is not None:
            instance.context_size = request.context_size
        if request.flash_attn is not None:
            instance.flash_attn = request.flash_attn
        if request.mtp_draft_max is not None:
            instance.mtp_draft_max = request.mtp_draft_max
        if request.server_options is not None:
            instance.server_options = validate_server_options(request.server_options)
        if request.inactivity_timeout_seconds is not None:
            instance.inactivity_timeout_seconds = request.inactivity_timeout_seconds

        # Keep the legacy JSON config in sync for old readers
        instance.config = {
            **(instance.config or {}),
            "gpu_layers": str(instance.gpu_layers),
            "context_size": str(instance.context_size),
            "flash_attn": str(instance.flash_attn).lower(),
            "mtp_draft_max": str(instance.mtp_draft_max)
            if instance.mtp_draft_max is not None
            else None,
        }
        await session.commit()

        needs_start = model_changed or mmproj_changed or dflash_changed

        # A running server must restart to load a different model file
        if was_running and (request.restart or needs_start):
            await session.refresh(instance)
            agent = await session.get(Agent, instance.agent_id)
            if not agent or agent.status != "online":
                raise HTTPException(
                    status_code=503,
                    detail="Settings saved, but agent is offline — restart manually",
                )
            model = await session.get(Model, instance.model_id)
            if not model:
                raise HTTPException(
                    status_code=503,
                    detail="Settings saved, but model no longer exists",
                )

            # Stop the running llama-server on the agent
            try:
                await agent_manager.send_to_agent(
                    str(instance.agent_id),
                    "POST",
                    "/servers/stop",
                    {"server_id": str(instance.id)},
                    timeout=60.0,
                )
            except Exception as e:
                logger.warning(f"Failed to stop server before restart: {e}")

            instance.status = "starting"
            instance.error_message = None
            await session.commit()

            payload = _build_start_payload(instance, model)
            asyncio.create_task(
                _dispatch_start(str(instance.agent_id), str(instance.id), payload)
            )

            return {
                "status": "restarting",
                "server_id": str(instance.id),
                "message": "Settings saved — restarting server",
            }

        # A stopped/errored server whose model or projector changed gets
        # started right away (download included, status stays "starting"
        # until the agent reports healthy).
        if not was_running and needs_start:
            agent = await session.get(Agent, instance.agent_id)
            if agent and agent.status == "online":
                model = await session.get(Model, instance.model_id)
                if model:
                    instance.status = "starting"
                    instance.error_message = None
                    await session.commit()

                    payload = _build_start_payload(instance, model)
                    agent_id = str(agent.id)
                    server_id = str(instance.id)
                    asyncio.create_task(_dispatch_start(agent_id, server_id, payload))

                    return {
                        "status": "starting",
                        "server_id": server_id,
                        "message": "Settings saved — starting server (downloads may take a while)",
                    }

        return {
            "status": instance.status,
            "server_id": str(instance.id),
            "message": "Settings saved",
        }


@router.post("/{server_id}/start")
async def restart_server(server_id: str) -> dict[str, Any]:
    """Start an existing (stopped or errored) server instance."""
    if await is_benchmark_blocking():
        raise HTTPException(
            status_code=409,
            detail="Servers cannot be started while a benchmark is running",
        )
    async with AsyncSessionMaker() as session:
        instance = await session.get(ServerInstance, uuid_module.UUID(server_id))
        if not instance:
            raise HTTPException(status_code=404, detail="Server instance not found")

        if instance.status in ("starting", "running"):
            raise HTTPException(
                status_code=409,
                detail=f"Server instance is already {instance.status}",
            )

        agent = await session.get(Agent, instance.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        model = await session.get(Model, instance.model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        if agent.status != "online":
            raise HTTPException(
                status_code=503, detail=f"Agent {agent.name} is {agent.status}"
            )

        instance.status = "starting"
        instance.error_message = None
        await session.commit()

        payload = _build_start_payload(instance, model)

        # Dispatch in the background; the row is already "starting".
        agent_id = str(agent.id)
        asyncio.create_task(_dispatch_start(agent_id, str(instance.id), payload))

        return {
            "status": "starting",
            "server_id": str(instance.id),
            "agent_id": agent_id,
            "message": "Server start request sent to agent",
        }


@router.post("/{server_id}/stop")
async def stop_server(server_id: str) -> dict[str, str]:
    """Stop a server instance."""
    async with AsyncSessionMaker() as session:
        instance = await session.get(ServerInstance, uuid_module.UUID(server_id))
        if not instance:
            raise HTTPException(status_code=404, detail="Server instance not found")

        instance.status = "stopped"
        instance.health_status = "unknown"
        await session.commit()

        # Forward the stop command to the agent (best effort).
        try:
            await agent_manager.send_to_agent(
                str(instance.agent_id),
                "POST",
                "/servers/stop",
                {"server_id": str(instance.id)},
                timeout=60.0,
            )
        except Exception as e:
            logger.warning(f"Failed to stop server on agent: {e}")

        return {"status": "stopped", "message": "Server stop request sent"}


@router.delete("/{server_id}")
async def delete_server(server_id: str) -> dict[str, str]:
    """Delete a server instance, stopping it on the agent if needed."""
    async with AsyncSessionMaker() as session:
        instance = await session.get(ServerInstance, uuid_module.UUID(server_id))
        if not instance:
            raise HTTPException(status_code=404, detail="Server instance not found")

        was_active = instance.status in ("starting", "running")
        agent_id = str(instance.agent_id)
        instance_id = str(instance.id)

        # Stop the llama-server on the agent first (best effort) so the
        # process does not outlive its row.
        if was_active or instance.engine == "halogen":
            try:
                await agent_manager.send_to_agent(
                    agent_id,
                    "POST",
                    "/servers/delete"
                    if instance.engine == "halogen"
                    else "/servers/stop",
                    {"server_id": instance_id},
                    timeout=60.0,
                )
            except Exception as e:
                logger.warning(f"Failed to stop server on agent before delete: {e}")

        await session.delete(instance)
        await session.commit()

    return {
        "status": "deleted",
        "server_id": instance_id,
        "message": "Server instance deleted"
        + (" (stop request sent to agent)" if was_active else ""),
    }

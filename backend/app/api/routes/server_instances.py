"""Server instances management endpoints."""

import asyncio
import logging
import uuid as uuid_module
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlmodel import col

from app.db.session import AsyncSessionMaker
from app.models import Agent, Model, ServerInstance
from app.services.agent_manager import agent_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/server-instances", tags=["server-instances"])


class ServerInstanceResponse(BaseModel):
    id: str
    model_id: str
    model_name: str | None = None
    alias: str
    status: str
    health_status: str
    agent_id: str
    agent_name: str | None = None
    agent_host: str | None = None
    agent_port: int | None = None
    proxy_url: str | None = None
    started_at: str | None = None
    total_requests: int
    cpu_usage_percent: float | None = None
    ram_usage_bytes: int | None = None
    vram_usage_bytes: int | None = None
    gpu_layers: int = 35
    context_size: int = 4096
    flash_attn: bool = True
    inactivity_timeout_seconds: int = 300


class ServerInstanceListResponse(BaseModel):
    server_instances: list[ServerInstanceResponse]


class StartServerRequest(BaseModel):
    model_id: str
    agent_id: str | None = None
    gpu_layers: int = 35
    context_size: int = 4096
    # Public name clients use in OpenAI-compatible requests
    alias: str = Field(..., min_length=1, max_length=255)


class UpdateServerRequest(BaseModel):
    """Editable server settings. All fields optional."""

    alias: str | None = Field(None, min_length=1, max_length=255)
    gpu_layers: int | None = Field(None, ge=0, le=1000)
    context_size: int | None = Field(None, ge=256, le=1_048_576)
    flash_attn: bool | None = None
    inactivity_timeout_seconds: int | None = Field(None, ge=0, le=86400)
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
                    alias=instance.alias,
                    status=instance.status,
                    health_status=instance.health_status,
                    agent_id=str(instance.agent_id),
                    agent_name=instance.agent.name if instance.agent else None,
                    agent_host=instance.agent.host if instance.agent else None,
                    agent_port=instance.agent.port if instance.agent else None,
                    proxy_url=instance.proxy_url,
                    started_at=instance.started_at.isoformat()
                    if instance.started_at
                    else None,
                    total_requests=instance.total_requests,
                    cpu_usage_percent=instance.cpu_usage_percent,
                    ram_usage_bytes=instance.ram_usage_bytes,
                    vram_usage_bytes=instance.vram_usage_bytes,
                    gpu_layers=instance.gpu_layers,
                    context_size=instance.context_size,
                    flash_attn=instance.flash_attn,
                    inactivity_timeout_seconds=instance.inactivity_timeout_seconds,
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
            alias=instance.alias,
            status=instance.status,
            health_status=instance.health_status,
            agent_id=str(instance.agent_id),
            agent_name=instance.agent.name if instance.agent else None,
            agent_host=instance.agent.host if instance.agent else None,
            agent_port=instance.agent.port if instance.agent else None,
            proxy_url=instance.proxy_url,
            started_at=instance.started_at.isoformat() if instance.started_at else None,
            total_requests=instance.total_requests,
            cpu_usage_percent=instance.cpu_usage_percent,
            ram_usage_bytes=instance.ram_usage_bytes,
            vram_usage_bytes=instance.vram_usage_bytes,
            gpu_layers=instance.gpu_layers,
            context_size=instance.context_size,
            flash_attn=instance.flash_attn,
            inactivity_timeout_seconds=instance.inactivity_timeout_seconds,
        )


@router.post("/start")
async def start_server(request: StartServerRequest) -> dict[str, Any]:
    """Start a new server instance."""
    async with AsyncSessionMaker() as session:
        model = await session.get(Model, uuid_module.UUID(request.model_id))
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        agent = None
        if request.agent_id:
            agent = await session.get(Agent, uuid_module.UUID(request.agent_id))
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
        else:
            result = await session.execute(
                select(Agent).where(col(Agent.status) == "online").limit(1)
            )
            agent = result.scalar_one_or_none()

        if not agent:
            raise HTTPException(
                status_code=503, detail="No online agent available to start server"
            )

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
        server = ServerInstance(
            model_id=model.id,
            agent_id=agent.id,
            alias=request.alias,
            process_command=f"{model.source_file or model.path}",
            gpu_layers=request.gpu_layers,
            context_size=request.context_size,
            flash_attn=True,
            status="starting",
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
        asyncio.create_task(_dispatch_start(agent_id, server_id, payload))

        return {
            "status": "starting",
            "server_id": server_id,
            "model_id": request.model_id,
            "agent_id": agent_id,
            "message": "Server start request sent to agent",
        }


def _build_start_payload(instance: ServerInstance, model: Model) -> dict[str, Any]:
    """Build the agent /servers/start payload from instance + model.

    The agent allocates the port (random free one) when none is sent.
    """
    filename = model.source_file or model.path.rsplit("/", 1)[-1]
    return {
        "config": {
            "id": str(instance.id),
            "model_path": model.path,
            "gpu_layers": instance.gpu_layers,
            "context_size": instance.context_size,
            "batch_size": 512,
            "cache_prompt": True,
            "flash_attn": instance.flash_attn,
        },
        # The agent downloads the model file first if it is missing.
        "source": {
            "source": model.source,
            "repo_id": model.source_repo_id or "",
            "filename": filename,
            "job_id": f"server-{instance.id}",
        }
        if model.source_repo_id
        else None,
    }


async def _dispatch_start(
    agent_id: str, server_id: str, payload: dict[str, Any]
) -> None:
    """Send the start command to the agent, marking failure on the row."""
    try:
        response = await agent_manager.send_to_agent(
            agent_id,
            "POST",
            "/servers/start",
            payload,
            timeout=900.0,
        )
        # The agent returned success (llama-server healthy). Mark running in
        # case the server.started event was lost (e.g. backend restart). The
        # agent-allocated port is echoed back and kept in the JSON config
        # for display only (not a column).
        allocated_port = response.get("port") if isinstance(response, dict) else None
        async with AsyncSessionMaker() as session:
            server = await session.get(ServerInstance, uuid_module.UUID(server_id))
            if server and server.status != "running":
                server.status = "running"
                server.health_status = "healthy"
                server.started_at = server.started_at or datetime.now(UTC)
                if allocated_port:
                    server.config = {
                        **(server.config or {}),
                        "port": str(allocated_port),
                    }
                session.add(server)
                await session.commit()
                logger.info(f"Server {server_id} marked as running (dispatch ack)")
    except Exception as e:
        logger.error(f"Failed to start server {server_id} on agent {agent_id}: {e}")
        async with AsyncSessionMaker() as session:
            server = await session.get(ServerInstance, uuid_module.UUID(server_id))
            if server:
                server.status = "error"
                server.error_message = str(e)
                session.add(server)
                await session.commit()


@router.put("/{server_id}")
async def update_server(server_id: str, request: UpdateServerRequest) -> dict[str, Any]:
    """Update server settings; restart the server if it is running."""
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

        if request.gpu_layers is not None:
            instance.gpu_layers = request.gpu_layers
        if request.context_size is not None:
            instance.context_size = request.context_size
        if request.flash_attn is not None:
            instance.flash_attn = request.flash_attn
        if request.inactivity_timeout_seconds is not None:
            instance.inactivity_timeout_seconds = request.inactivity_timeout_seconds

        # Keep the legacy JSON config in sync for old readers
        instance.config = {
            **(instance.config or {}),
            "gpu_layers": str(instance.gpu_layers),
            "context_size": str(instance.context_size),
            "flash_attn": str(instance.flash_attn).lower(),
        }
        await session.commit()

        if was_running and request.restart:
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

        return {
            "status": instance.status,
            "server_id": str(instance.id),
            "message": "Settings saved",
        }


@router.post("/{server_id}/start")
async def restart_server(server_id: str) -> dict[str, Any]:
    """Start an existing (stopped or errored) server instance."""
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
        if was_active:
            try:
                await agent_manager.send_to_agent(
                    agent_id,
                    "POST",
                    "/servers/stop",
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

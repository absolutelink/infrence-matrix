"""Server instances management endpoints."""

import asyncio
import logging
import socket
import uuid as uuid_module
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlmodel import col

from app.db.session import AsyncSessionMaker
from app.models import Agent, Model, ServerInstance
from app.services.agent_manager import agent_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/server-instances", tags=["server-instances"])

# Port range used for llama.cpp servers on agents
AGENT_SERVER_PORT_START = 8090
AGENT_SERVER_PORT_END = 8190


class ServerInstanceResponse(BaseModel):
    id: str
    model_id: str
    model_name: str | None = None
    port: int
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


class ServerInstanceListResponse(BaseModel):
    server_instances: list[ServerInstanceResponse]


class StartServerRequest(BaseModel):
    model_id: str
    agent_id: str | None = None
    gpu_layers: int = 35
    context_size: int = 4096


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
                    port=instance.port,
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
            port=instance.port,
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

        # Allocate a port on the agent: pick one not used by an active
        # instance of this agent, then double-check nothing is listening.
        result = await session.execute(
            select(col(ServerInstance.port)).where(
                col(ServerInstance.agent_id) == agent.id,
                col(ServerInstance.status).in_(["starting", "running"]),
            )
        )
        used_ports = set(result.scalars().all())

        port = None
        for candidate in range(AGENT_SERVER_PORT_START, AGENT_SERVER_PORT_END):
            if candidate in used_ports:
                continue
            if _port_in_use(agent.host, candidate):
                continue
            port = candidate
            break

        if port is None:
            raise HTTPException(
                status_code=503, detail=f"No free ports on agent {agent.name}"
            )

        # Create the instance row up front so the UI can track it. The agent
        # identifies the llama-server by this id.
        server = ServerInstance(
            model_id=model.id,
            agent_id=agent.id,
            port=port,
            process_command=f"{model.source_file or model.path}",
            config={
                "gpu_layers": str(request.gpu_layers),
                "context_size": str(request.context_size),
            },
            status="starting",
            health_status="unknown",
            inactivity_timeout_seconds=300,
        )
        session.add(server)
        await session.commit()
        await session.refresh(server)

        server_id = str(server.id)
        filename = model.source_file or model.path.rsplit("/", 1)[-1]

        payload = {
            "config": {
                "id": server_id,
                "model_path": model.path,
                "port": port,
                "gpu_layers": request.gpu_layers,
                "context_size": request.context_size,
                "batch_size": 512,
                "cache_prompt": True,
                "flash_attn": True,
            },
            # The agent downloads the model file first if it is missing.
            "source": {
                "source": model.source,
                "repo_id": model.source_repo_id or "",
                "filename": filename,
                "job_id": f"server-{server_id}",
            }
            if model.source_repo_id
            else None,
        }

        # Dispatch to the agent in the background: the download can take a
        # long time and the HTTP request from the UI must not block on it.
        agent_id = str(agent.id)
        asyncio.create_task(_dispatch_start(agent_id, server_id, payload))

        return {
            "status": "starting",
            "server_id": server_id,
            "model_id": request.model_id,
            "agent_id": agent_id,
            "port": port,
            "message": "Server start request sent to agent",
        }


def _port_in_use(host: str, port: int) -> bool:
    """Check whether something already listens on host:port."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


async def _dispatch_start(
    agent_id: str, server_id: str, payload: dict[str, Any]
) -> None:
    """Send the start command to the agent, marking failure on the row."""
    try:
        await agent_manager.send_to_agent(
            agent_id,
            "POST",
            "/servers/start",
            payload,
            timeout=900.0,
        )
    except Exception as e:
        logger.error(f"Failed to start server {server_id} on agent {agent_id}: {e}")
        async with AsyncSessionMaker() as session:
            server = await session.get(ServerInstance, uuid_module.UUID(server_id))
            if server:
                server.status = "error"
                server.error_message = str(e)
                session.add(server)
                await session.commit()


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

        filename = model.source_file or model.path.rsplit("/", 1)[-1]
        config = instance.config or {}

        payload = {
            "config": {
                "id": str(instance.id),
                "model_path": model.path,
                "port": instance.port,
                "gpu_layers": int(config.get("gpu_layers", 35)),
                "context_size": int(config.get("context_size", 4096)),
                "batch_size": 512,
                "cache_prompt": True,
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

        # Dispatch in the background; the row is already "starting".
        agent_id = str(agent.id)
        asyncio.create_task(_dispatch_start(agent_id, str(instance.id), payload))

        return {
            "status": "starting",
            "server_id": str(instance.id),
            "agent_id": agent_id,
            "port": instance.port,
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

"""Server instances management endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.db.session import AsyncSessionMaker
from app.models import Agent, Model, ServerInstance

router = APIRouter(prefix="/server-instances", tags=["server-instances"])


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
            .join(Model, ServerInstance.model_id == Model.id)
            .join(Agent, ServerInstance.agent_id == Agent.id)
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
                    started_at=instance.started_at.isoformat() if instance.started_at else None,
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
            .where(ServerInstance.id == server_id)
            .join(Model, ServerInstance.model_id == Model.id)
            .join(Agent, ServerInstance.agent_id == Agent.id)
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
        model = await session.get(Model, request.model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        agent = None
        if request.agent_id:
            agent = await session.get(Agent, request.agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
        else:
            result = await session.execute(
                select(Agent).where(Agent.status == "online").limit(1)
            )
            agent = result.scalar_one_or_none()

        if not agent:
            raise HTTPException(
                status_code=503,
                detail="No online agent available to start server"
            )

        return {
            "status": "starting",
            "model_id": request.model_id,
            "agent_id": str(agent.id),
            "message": "Server start request sent to agent"
        }


@router.post("/{server_id}/stop")
async def stop_server(server_id: str) -> dict[str, str]:
    """Stop a server instance."""
    async with AsyncSessionMaker() as session:
        instance = await session.get(ServerInstance, server_id)
        if not instance:
            raise HTTPException(status_code=404, detail="Server instance not found")

        instance.status = "stopped"
        await session.commit()

        return {"status": "stopped", "message": "Server stop request sent"}

"""Agent management endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlmodel import update

from app.db.session import AsyncSessionMaker
from app.models import Agent
from app.services.agent_manager import agent_manager

router = APIRouter(prefix="/agents", tags=["agents"])


class AgentRegisterRequest(BaseModel):
    agent_id: str
    name: str
    host: str
    port: int
    gpu_info: dict | None = None


class AgentRegisterResponse(BaseModel):
    registered: bool
    frontend_version: str


class AgentListResponse(BaseModel):
    agents: list[dict]


class AgentUpdateRequest(BaseModel):
    name: str | None = None
    host: str | None = None
    port: int | None = None


@router.post("/register", response_model=AgentRegisterResponse)
async def register_agent(request: AgentRegisterRequest) -> AgentRegisterResponse:
    """Register an agent with the frontend."""
    try:
        await agent_manager.register_agent(request.model_dump())
        return AgentRegisterResponse(
            registered=True,
            frontend_version="0.1.0",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")


@router.get("", response_model=AgentListResponse)
async def list_agents() -> AgentListResponse:
    """List all registered agents."""
    agents = await agent_manager.list_agents()
    return AgentListResponse(agents=agents)


@router.get("/{agent_id}")
async def get_agent(agent_id: str) -> dict:
    """Get agent details."""
    agent = await agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    return {
        "id": str(agent.id),
        "name": agent.name,
        "host": agent.host,
        "port": agent.port,
        "status": agent.status,
        "websocket_connected": agent.websocket_connected,
        "gpu_info": agent.gpu_info,
        "last_seen": agent.last_seen,
    }


@router.put("/{agent_id}")
async def update_agent(agent_id: str, request: AgentUpdateRequest) -> dict:
    """Update agent configuration."""
    async with AsyncSessionMaker() as session:
        agent = await session.get(Agent, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Update fields if provided
        if request.name is not None:
            agent.name = request.name
        if request.host is not None:
            agent.host = request.host
        if request.port is not None:
            agent.port = request.port

        session.add(agent)
        await session.commit()
        await session.refresh(agent)

        # Update agent in manager
        agent_manager.agents[str(agent.id)] = agent

        return {
            "id": str(agent.id),
            "name": agent.name,
            "host": agent.host,
            "port": agent.port,
            "status": agent.status,
        }


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str) -> dict:
    """Delete an agent."""
    async with AsyncSessionMaker() as session:
        agent = await session.get(Agent, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Remove from manager
        agent_manager.agents.pop(str(agent.id), None)
        # Remove from reconnect tasks if exists
        agent_manager._reconnect_tasks.pop(str(agent.id), None)

        await session.delete(agent)
        await session.commit()

        return {"status": "deleted", "message": f"Agent {agent_id} deleted successfully"}


@router.post("/{agent_id}/command")
async def send_command(agent_id: str, command: dict) -> dict:  # noqa: ARG001
    """Send a command to an agent."""
    try:
        # TODO: Implement command sending via WebSocket
        return {"status": "sent", "command": command}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

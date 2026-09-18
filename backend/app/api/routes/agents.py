"""Agent management endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.services.agent_manager import agent_manager
from app.models import Agent

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


@router.post("/register", response_model=AgentRegisterResponse)
async def register_agent(request: AgentRegisterRequest) -> AgentRegisterResponse:
    """Register an agent with the frontend."""
    try:
        agent = await agent_manager.register_agent(request.model_dump())
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


@router.post("/{agent_id}/command")
async def send_command(agent_id: str, command: dict) -> dict:
    """Send a command to an agent."""
    try:
        # TODO: Implement command sending via WebSocket
        return {"status": "sent", "command": command}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

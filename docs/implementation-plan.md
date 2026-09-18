# Inference Matrix Implementation Plan

## Overview

This plan covers implementing the split-service architecture with Frontend Service and Agent Service.

**Estimated Timeline:** 2-3 weeks for MVP

---

## Phase 1: Foundation (Days 1-3)

### 1.1 Database Migrations

**File:** `backend/app/models.py` (update)

Add Agent tracking model:
```python
class Agent(SQLModel, table=True):
    __tablename__ = "agents"
    
    id: uuid.UUID = Field(default_factory=uuid4, primary_key=True)
    name: str = Field(max_length=255, unique=True)
    host: str  # hostname or IP
    port: int  # Agent API port
    
    status: str = "offline"  # online, offline, unreachable
    gpu_info: dict = Field(default_factory=dict, sa_column=Column(JSON))
    
    last_seen: datetime | None = None
    websocket_connected: bool = False
    
    created_at: datetime = Field(default_factory=get_datetime_utc)
    
    # Relationships
    servers: list[ServerInstance] = Relationship(back_populates="agent")
```

**File:** `backend/app/models.py` (update ServerInstance)
```python
class ServerInstance(SQLModel, table=True):
    # ... existing fields ...
    
    agent_id: uuid.UUID = Field(foreign_key="agents.id", ondelete="CASCADE")
    proxy_url: str  # Full URL to proxy through Agent
    
    agent: Agent | None = Relationship(back_populates="servers")
```

**File:** `backend/app/models.py` (update PromptCache)
```python
class PromptCache(SQLModel, table=True):
    # ... existing fields ...
    
    agent_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="agents.id",
        ondelete="SET NULL"
    )
    cache_path: str  # Path on Agent machine
```

**Migration:**
```bash
cd backend
alembic revision --autogenerate -m "Add agent tracking and update server instances"
alembic upgrade head
```

---

### 1.2 Agent Service Structure

**Directory:** `agent/`

Create new service structure:
```
agent/
├── Dockerfile
├── pyproject.toml
├── README.md
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Agent settings
│   │   └── logging.py       # Logging configuration
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── servers.py   # Server management endpoints
│   │   │   ├── models.py    # Model file management
│   │   │   ├── gpu.py       # GPU info endpoints
│   │   │   └── websocket.py # WebSocket event stream
│   │   └── deps.py          # Dependencies
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llama_server.py  # llama.cpp subprocess management
│   │   ├── model_manager.py # Model downloads
│   │   ├── gpu_monitor.py   # GPU monitoring
│   │   ├── proxy.py         # llama.cpp HTTP proxy
│   │   └── frontend_client.py # Frontend registration + WebSocket
│   └── utils/
│       ├── __init__.py
│       └── llama_cpp.py     # llama.cpp helpers
└── tests/
```

**File:** `agent/pyproject.toml`
```toml
[project]
name = "inference-matrix-agent"
version = "0.1.0"
requires-python = ">=3.14,<4.0"
dependencies = [
    "fastapi>=0.141.0,<1.0.0",
    "uvicorn[standard]>=0.30.0,<1.0.0",
    "httpx>=0.27.0,<1.0.0",
    "websockets>=12.0,<13.0",
    "huggingface_hub>=0.20.0,<1.0.0",
    "psutil>=5.9.0,<7.0.0",
    "pydantic-settings>=2.0.0,<3.0.0",
]

[dependency-groups]
dev = [
    "pytest>=7.4.0,<8.0.0",
    "ruff>=0.2.0,<1.0.0",
    "mypy>=1.8.0,<2.0.0",
]
```

**File:** `agent/app/core/config.py`
```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="../.env",
        env_ignore_empty=True,
        extra="ignore",
    )
    
    # Agent identity
    AGENT_ID: str
    AGENT_NAME: str = "inference-agent"
    
    # Frontend connection
    FRONTEND_URL: str
    FRONTEND_API_KEY: str | None = None  # Optional auth
    
    # llama.cpp
    LLAMA_SERVER_PATH: str = "/usr/local/bin/llama-server"
    DEFAULT_GPU_LAYERS: int = 35
    DEFAULT_CONTEXT_SIZE: int = 4096
    DEFAULT_BATCH_SIZE: int = 512
    SERVER_INACTIVITY_TIMEOUT: int = 300
    
    # GPU
    GPU_BACKEND: str = "auto"  # auto, cuda, metal, vulkan
    
    # Storage
    MODELS_PATH: str = "/models"
    CACHE_PATH: str = "/cache"
    
    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_RECONNECT_INTERVAL: int = 5
    WS_MAX_BUFFER_EVENTS: int = 1000

settings = Settings()
```

---

### 1.3 Frontend Service Updates

**File:** `backend/app/core/config.py` (add)
```python
# Agent settings
AGENT_DISCOVERY_ENABLED: bool = True
AGENT_REGISTRATION_TIMEOUT: int = 300  # seconds
AGENT_MAX_RECONNECT_ATTEMPTS: int = 10
```

**File:** `backend/app/services/agent_manager.py` (new)
```python
"""Manages Agent registration, discovery, and WebSocket connections."""

import asyncio
from typing import Dict, Optional
import httpx
from websockets.client import connect, WebSocketClientProtocol

from app.core.config import settings
from app.models import Agent
from app.db.session import session_maker

class AgentManager:
    """Manages Agent lifecycle and WebSocket connections."""
    
    def __init__(self) -> None:
        self.agents: Dict[str, Agent] = {}  # agent_id -> Agent
        self.ws_connections: Dict[str, WebSocketClientProtocol] = {}
        self._reconnect_tasks: Dict[str, asyncio.Task] = {}
    
    async def register_agent(self, agent_data: dict) -> Agent:
        """Register an Agent with the Frontend."""
        async with session_maker() as session:
            agent = Agent(
                id=agent_data["agent_id"],
                name=agent_data["name"],
                host=agent_data["host"],
                port=agent_data["port"],
                gpu_info=agent_data.get("gpu_info", {}),
                status="online",
            )
            session.add(agent)
            await session.commit()
            await session.refresh(agent)
            self.agents[str(agent.id)] = agent
            return agent
    
    async def connect_websocket(self, agent: Agent) -> None:
        """Establish WebSocket connection to Agent."""
        ws_url = f"ws://{agent.host}:{agent.port}/api/ws/status"
        
        try:
            async with connect(ws_url) as websocket:
                self.ws_connections[str(agent.id)] = websocket
                agent.websocket_connected = True
                
                # Listen for events
                async for message in websocket:
                    await self._handle_agent_event(agent.id, message)
                    
        except Exception as e:
            agent.websocket_connected = False
            self._schedule_reconnect(agent)
    
    def _schedule_reconnect(self, agent: Agent) -> None:
        """Schedule WebSocket reconnection."""
        if str(agent.id) in self._reconnect_tasks:
            return
        
        task = asyncio.create_task(self._reconnect_loop(agent))
        self._reconnect_tasks[str(agent.id)] = task
    
    async def _reconnect_loop(self, agent: Agent) -> None:
        """Attempt to reconnect to Agent."""
        attempts = 0
        while attempts < settings.AGENT_MAX_RECONNECT_ATTEMPTS:
            try:
                await asyncio.sleep(settings.WS_RECONNECT_INTERVAL)
                await self.connect_websocket(agent)
                return
            except Exception:
                attempts += 1
        
        agent.status = "unreachable"
        self._reconnect_tasks.pop(str(agent.id), None)
    
    async def _handle_agent_event(self, agent_id: str, event: dict) -> None:
        """Process event from Agent."""
        # Update database based on event type
        # server.started, server.stopped, gpu.usage, etc.
        pass
    
    async def send_to_agent(
        self, 
        agent_id: str, 
        method: str, 
        path: str, 
        json: dict | None = None
    ) -> dict:
        """Send HTTP request to Agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        url = f"http://{agent.host}:{agent.port}/api{path}"
        
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=url,
                json=json,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
```

---

## Phase 2: Agent Service Core (Days 4-8)

### 2.1 llama.cpp Server Management

**File:** `agent/app/services/llama_server.py`
```python
"""Manages llama.cpp subprocess lifecycle."""

import asyncio
import subprocess
import signal
from typing import Dict, Optional
from dataclasses import dataclass

from app.core.config import settings

@dataclass
class ServerConfig:
    model_path: str
    port: int
    gpu_layers: int = settings.DEFAULT_GPU_LAYERS
    context_size: int = settings.DEFAULT_CONTEXT_SIZE
    batch_size: int = settings.DEFAULT_BATCH_SIZE
    cache_prompt: bool = True
    flash_attn: bool = True

class LlamaServerManager:
    """Manages llama.cpp subprocesses."""
    
    def __init__(self) -> None:
        self.servers: Dict[str, subprocess.Popen] = {}
        self.configs: Dict[str, ServerConfig] = {}
    
    async def start_server(
        self, 
        server_id: str, 
        config: ServerConfig
    ) -> bool:
        """Start a llama.cpp server subprocess."""
        if server_id in self.servers:
            return False
        
        cmd = [
            settings.LLAMA_SERVER_PATH,
            "--model", config.model_path,
            "--port", str(config.port),
            "--n-gpu-layers", str(config.gpu_layers),
            "--ctx-size", str(config.context_size),
            "--batch-size", str(config.batch_size),
        ]
        
        if config.cache_prompt:
            cmd.append("--prompt-cache")
            cmd.append(f"/cache/{server_id}.cache")
        
        if config.flash_attn:
            cmd.append("--flash-attn")
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        self.servers[server_id] = proc
        self.configs[server_id] = config
        
        # Wait for server to start
        await self._wait_for_server(server_id, config.port)
        
        return True
    
    async def stop_server(self, server_id: str, force: bool = False) -> bool:
        """Stop a llama.cpp server."""
        if server_id not in self.servers:
            return False
        
        proc = self.servers[server_id]
        
        if force:
            proc.kill()
        else:
            proc.send_signal(signal.SIGTERM)
        
        proc.wait(timeout=30)
        
        self.servers.pop(server_id)
        self.configs.pop(server_id)
        
        return True
    
    async def _wait_for_server(
        self, 
        server_id: str, 
        port: int, 
        timeout: float = 30.0
    ) -> None:
        """Wait for server to be healthy."""
        import time
        start = time.time()
        
        while time.time() - start < timeout:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(
                        f"http://localhost:{port}/health"
                    )
                    if response.status_code == 200:
                        return
                except Exception:
                    pass
            
            await asyncio.sleep(1)
        
        raise TimeoutError(f"Server {server_id} failed to start")
```

---

### 2.2 Agent API Routes

**File:** `agent/app/api/routes/servers.py`
```python
"""Server management endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid

from app.services.llama_server import LlamaServerManager, ServerConfig

router = APIRouter(prefix="/servers", tags=["servers"])
server_manager = LlamaServerManager()

class StartServerRequest(BaseModel):
    model_id: str
    model_path: str
    config: dict

class StartServerResponse(BaseModel):
    server_id: str
    status: str
    proxy_url: str

@router.post("/start", response_model=StartServerResponse)
async def start_server(request: StartServerRequest) -> StartServerResponse:
    """Start a llama.cpp server."""
    server_id = str(uuid.uuid4())
    
    config = ServerConfig(
        model_path=request.model_path,
        port=8081,  # Dynamic port assignment
        **request.config
    )
    
    success = await server_manager.start_server(server_id, config)
    
    if not success:
        raise HTTPException(400, "Server already running")
    
    return StartServerResponse(
        server_id=server_id,
        status="running",
        proxy_url=f"http://localhost:8080/proxy/{server_id}"
    )

@router.post("/{server_id}/stop")
async def stop_server(server_id: str, force: bool = False) -> dict:
    """Stop a llama.cpp server."""
    success = await server_manager.stop_server(server_id, force)
    
    if not success:
        raise HTTPException(404, "Server not found")
    
    return {"status": "stopped"}

@router.get("")
async def list_servers() -> dict:
    """List all running servers."""
    servers = []
    
    for server_id, proc in server_manager.servers.items():
        servers.append({
            "server_id": server_id,
            "model_id": server_manager.configs[server_id].model_path,
            "status": "running",
            "uptime_seconds": 0,  # Calculate from start time
        })
    
    return {"servers": servers}
```

**File:** `agent/app/api/routes/websocket.py`
```python
"""WebSocket event streaming."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json

from app.core.config import settings

router = APIRouter(tags=["websocket"])

class EventBuffer:
    """Buffer events for replay on reconnect."""
    
    def __init__(self, max_events: int = 1000) -> None:
        self.events: list[dict] = []
        self.max_events = max_events
    
    def add(self, event: dict) -> None:
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)
    
    def get_all(self) -> list[dict]:
        return self.events.copy()

event_buffer = EventBuffer(settings.WS_MAX_BUFFER_EVENTS)

@router.websocket("/ws/status")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket connection for real-time events."""
    await websocket.accept()
    
    agent_id = websocket.headers.get("X-Agent-ID")
    
    try:
        # Send buffered events on reconnect
        for event in event_buffer.get_all():
            await websocket.send_json(event)
        
        # Keep connection alive
        while True:
            await websocket.send_json({
                "event": "heartbeat",
                "data": {"timestamp": asyncio.get_event_loop().time()}
            })
            await asyncio.sleep(settings.WS_HEARTBEAT_INTERVAL)
            
    except WebSocketDisconnect:
        pass
```

---

### 2.3 Frontend Registration

**File:** `agent/app/services/frontend_client.py`
```python
"""Manages connection to Frontend Service."""

import asyncio
import httpx
from websockets.client import connect

from app.core.config import settings

class FrontendClient:
    """Handles Frontend registration and WebSocket connection."""
    
    def __init__(self) -> None:
        self.registered = False
        self.ws_connected = False
    
    async def register(self) -> bool:
        """Register Agent with Frontend."""
        import socket
        
        registration_data = {
            "agent_id": settings.AGENT_ID,
            "name": settings.AGENT_NAME,
            "host": socket.gethostname(),
            "port": 8080,
            "gpu_info": await self._get_gpu_info()
        }
        
        url = f"{settings.FRONTEND_URL}/api/agents/register"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=registration_data)
                response.raise_for_status()
                self.registered = True
                return True
            except Exception as e:
                print(f"Registration failed: {e}")
                return False
    
    async def _get_gpu_info(self) -> dict:
        """Get GPU information."""
        # Implement GPU detection based on backend
        return {
            "name": "NVIDIA RTX 4090",
            "vram_total": 24576000000,
            "backend": "cuda"
        }
    
    async def connect_websocket(self) -> None:
        """Establish WebSocket connection to Frontend."""
        ws_url = f"ws://{settings.FRONTEND_URL.replace('http://', '')}/api/ws/agents/{settings.AGENT_ID}"
        
        while True:
            try:
                async with connect(ws_url) as websocket:
                    self.ws_connected = True
                    
                    while True:
                        # Send events from buffer
                        message = await websocket.recv()
                        # Handle commands from Frontend
                        
            except Exception as e:
                self.ws_connected = False
                await asyncio.sleep(settings.WS_RECONNECT_INTERVAL)
```

---

## Phase 3: Integration (Days 9-12)

### 3.1 Proxy Implementation

**File:** `agent/app/services/proxy.py`
```python
"""Proxies llama.cpp HTTP API."""

from typing import AsyncGenerator
import httpx

class LlamaCppProxy:
    """Proxies requests to llama.cpp servers."""
    
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def proxy_request(
        self,
        server_id: str,
        method: str,
        path: str,
        headers: dict,
        json: dict | None = None
    ) -> httpx.Response:
        """Proxy HTTP request to llama.cpp."""
        # Get server port from server manager
        port = self._get_server_port(server_id)
        
        url = f"http://localhost:{port}{path}"
        
        response = await self.client.request(
            method=method,
            url=url,
            headers=headers,
            json=json
        )
        
        return response
    
    async def proxy_stream(
        self,
        server_id: str,
        method: str,
        path: str,
        json: dict | None = None
    ) -> AsyncGenerator[bytes, None]:
        """Proxy streaming request (SSE) to llama.cpp."""
        port = self._get_server_port(server_id)
        url = f"http://localhost:{port}{path}"
        
        async with self.client.stream(
            method=method,
            url=url,
            json=json
        ) as response:
            async for chunk in response.aiter_bytes():
                yield chunk
```

---

### 3.2 Frontend Inference Flow

**File:** `backend/app/api/routes/v1/v1_chat_completions.py` (update)
```python
"""Chat completions endpoint - updated for Agent proxy."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import httpx

from app.models import Model, ServerInstance, Agent
from app.db.session import session_maker
from app.services.agent_manager import agent_manager

router = APIRouter(prefix="/chat/completions", tags=["v1/chat/completions"])

class ChatCompletionRequest(BaseModel):
    model: str
    agent_id: str | None = None
    messages: list[dict]
    stream: bool = False
    # ... other OpenAI parameters

@router.post("")
async def create_chat_completion(
    request: ChatCompletionRequest,
    http_request: Request
):
    """Create chat completion via Agent proxy."""
    async with session_maker() as session:
        # Find model
        model = await session.get(Model, request.model)
        if not model:
            raise HTTPException(404, "Model not found")
        
        # Find or create server
        if request.agent_id:
            agent = await session.get(Agent, request.agent_id)
        else:
            # Find agent with this model
            agent = await find_agent_with_model(model.id)
        
        if not agent:
            raise HTTPException(503, "No agent available for model")
        
        # Check if server is running
        server = await get_server_for_model(model.id, agent.id)
        
        if not server:
            # Start server via Agent
            await agent_manager.send_to_agent(
                agent.id,
                "POST",
                "/servers/start",
                {
                    "model_id": model.id,
                    "model_path": model.path,
                    "config": {...}
                }
            )
            # Wait for server.started event
            server = await wait_for_server(model.id, agent.id)
        
        # Proxy request through Agent
        if request.stream:
            return StreamingResponse(
                stream_inference_request(server, request),
                media_type="text/event-stream"
            )
        else:
            response = await agent_manager.send_to_agent(
                agent.id,
                "POST",
                f"/proxy/{server.id}/v1/chat/completions",
                request.dict()
            )
            return response
```

---

## Phase 4: Testing & Deployment (Days 13-15)

### 4.1 Testing

**Agent Tests:**
```bash
cd agent
uv run pytest tests/ -v

# Test coverage
uv run pytest --cov=app tests/
```

**Frontend Tests:**
```bash
cd backend
uv run pytest tests/ -v

# Test Agent integration
uv run pytest tests/services/test_agent_manager.py -v
```

### 4.2 Docker Configuration

**File:** `agent/Dockerfile`
```dockerfile
FROM python:3.14-slim

WORKDIR /app

# Install llama.cpp
RUN apt-get update && apt-get install -y \
    cmake \
    cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ggerganov/llama.cpp.git \
    && cd llama.cpp \
    && cmake -B build -DLLAMA_CUDA=ON \
    && cmake --build build --config Release \
    && cp build/bin/llama-server /usr/local/bin/

# Install Python dependencies
COPY pyproject.toml .
RUN pip install uv && uv sync --frozen

# Copy code
COPY app/ ./app/

EXPOSE 8080

CMD ["uv", "run", "python", "-m", "app.main"]
```

**File:** `compose.yml` (update)
```yaml
services:
  frontend:
    build: ./backend
    ports:
      - "3000:3000"
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+psycopg://inference:secret@postgres/inference_matrix
      - AGENT_DISCOVERY_ENABLED=true
    depends_on:
      - postgres

  agent:
    build: ./agent
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models
      - ./cache:/cache
    environment:
      - AGENT_ID=agent-1
      - AGENT_NAME=GPU-Agent-1
      - FRONTEND_URL=http://frontend:8000
      - LLAMA_SERVER_PATH=/usr/local/bin/llama-server
    devices:
      - /dev/nvidia0:/dev/nvidia0
      - /dev/nvidiactl:/dev/nvidiactl
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  postgres:
    image: postgres:16
    volumes:
      - postgres_data:/var/lib/postgresql/data
```

---

## Phase 5: Documentation & Polish (Days 16-18)

### 5.1 Update Documentation

- ✅ docs/architecture.md - Already updated
- ✅ docs/api-endpoints.md - Already updated
- Update docs/deployment.md - Add Agent deployment
- Update docs/user-guide.md - Multi-agent workflows
- Add docs/agent-guide.md - Agent-specific documentation

### 5.2 Monitoring Dashboard

Add Agent status to WebUI:
- Agent online/offline status
- GPU utilization per Agent
- Server status per Agent
- WebSocket connection health

---

## Testing Checklist

### Agent Service
- [ ] Agent registers with Frontend on startup
- [ ] WebSocket connection established
- [ ] Server start/stop works
- [ ] llama.cpp subprocess management
- [ ] GPU monitoring reports correct data
- [ ] Model download from HuggingFace
- [ ] Model download from ModelScope
- [ ] Proxy forwards requests correctly
- [ ] SSE streaming through proxy
- [ ] Cache save/load works
- [ ] Agent reconnects after Frontend restart

### Frontend Service
- [ ] Agent registration endpoint works
- [ ] WebSocket receives events from Agent
- [ ] List models across all Agents
- [ ] Start server on remote Agent
- [ ] Inference requests proxied correctly
- [ ] Streaming responses work
- [ ] Cache operations through proxy
- [ ] Agent failure detection
- [ ] Agent reconnection
- [ ] Multi-agent model selection

### Integration
- [ ] End-to-end inference request
- [ ] Multiple Agents running
- [ ] Agent restart doesn't lose servers
- [ ] Frontend restart reconnects to Agents
- [ ] Cache persists across restarts
- [ ] GPU monitoring accurate
- [ ] Download progress reported correctly

---

## Deployment Checklist

- [ ] Docker Compose starts both services
- [ ] Agent has GPU access
- [ ] Frontend can reach Agent
- [ ] Agent can reach Frontend
- [ ] Database migrations applied
- [ ] Models directory shared/mounted
- [ ] Cache directory mounted
- [ ] Health checks passing
- [ ] Logs accessible
- [ ] Metrics exposed

---

## Next Steps After MVP

1. **Automatic Agent Load Balancing** - Distribute requests across Agents
2. **Agent Failover** - Automatic server migration on Agent failure
3. **Distributed Cache** - Shared cache storage across Agents
4. **Agent WebUI** - Status page for debugging
5. **Additional Audio Endpoints** - Full Whisper integration on Agent
6. **Model Sync** - Automatic model replication across Agents

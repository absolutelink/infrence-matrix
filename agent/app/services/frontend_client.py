"""Manages connection to Frontend Service."""

import asyncio
import json
import socket

import httpx
from websockets.client import connect

from app.core.config import settings
from app.core.logging import logger
from app.services.event_bus import publish_event
from app.services.gpu_monitor import _sample_gpu, aggregate_gpu_info


class FrontendClient:
    """Handles Frontend registration and WebSocket connection."""

    # Re-register periodically: if the backend restarts it loses its
    # in-memory agent cache and needs a fresh register call to re-open
    # the event WebSocket.
    REREGISTER_INTERVAL: int = 60

    def __init__(self) -> None:
        self.registered = False
        self.ws_connected = False
        self._ws_task: asyncio.Task | None = None
        self._ws_connection = None

    async def register(self) -> bool:
        """Register Agent with Frontend."""
        from app.services.inference_operations import INFERENCE_SLOT_PROTOCOL_VERSION
        from app.services.server_manager import server_manager

        registration_data = {
            "agent_id": settings.AGENT_ID,
            "name": settings.AGENT_NAME,
            "platform": settings.AGENT_PLATFORM,
            "type": settings.AGENT_TYPE,
            "inference_slot_protocol": INFERENCE_SLOT_PROTOCOL_VERSION,
            "host": settings.AGENT_HOST or socket.gethostname(),
            "port": settings.AGENT_PORT,
            "gpu_info": await self._get_gpu_info(),
            # Server ids live in the agent's memory; the backend uses this
            # to only clear instances that are no longer actually running.
            "running_server_ids": list(server_manager.servers.keys()),
            "healthy_server_ids": list(server_manager.healthy_servers),
            "server_statuses": [
                {
                    "id": server_id,
                    "status": (
                        "running"
                        if server_id in server_manager.healthy_servers
                        else "starting"
                    ),
                    "health_status": (
                        "healthy"
                        if server_id in server_manager.healthy_servers
                        else "unknown"
                    ),
                    "port": config.port,
                    "slot_generation": config.slot_generation,
                    "effective_capacity": server_manager.get_effective_capacity(
                        server_id
                    ),
                    "active_inference_requests": getattr(
                        server_manager, "_active_inference_requests", {}
                    ).get(server_id, 0),
                }
                for server_id, config in server_manager.configs.items()
                if server_id in server_manager.servers
            ],
        }

        url = f"{settings.FRONTEND_URL}/api/v1/agents/register"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=registration_data)
                response.raise_for_status()
                self.registered = True
                logger.info(f"Registered with frontend at {url}")
                return True
            except Exception as e:
                logger.error(f"Registration failed: {e}")
                return False

    async def _get_gpu_info(self) -> dict:
        """Get GPU information."""
        return aggregate_gpu_info(_sample_gpu(), settings.GPU_BACKEND)

    async def start_background_tasks(self) -> None:
        """Start background tasks: register with frontend, then connect WebSocket."""
        self._ws_task = asyncio.create_task(self._connection_loop())

    async def _connection_loop(self) -> None:
        """Register with the frontend, then keep the WebSocket alive.

        Re-registers periodically so a backend restart re-opens the
        backend->agent event WebSocket (registration is what triggers
        the backend to connect).
        """
        while not self.registered:
            if await self.register():
                break
            await asyncio.sleep(settings.WS_RECONNECT_INTERVAL)

        reregister_task = asyncio.create_task(self._reregister_loop())

        try:
            await self.connect_websocket()
        finally:
            reregister_task.cancel()

    async def _reregister_loop(self) -> None:
        """Re-register with the frontend on an interval."""
        while True:
            await asyncio.sleep(self.REREGISTER_INTERVAL)
            await self.register()

    def _build_ws_url(self, frontend_url: str) -> str:
        """Convert the frontend HTTP(S) URL to a WebSocket URL."""
        if frontend_url.startswith("https://"):
            scheme = "wss://"
        elif frontend_url.startswith("http://"):
            scheme = "ws://"
        else:
            raise ValueError(
                f"FRONTEND_URL must start with http:// or https://, got: {frontend_url}"
            )

        host = frontend_url.split("://", 1)[1].rstrip("/")
        return f"{scheme}{host}/api/ws/agents/{settings.AGENT_ID}"

    async def connect_websocket(self) -> None:
        """Establish WebSocket connection to Frontend."""
        ws_url = self._build_ws_url(settings.FRONTEND_URL)

        while True:
            try:
                async with connect(
                    ws_url,
                    ping_interval=settings.WS_PING_INTERVAL,
                    ping_timeout=settings.WS_PING_TIMEOUT,
                ) as websocket:
                    self._ws_connection = websocket
                    self.ws_connected = True
                    logger.info("WebSocket connected to frontend")

                    while True:
                        message = await websocket.recv()
                        await self._handle_command(message)

            except Exception as e:
                self.ws_connected = False
                self._ws_connection = None
                logger.error(f"WebSocket connection error: {e}")
                await asyncio.sleep(settings.WS_RECONNECT_INTERVAL)

    async def _handle_command(self, message: str | bytes) -> None:
        """Handle command from Frontend."""
        try:
            command = json.loads(message)
            cmd_type = command.get("type")

            if cmd_type == "start_server":
                await self._handle_start_server(command)
            elif cmd_type == "stop_server":
                await self._handle_stop_server(command)
            elif cmd_type == "update_model":
                await self._handle_update_model(command)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse command: {message}")

    async def _handle_start_server(self, command: dict) -> None:
        """Handle start server command."""
        from app.services.server_manager import server_manager

        server_config = command.get("config", {})
        server_id = server_config.get("id")
        slot_generation = int(server_config.get("slot_generation", 0))
        try:
            config = server_manager.ServerConfig(
                **{key: value for key, value in server_config.items() if key != "id"}
            )
            await server_manager.start_server(server_id, config)
            await self._send_status_update(
                "server.started",
                {
                    "server_id": server_id,
                    "status": "running",
                    "slot_generation": slot_generation,
                    "effective_capacity": server_manager.get_effective_capacity(
                        server_id
                    ),
                },
            )
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            await self._send_status_update(
                "server.error",
                {
                    "server_id": server_id,
                    "error": str(e),
                    "slot_generation": slot_generation,
                },
            )

    async def _handle_stop_server(self, command: dict) -> None:
        """Handle stop server command."""
        from app.services.server_manager import server_manager

        server_id = command.get("server_id")
        slot_generation = int(command.get("slot_generation") or 0)
        try:
            await server_manager.stop_server(server_id, slot_generation=slot_generation)
            await self._send_status_update(
                "server.stopped",
                {
                    "server_id": server_id,
                    "status": "stopped",
                    "slot_generation": slot_generation,
                },
            )
        except Exception as e:
            logger.error(f"Failed to stop server: {e}")
            await self._send_status_update(
                "server.error",
                {
                    "server_id": server_id,
                    "error": str(e),
                    "slot_generation": slot_generation,
                },
            )

    async def _handle_update_model(self, command: dict) -> None:
        """Handle update model command."""
        from app.services.model_manager import model_manager

        model_id = command.get("model_id")
        try:
            await model_manager.update_model(model_id)
            await self._send_status_update(
                "model.updated",
                {
                    "model_id": model_id,
                    "status": "updated",
                },
            )
        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            await self._send_status_update(
                "model.error",
                {
                    "model_id": model_id,
                    "error": str(e),
                },
            )

    async def _send_status_update(self, event_type: str, data: dict) -> None:
        """Publish a status update on the shared event bus."""
        publish_event(event_type, data)

    async def close(self) -> None:
        """Close connections."""
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass


frontend_client = FrontendClient()

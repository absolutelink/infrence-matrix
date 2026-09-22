"""Manages connection to Frontend Service."""

import asyncio
import json
import socket

import httpx
from websockets.client import connect

from app.core.config import settings
from app.core.logging import logger
from app.services.event_bus import publish_event


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
        from app.services.llama_server import llama_server_manager

        registration_data = {
            "agent_id": settings.AGENT_ID,
            "name": settings.AGENT_NAME,
            "host": settings.AGENT_HOST or socket.gethostname(),
            "port": settings.AGENT_PORT,
            "gpu_info": await self._get_gpu_info(),
            # Server ids live in the agent's memory; the backend uses this
            # to only clear instances that are no longer actually running.
            "running_server_ids": list(llama_server_manager.servers.keys()),
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
        gpu_info = {
            "name": "Unknown",
            "vram_total": 0,
            "backend": settings.GPU_BACKEND,
        }

        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                gpu_info["name"] = parts[0]
                gpu_info["vram_total"] = int(parts[1]) * 1024 * 1024
                gpu_info["backend"] = "cuda"
        except Exception:
            pass

        return gpu_info

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
                async with connect(ws_url) as websocket:
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
        from app.services.llama_server import llama_server_manager

        server_config = command.get("config", {})
        try:
            config = llama_server_manager.ServerConfig(**server_config)
            await llama_server_manager.start_server(server_config.get("id"), config)
            await self._send_status_update(
                "server.started",
                {
                    "server_id": server_config.get("id"),
                    "status": "running",
                },
            )
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            await self._send_status_update(
                "server.error",
                {
                    "server_id": server_config.get("id"),
                    "error": str(e),
                },
            )

    async def _handle_stop_server(self, command: dict) -> None:
        """Handle stop server command."""
        from app.services.llama_server import llama_server_manager

        server_id = command.get("server_id")
        try:
            await llama_server_manager.stop_server(server_id)
            await self._send_status_update(
                "server.stopped",
                {
                    "server_id": server_id,
                    "status": "stopped",
                },
            )
        except Exception as e:
            logger.error(f"Failed to stop server: {e}")
            await self._send_status_update(
                "server.error",
                {
                    "server_id": server_id,
                    "error": str(e),
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

"""Manages connection to Frontend Service."""

import asyncio
import socket
import httpx
from websockets.client import connect

from app.core.config import settings


class FrontendClient:
    """Handles Frontend registration and WebSocket connection."""

    def __init__(self) -> None:
        self.registered = False
        self.ws_connected = False
        self._ws_task: asyncio.Task | None = None

    async def register(self) -> bool:
        """Register Agent with Frontend."""
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
        gpu_info = {
            "name": "Unknown",
            "vram_total": 0,
            "backend": "auto"
        }

        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                gpu_info["name"] = parts[0]
                gpu_info["vram_total"] = int(parts[1]) * 1024 * 1024
                gpu_info["backend"] = "cuda"
        except Exception:
            pass

        return gpu_info

    async def connect_websocket(self) -> None:
        """Establish WebSocket connection to Frontend."""
        ws_url = f"ws://{settings.FRONTEND_URL.replace('http://', '').replace('https://', '')}/api/ws/agents/{settings.AGENT_ID}"

        while True:
            try:
                async with connect(ws_url) as websocket:
                    self.ws_connected = True

                    while True:
                        message = await websocket.recv()
                        await self._handle_command(message)

            except Exception as e:
                self.ws_connected = False
                await asyncio.sleep(settings.WS_RECONNECT_INTERVAL)

    async def _handle_command(self, message: str) -> None:
        """Handle command from Frontend."""
        import json

        try:
            command = json.loads(message)
            cmd_type = command.get("type")

            if cmd_type == "start_server":
                pass
            elif cmd_type == "stop_server":
                pass
        except json.JSONDecodeError:
            pass

    async def start_background_tasks(self) -> None:
        """Start background tasks for Frontend connection."""
        self._ws_task = asyncio.create_task(self.connect_websocket())

    async def close(self) -> None:
        """Close connections."""
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass


frontend_client = FrontendClient()

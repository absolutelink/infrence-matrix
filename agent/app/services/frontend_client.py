"""Manages connection to Frontend Service."""

import asyncio
import socket
import httpx
from datetime import datetime
from websockets.client import connect

from app.core.config import settings
from app.core.logging import logger


class FrontendClient:
    """Handles Frontend registration and WebSocket connection."""

    def __init__(self) -> None:
        self.registered = False
        self.ws_connected = False
        self._ws_task: asyncio.Task | None = None
        self._ws_connection: any = None

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
                logger.error(f"Registration failed: {e}")
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

    async def _handle_command(self, message: str) -> None:
        """Handle command from Frontend."""
        import json

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
            # Start the llama-server
            config = llama_server_manager.ServerConfig(**server_config)
            await llama_server_manager.start_server(server_config.get("id"), config)
            # Send status update back to frontend
            await self._send_status_update("server.started", {
                "server_id": server_config.get("id"),
                "status": "running"
            })
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            await self._send_status_update("server.error", {
                "server_id": server_config.get("id"),
                "error": str(e)
            })

    async def _handle_stop_server(self, command: dict) -> None:
        """Handle stop server command."""
        from app.services.llama_server import llama_server_manager
        
        server_id = command.get("server_id")
        try:
            # Stop the llama-server
            await llama_server_manager.stop_server(server_id)
            # Send status update back to frontend
            await self._send_status_update("server.stopped", {
                "server_id": server_id,
                "status": "stopped"
            })
        except Exception as e:
            logger.error(f"Failed to stop server: {e}")
            await self._send_status_update("server.error", {
                "server_id": server_id,
                "error": str(e)
            })

    async def _handle_update_model(self, command: dict) -> None:
        """Handle update model command."""
        from app.services.model_manager import model_manager
        
        model_id = command.get("model_id")
        try:
            # Update model
            await model_manager.update_model(model_id)
            # Send status update back to frontend
            await self._send_status_update("model.updated", {
                "model_id": model_id,
                "status": "updated"
            })
        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            await self._send_status_update("model.error", {
                "model_id": model_id,
                "error": str(e)
            })

    async def _send_status_update(self, event_type: str, data: dict) -> None:
        """Send status update to Frontend."""
        import json
        
        if not self.ws_connected or not self._ws_connection:
            return
            
        try:
            message = {
                "event": event_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self._ws_connection.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send status update: {e}")

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
                await self._handle_start_server(command)
            elif cmd_type == "stop_server":
                await self._handle_stop_server(command)
            elif cmd_type == "update_model":
                await self._handle_update_model(command)
        except json.JSONDecodeError:
            pass

    async def _handle_start_server(self, command: dict) -> None:
        """Handle start server command."""
        from app.services.llama_server import llama_server_manager
        
        server_config = command.get("config", {})
        try:
            # Start the llama-server
            await llama_server_manager.start_server(server_config)
            # Send status update back to frontend
            await self._send_status_update("server.started", {
                "server_id": server_config.get("id"),
                "status": "running"
            })
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            await self._send_status_update("server.error", {
                "server_id": server_config.get("id"),
                "error": str(e)
            })

    async def _handle_stop_server(self, command: dict) -> None:
        """Handle stop server command."""
        from app.services.llama_server import llama_server_manager
        
        server_id = command.get("server_id")
        try:
            # Stop the llama-server
            await llama_server_manager.stop_server(server_id)
            # Send status update back to frontend
            await self._send_status_update("server.stopped", {
                "server_id": server_id,
                "status": "stopped"
            })
        except Exception as e:
            logger.error(f"Failed to stop server: {e}")
            await self._send_status_update("server.error", {
                "server_id": server_id,
                "error": str(e)
            })

    async def _handle_update_model(self, command: dict) -> None:
        """Handle update model command."""
        from app.services.model_manager import model_manager
        
        model_id = command.get("model_id")
        try:
            # Update model
            await model_manager.update_model(model_id)
            # Send status update back to frontend
            await self._send_status_update("model.updated", {
                "model_id": model_id,
                "status": "updated"
            })
        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            await self._send_status_update("model.error", {
                "model_id": model_id,
                "error": str(e)
            })

    async def _send_status_update(self, event_type: str, data: dict) -> None:
        """Send status update to Frontend."""
        import json
        
        if not self.ws_connected:
            return
            
        try:
            message = {
                "event": event_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self._ws_task._coro.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send status update: {e}")

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

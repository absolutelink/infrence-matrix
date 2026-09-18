"""Manages connection to Frontend Service."""

import asyncio
import socket

import httpx
from websockets.client import connect

from app.core.config import settings
from app.core.logging import logger
from app.services.gpu_monitor import GPUMonitor


class FrontendClient:
    """Handles Frontend registration and WebSocket connection."""
    
    def __init__(self) -> None:
        self.registered = False
        self.ws_connected = False
        self.gpu_monitor = GPUMonitor()
    
    async def register(self) -> bool:
        """Register Agent with Frontend."""
        hostname = socket.gethostname()
        
        registration_data = {
            "agent_id": settings.AGENT_ID,
            "name": settings.AGENT_NAME,
            "host": hostname,
            "port": settings.PORT,
            "gpu_info": await self.gpu_monitor.get_gpu_info(),
        }
        
        url = f"{settings.FRONTEND_URL}/api/agents/register"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=registration_data)
                response.raise_for_status()
                self.registered = True
                logger.info(f"Registered with Frontend at {url}")
                return True
            except Exception as e:
                logger.error(f"Registration failed: {e}")
                return False
    
    async def connect_websocket(self) -> None:
        """Establish WebSocket connection to Frontend."""
        # Convert http:// to ws://
        ws_url = settings.FRONTEND_URL.replace("http://", "ws://")
        ws_url = f"{ws_url}/api/ws/agents/{settings.AGENT_ID}"
        
        while True:
            try:
                async with connect(ws_url) as websocket:
                    self.ws_connected = True
                    logger.info(f"Connected to Frontend WebSocket: {ws_url}")
                    
                    while True:
                        # Listen for commands from Frontend
                        message = await websocket.recv()
                        await self._handle_command(message, websocket)
                        
            except Exception as e:
                self.ws_connected = False
                logger.error(f"WebSocket error: {e}, reconnecting in {settings.WS_RECONNECT_INTERVAL}s")
                await asyncio.sleep(settings.WS_RECONNECT_INTERVAL)
    
    async def _handle_command(self, message: str, websocket) -> None:
        """Handle command from Frontend."""
        # TODO: Implement command handling
        pass

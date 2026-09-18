from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
import json
from typing import Dict, List

from app.core.config import settings
from app.core.logging import logger

router = APIRouter(tags=["websocket"])


class EventBuffer:
    """Buffer events for replay on reconnect."""
    
    def __init__(self, max_events: int = 1000) -> None:
        self.events: List[dict] = []
        self.max_events = max_events
    
    def add(self, event: dict) -> None:
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)
    
    def get_all(self) -> List[dict]:
        return self.events.copy()


event_buffer = EventBuffer(settings.WS_MAX_BUFFER_EVENTS)


@router.websocket("/ws/status")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket connection for real-time events."""
    await websocket.accept()
    
    agent_id = websocket.headers.get("X-Agent-ID")
    logger.info(f"WebSocket connection from agent {agent_id}")
    
    try:
        # Send buffered events on reconnect
        for event in event_buffer.get_all():
            await websocket.send_json(event)
        
        # Keep connection alive with heartbeat
        while True:
            await websocket.send_json({
                "event": "heartbeat",
                "data": {"timestamp": asyncio.get_event_loop().time()}
            })
            await asyncio.sleep(settings.WS_HEARTBEAT_INTERVAL)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for agent {agent_id}")

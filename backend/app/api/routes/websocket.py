"""WebSocket endpoints for agent communication."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.agent_manager import agent_manager
from app.core.logging import logger

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/agents/{agent_id}")
async def agent_websocket_endpoint(websocket: WebSocket, agent_id: str) -> None:
    """WebSocket connection for agent communication."""
    await websocket.accept()

    logger.info(f"Agent {agent_id} connected via WebSocket")

    try:
        # Send any buffered events
        if agent_id in agent_manager._event_buffers:
            buffer = agent_manager._event_buffers[agent_id]
            for event in buffer:
                await websocket.send_json(event)

        # Keep connection alive
        while True:
            # Listen for messages from agent
            message = await websocket.receive_json()
            logger.debug(f"Received message from agent {agent_id}: {message}")

            # Process message (could be status update, event, etc.)
            # For now, just acknowledge
            await websocket.send_json({
                "status": "received",
                "message": message,
            })

    except WebSocketDisconnect:
        logger.info(f"Agent {agent_id} disconnected")
        # Agent manager will handle reconnection
    except Exception as e:
        logger.error(f"WebSocket error for agent {agent_id}: {e}")

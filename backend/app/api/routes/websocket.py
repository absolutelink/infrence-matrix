"""WebSocket endpoints for agent communication."""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logging import logger
from app.services.agent_manager import agent_manager
from app.services.inference_scheduler import inference_scheduler

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
            await websocket.send_json(
                {
                    "status": "received",
                    "message": message,
                }
            )

    except WebSocketDisconnect:
        logger.info(f"Agent {agent_id} disconnected")
        # Agent manager will handle reconnection
    except Exception as e:
        logger.error(f"WebSocket error for agent {agent_id}: {e}")


@router.websocket("/ws/events/{agent_id}")
async def agent_events_websocket(websocket: WebSocket, agent_id: str) -> None:
    """Stream an agent's events to UI clients."""
    await websocket.accept()

    queue = agent_manager.subscribe_events(agent_id)
    logger.info(f"UI client subscribed to events for agent {agent_id}")

    async def forward() -> None:
        while True:
            event = await queue.get()
            await websocket.send_json(event)

    forward_task = asyncio.create_task(forward())
    try:
        # Wait for disconnect while the forward task pushes events.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"Events WebSocket error for agent {agent_id}: {e}")
    finally:
        forward_task.cancel()
        agent_manager.unsubscribe_events(agent_id, queue)
        logger.info(f"UI client unsubscribed from agent {agent_id} events")


@router.websocket("/ws/queue-status")
async def queue_status_websocket(websocket: WebSocket) -> None:
    """Push aggregate inference queue and slot state to the UI."""
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(await inference_scheduler.status_snapshot())
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"Queue status WebSocket error: {e}")

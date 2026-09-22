"""WebSocket event streaming."""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.logging import logger
from app.services.event_bus import event_buffer, subscribe, unsubscribe

router = APIRouter(tags=["websocket"])


async def _heartbeat(websocket: WebSocket) -> None:
    """Send periodic heartbeats until cancelled."""
    while True:
        await websocket.send_json(
            {
                "event": "heartbeat",
                "data": {"timestamp": asyncio.get_event_loop().time()},
            }
        )
        await asyncio.sleep(settings.WS_HEARTBEAT_INTERVAL)


@router.websocket("/ws/status")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket connection for real-time events."""
    await websocket.accept()

    queue = subscribe()
    heartbeat_task = asyncio.create_task(_heartbeat(websocket))
    sender_task = asyncio.create_task(_send_events(websocket, queue))

    try:
        # Keep the connection open until the client disconnects. Any send
        # failure in the background tasks surfaces here as an exception.
        while True:
            await asyncio.sleep(3600)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as e:  # noqa: BLE001 - keep connection errors from crashing the endpoint
        logger.warning(f"WebSocket error: {e}")
    finally:
        heartbeat_task.cancel()
        sender_task.cancel()
        unsubscribe(queue)


async def _send_events(websocket: WebSocket, queue: asyncio.Queue) -> None:
    """Forward queued events to the WebSocket client."""
    # Replay buffered events on (re)connect
    for event in event_buffer.get_all():
        await websocket.send_json(event)

    while True:
        event = await queue.get()
        await websocket.send_json(event)

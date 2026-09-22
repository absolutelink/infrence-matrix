"""Event bus for broadcasting agent events to WebSocket subscribers."""

import asyncio
from datetime import UTC, datetime

from app.core.config import settings
from app.core.logging import logger


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

_subscribers: set[asyncio.Queue] = set()


def subscribe() -> asyncio.Queue:
    """Register a subscriber queue for event delivery."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=settings.WS_MAX_BUFFER_EVENTS)
    _subscribers.add(queue)
    return queue


def unsubscribe(queue: asyncio.Queue) -> None:
    """Remove a subscriber queue."""
    _subscribers.discard(queue)


def publish_event(event_type: str, data: dict) -> dict:
    """Publish an event to the buffer and all connected subscribers."""
    event = {
        "event": event_type,
        "data": data,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    event_buffer.add(event)

    for queue in list(_subscribers):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(f"Subscriber queue full, dropped event {event_type}")

    logger.debug(f"Published event {event_type}: {data}")
    return event

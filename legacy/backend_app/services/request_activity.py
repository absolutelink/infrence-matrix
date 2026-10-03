"""Process-local inference request activity used by benchmark coordination."""

import asyncio

_active_requests = 0
_lock = asyncio.Lock()


async def request_started() -> None:
    global _active_requests
    async with _lock:
        _active_requests += 1


async def request_finished() -> None:
    global _active_requests
    async with _lock:
        _active_requests = max(0, _active_requests - 1)


async def active_request_count() -> int:
    async with _lock:
        return _active_requests

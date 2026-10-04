"""Shared async Redis client wiring.

The client is created in the app lifespan and stored on ``app.state.redis``.
Helpers here retrieve it for HTTP dependencies and the connection manager.
"""

from typing import Any

import redis.asyncio as aioredis
from fastapi import HTTPException, Request
from starlette.applications import Starlette

from app.core.config import settings


def create_redis_client(url: str | None = None, **kwargs: Any) -> aioredis.Redis:
    kwargs.setdefault("decode_responses", True)
    return aioredis.from_url(url or settings.REDIS_URL, **kwargs)


class RedisNotInitialized(RuntimeError):
    """Raised when Redis is accessed before the lifespan created the client."""


def get_redis_from_app(app: Starlette) -> aioredis.Redis:
    redis_client = getattr(app.state, "redis", None)
    if redis_client is None:
        raise RedisNotInitialized("redis client not initialized")
    return redis_client  # type: ignore[no-any-return]


def get_redis(request: Request) -> aioredis.Redis:
    try:
        return get_redis_from_app(request.app)
    except RedisNotInitialized:
        raise HTTPException(status_code=503, detail="redis client not initialized")

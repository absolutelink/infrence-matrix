"""Stable PostgreSQL advisory lock keys used by scheduler coordination."""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

logger = logging.getLogger(__name__)

BENCHMARK_ADVISORY_LOCK_KEY = 0x494E464552454E43
TOKEN_STATS_PRUNE_LOCK_KEY = 0x544F4B454E505255


async def release_session_advisory_lock(connection: AsyncConnection, key: int) -> None:
    """Unlock ``key``, commit, and close ``connection`` without stranding the lock.

    Session-level advisory locks survive transaction rollback, so a caller that
    is cancelled after acquiring one can return the pooled connection to the
    pool with the lock still held. The unlock/commit/close runs as an
    independent task protected by ``asyncio.shield``; ``CancelledError`` is
    swallowed while waiting so the release always finishes, then re-raised to
    preserve the caller's cancellation semantics. Unlocking a key this
    session never held is a safe no-op, so callers may invoke this
    unconditionally from a ``finally`` block.
    """

    async def _release() -> None:
        try:
            await connection.execute(
                text("SELECT pg_advisory_unlock(:key)"), {"key": key}
            )
            await connection.commit()
        finally:
            await connection.close()

    release = asyncio.create_task(_release())
    cancelled = False
    while True:
        try:
            await asyncio.shield(release)
            break
        except asyncio.CancelledError:
            cancelled = True
            if release.done():
                break
        except Exception:
            logger.warning(
                "Advisory lock release for key %s failed", key, exc_info=True
            )
            break
    if cancelled:
        if not release.cancelled() and (exc := release.exception()) is not None:
            logger.warning("Advisory lock release for key %s failed", key, exc_info=exc)
        raise asyncio.CancelledError

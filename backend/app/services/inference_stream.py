"""Bounded, client-visible waiting for an agent's first inference frame."""

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from app.services.inference_scheduler import InferenceLeaseHandle

FIRST_FRAME_TIMEOUT_SECONDS = 150.0
KEEPALIVE_SECONDS = 10.0
logger = logging.getLogger(__name__)


async def lines_with_keepalive(
    lines: AsyncIterator[str],
    lease: InferenceLeaseHandle,
    *,
    deadline: float | None = None,
) -> AsyncGenerator[str | None]:
    """Yield None for SSE comments, but never let keepalives extend dispatch time."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    if deadline is None:
        deadline = started + FIRST_FRAME_TIMEOUT_SECONDS
    first_frame = False
    pending: asyncio.Task[str] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(lease.guard(anext(lines)))
            remaining = deadline - loop.time()
            if not first_frame and remaining <= 0:
                raise TimeoutError(
                    "LLM did not produce a frame before the dispatch deadline"
                )
            done, _ = await asyncio.wait(
                {pending},
                timeout=KEEPALIVE_SECONDS
                if first_frame
                else min(KEEPALIVE_SECONDS, remaining),
            )
            if not done:
                yield None
                continue
            try:
                line = await pending
            except StopAsyncIteration:
                return
            finally:
                pending = None
            if line.startswith("data:"):
                if not first_frame:
                    logger.info(
                        "inference_first_frame request_id=%s server_id=%s delay_ms=%.1f",
                        lease.request_id,
                        getattr(lease.server, "id", "-"),
                        (loop.time() - (deadline - FIRST_FRAME_TIMEOUT_SECONDS)) * 1000,
                    )
                first_frame = True
            yield line
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


async def upstream_with_keepalive(
    client: Any,
    url: str,
    payload: dict,
    lease: InferenceLeaseHandle,
) -> AsyncGenerator[tuple[Any, float] | None]:
    """Keep the downstream alive while connecting, under the same first-frame deadline."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + FIRST_FRAME_TIMEOUT_SECONDS
    opened: asyncio.Future[Any] = loop.create_future()
    close = asyncio.Event()

    async def hold_open() -> None:
        async with client.stream(
            "POST", url, json=payload, headers=lease.dispatch_headers()
        ) as response:
            opened.set_result(response)
            await close.wait()

    worker = asyncio.create_task(lease.guard(hold_open()))
    try:
        while not opened.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(
                    "Agent did not open a stream before the dispatch deadline"
                )
            done, _ = await asyncio.wait(
                {opened, worker},
                timeout=min(KEEPALIVE_SECONDS, remaining),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if worker in done:
                await worker
                if not opened.done():
                    raise RuntimeError("Agent stream closed before sending headers")
            if not opened.done():
                yield None
        yield opened.result(), deadline
    finally:
        close.set()
        if not opened.done() and not worker.done():
            worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

"""Best-effort recording of per-request token usage samples.

Every inference path (responses HTTP/WS, chat completions, completions,
embeddings) calls :func:`record_usage` once the upstream usage and
llama.cpp timings are parsed. Failures are swallowed: statistics must
never break inference.
"""

import asyncio
import logging
from typing import Any

from sqlalchemy import text
from sqlmodel import Session

from app.core.db import engine
from app.db.session import engine as async_engine
from app.models import ServerInstance, TokenUsageSample

logger = logging.getLogger(__name__)


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except TypeError, ValueError:
        return 0.0
    return result if result > 0 else 0.0


def _as_int(value: Any) -> int:
    try:
        result = int(value)
    except TypeError, ValueError:
        return 0
    return result if result > 0 else 0


def extract_usage_fields(
    usage: dict[str, Any] | None, timings: dict[str, Any] | None
) -> dict[str, Any]:
    """Normalize OpenAI-style and spec-style usage dicts into sample fields.

    Accepts both ``prompt_tokens``/``completion_tokens`` (OpenAI) and
    ``input_tokens``/``output_tokens`` (OpenResponses spec), plus cached
    tokens from ``prompt_tokens_details``/``input_tokens_details`` or the
    llama.cpp ``timings.cache_n`` fallback.
    """
    usage = usage or {}
    timings = timings or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    input_details = usage.get("input_tokens_details") or {}
    prompt_tokens = _as_int(usage.get("prompt_tokens", usage.get("input_tokens")))
    completion_tokens = _as_int(
        usage.get("completion_tokens", usage.get("output_tokens"))
    )
    prompt_ms = _as_float(timings.get("prompt_ms") or usage.get("prompt_ms"))
    predicted_ms = _as_float(timings.get("predicted_ms") or usage.get("predicted_ms"))
    # Some engines report only rates; derive durations so window math works.
    if not prompt_ms and prompt_tokens:
        rate = _as_float(
            timings.get("prompt_per_second") or usage.get("prompt_per_second")
        )
        if rate:
            prompt_ms = prompt_tokens / rate * 1000.0
    if not predicted_ms and completion_tokens:
        rate = _as_float(
            timings.get("predicted_per_second") or usage.get("predicted_per_second")
        )
        if rate:
            predicted_ms = completion_tokens / rate * 1000.0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": _as_int(
            prompt_details.get("cached_tokens")
            or input_details.get("cached_tokens")
            or timings.get("cache_n")
        ),
        "prompt_ms": prompt_ms,
        "predicted_ms": predicted_ms,
    }


def record_usage(
    server: ServerInstance | None,
    usage: dict[str, Any] | None,
    timings: dict[str, Any] | None = None,
) -> None:
    """Persist one token usage sample. Never raises."""
    try:
        fields = extract_usage_fields(usage, timings)
        if (
            not fields["prompt_tokens"]
            and not fields["completion_tokens"]
            and not fields["prompt_ms"]
            and not fields["predicted_ms"]
        ):
            return
        with Session(engine) as session:
            session.add(
                TokenUsageSample(
                    server_instance_id=server.id if server is not None else None,
                    agent_id=server.agent_id if server is not None else None,
                    model_id=server.model_id if server is not None else None,
                    **fields,
                )
            )
            session.commit()
    except Exception:
        logger.warning("Failed to record token usage sample", exc_info=True)


# Retention: keep ~35 days so the 30-day window always has full coverage.
TOKEN_SAMPLE_RETENTION_DAYS = 35
PRUNE_INTERVAL_SECONDS = 6 * 60 * 60


async def prune_old_samples() -> int:
    """Delete samples past the retention window. One worker at a time."""
    from app.services.scheduler_locks import TOKEN_STATS_PRUNE_LOCK_KEY

    async with async_engine.connect() as connection:
        locked = (
            await connection.execute(
                text("SELECT pg_try_advisory_lock(:key)"),
                {"key": TOKEN_STATS_PRUNE_LOCK_KEY},
            )
        ).scalar_one()
        if not locked:
            return 0
        try:
            result = await connection.execute(
                text(
                    "DELETE FROM token_usage_samples "
                    "WHERE created_at < now() - "
                    "make_interval(days => :days)"
                ),
                {"days": TOKEN_SAMPLE_RETENTION_DAYS},
            )
            await connection.commit()
            return int(result.rowcount or 0)
        finally:
            await connection.execute(
                text("SELECT pg_advisory_unlock(:key)"),
                {"key": TOKEN_STATS_PRUNE_LOCK_KEY},
            )
            await connection.commit()


async def token_stats_prune_loop() -> None:
    while True:
        await asyncio.sleep(PRUNE_INTERVAL_SECONDS)
        try:
            removed = await prune_old_samples()
            if removed:
                logger.info("Pruned %d expired token usage sample(s)", removed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Token usage sample pruning failed", exc_info=True)

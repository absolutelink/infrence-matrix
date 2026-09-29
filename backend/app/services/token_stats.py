"""Best-effort recording of per-request token usage samples.

Every inference path (responses HTTP/WS, chat completions, completions,
embeddings) calls :func:`record_usage` once the upstream usage and
llama.cpp timings are parsed. Failures are swallowed: statistics must
never break inference.
"""

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

from sqlalchemy import text
from sqlmodel import Session

from app.core.db import engine
from app.db.session import AsyncSessionMaker
from app.db.session import engine as async_engine
from app.models import ServerInstance, TokenUsageSample

logger = logging.getLogger(__name__)
_telemetry_tasks: set[asyncio.Task[None]] = set()


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
    # Engine-reported rates, prefer them over recomputation so live figures
    # match llama-server's own logs (which count cached prompt tokens in
    # the prompt rate and cover the whole stage). Derive only when missing.
    prompt_per_second = _as_float(
        timings.get("prompt_per_second") or usage.get("prompt_per_second")
    )
    predicted_per_second = _as_float(
        timings.get("predicted_per_second") or usage.get("predicted_per_second")
    )
    # Some engines report only rates; derive durations so window math works.
    if not prompt_ms and prompt_per_second:
        prompt_ms = prompt_tokens / prompt_per_second * 1000.0
    if not predicted_ms and predicted_per_second:
        predicted_ms = completion_tokens / predicted_per_second * 1000.0
    if not prompt_per_second and prompt_ms and prompt_tokens:
        prompt_per_second = prompt_tokens / (prompt_ms / 1000.0)
    if not predicted_per_second and predicted_ms and completion_tokens:
        predicted_per_second = completion_tokens / (predicted_ms / 1000.0)
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
        "prompt_per_second": prompt_per_second,
        "predicted_per_second": predicted_per_second,
    }


def record_usage(
    server: ServerInstance | None,
    usage: dict[str, Any] | None,
    timings: dict[str, Any] | None = None,
    request_id: str | None = None,
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
        logger.warning(
            "Failed to record token usage sample request_id=%s",
            request_id or "-",
            exc_info=True,
        )


def record_request_telemetry(
    server: ServerInstance | None,
    usage: dict[str, Any] | None,
    timings: dict[str, Any] | None = None,
    latency_ms: float = 0.0,
    status: str = "success",
    model_name: str | None = None,
    agent_id: str | None = None,
    request_id: str | None = None,
) -> None:
    """Record one finished inference request: usage sample, Prometheus
    metrics, and lifetime counters on the serving ServerInstance.

    ``latency_ms`` is wall-clock request latency; ``status`` is
    ``success``/``error``/``cancelled``. Database persistence runs off the
    request event loop so a telemetry insert waiting on a server-row lock
    cannot hold up stream completion or lease release. Best-effort: never raises.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _record_request_telemetry_sync(
            server, usage, timings, latency_ms, status, model_name, agent_id, request_id
        )
        return

    server_snapshot = None
    if server is not None:
        server_snapshot = SimpleNamespace(
            id=getattr(server, "id", None),
            agent_id=getattr(server, "agent_id", None),
            model_id=getattr(server, "model_id", None),
            alias=getattr(server, "alias", None),
        )
    task = loop.create_task(
        asyncio.to_thread(
            _record_request_telemetry_sync,
            server_snapshot,
            dict(usage) if usage is not None else None,
            dict(timings) if timings is not None else None,
            latency_ms,
            status,
            model_name,
            agent_id,
            request_id,
        )
    )
    _telemetry_tasks.add(task)
    task.add_done_callback(lambda finished: _telemetry_task_done(finished, request_id))


def _telemetry_task_done(task: asyncio.Task[None], request_id: str | None) -> None:
    _telemetry_tasks.discard(task)
    if task.cancelled():
        return
    try:
        task.result()
    except Exception:
        logger.warning(
            "Background request telemetry task failed request_id=%s",
            request_id or "-",
            exc_info=True,
        )


def _record_request_telemetry_sync(
    server: ServerInstance | None,
    usage: dict[str, Any] | None,
    timings: dict[str, Any] | None,
    latency_ms: float,
    status: str,
    model_name: str | None,
    agent_id: str | None,
    request_id: str | None = None,
) -> None:
    record_usage(server, usage, timings, request_id)
    try:
        fields = extract_usage_fields(usage, timings)
        total_tokens = fields["prompt_tokens"] + fields["completion_tokens"]

        resolved_model = model_name
        resolved_agent = agent_id
        if server is not None:
            if resolved_model is None:
                resolved_model = getattr(server, "alias", None) or str(
                    getattr(server, "model_id", "unknown")
                )
            if resolved_agent is None:
                resolved_agent = str(getattr(server, "agent_id", "unknown"))
        resolved_model = resolved_model or "unknown"
        resolved_agent = resolved_agent or "unknown"
    except Exception:
        logger.warning(
            "Failed to extract telemetry fields request_id=%s",
            request_id or "-",
            exc_info=True,
        )
        return

    try:
        from app.api.routes.metrics import record_inference_request

        record_inference_request(
            model=resolved_model,
            agent_id=resolved_agent,
            status=status,
            latency=max(latency_ms, 0.0) / 1000.0,
            tokens=total_tokens if status == "success" else 0,
        )
    except Exception:
        logger.warning(
            "Failed to record inference metrics request_id=%s",
            request_id or "-",
            exc_info=True,
        )

    if server is None or getattr(server, "id", None) is None or status != "success":
        return
    try:
        # SQL-side increments avoid read-modify-write races between
        # concurrent requests on the same server row.
        with Session(engine) as session:
            session.execute(
                text(
                    """
                    UPDATE server_instances
                    SET total_requests = total_requests + 1,
                        total_tokens_generated = total_tokens_generated + :tokens,
                        average_response_time_ms =
                            (average_response_time_ms * total_requests
                             + :latency_ms) / (total_requests + 1)
                    WHERE id = :server_id
                    """
                ),
                {
                    "tokens": total_tokens,
                    "latency_ms": max(latency_ms, 0.0),
                    "server_id": server.id,
                },
            )
            session.commit()
    except Exception:
        logger.warning(
            "Failed to update server usage counters request_id=%s",
            request_id or "-",
            exc_info=True,
        )


# Retention: keep ~35 days so the 30-day window always has full coverage.
TOKEN_SAMPLE_RETENTION_DAYS = 35
PRUNE_INTERVAL_SECONDS = 6 * 60 * 60
LIVE_RUNS = 10


async def token_stats_snapshot() -> dict[str, Any]:
    """Aggregate token usage for the status bar and popover.

    Live rates are the mean of each server's last ``LIVE_RUNS`` completed
    requests as reported by the engine: decode = ``predicted_per_second``,
    prefill = ``prompt_per_second``. These are taken directly from
    llama.cpp's timings rather than recomputed from token counts and
    durations (which undercount prefill because ``prompt_ms`` covers only
    uncached work while the engine's rate covers the whole stage).
    """
    async with AsyncSessionMaker() as session:
        rows = (
            (
                await session.execute(
                    text(
                        """
                        SELECT
                            s.id AS server_id,
                            s.alias AS alias,
                            coalesce(sum(t.prompt_tokens - t.cached_tokens) FILTER (
                                WHERE t.created_at >= now() - make_interval(hours => 24)
                            ), 0) AS prompt_24h,
                            coalesce(sum(t.completion_tokens) FILTER (
                                WHERE t.created_at >= now() - make_interval(hours => 24)
                            ), 0) AS completion_24h,
                            coalesce(sum(t.prompt_tokens - t.cached_tokens) FILTER (
                                WHERE t.created_at >= now() - make_interval(days => 7)
                            ), 0) AS prompt_7d,
                            coalesce(sum(t.completion_tokens) FILTER (
                                WHERE t.created_at >= now() - make_interval(days => 7)
                            ), 0) AS completion_7d,
                            coalesce(sum(t.prompt_tokens - t.cached_tokens) FILTER (
                                WHERE t.created_at >= now() - make_interval(days => 30)
                            ), 0) AS prompt_30d,
                            coalesce(sum(t.completion_tokens) FILTER (
                                WHERE t.created_at >= now() - make_interval(days => 30)
                            ), 0) AS completion_30d
                        FROM token_usage_samples t
                        LEFT JOIN server_instances s ON s.id = t.server_instance_id
                        WHERE t.created_at >= now() - make_interval(days => 30)
                        GROUP BY s.id, s.alias
                        """
                    ),
                )
            )
            .mappings()
            .all()
        )

        live_rows = (
            (
                await session.execute(
                    text(
                        """
                        WITH ranked AS (
                            SELECT
                                t.server_instance_id,
                                CASE WHEN t.predicted_per_second > 0
                                    THEN t.predicted_per_second
                                END AS decode_tps,
                                CASE WHEN t.prompt_per_second > 0
                                    THEN t.prompt_per_second
                                END AS prefill_tps,
                                row_number() OVER (
                                    PARTITION BY t.server_instance_id
                                    ORDER BY t.created_at DESC, t.id DESC
                                ) AS rn
                            FROM token_usage_samples t
                            WHERE t.created_at >= now() - make_interval(days => 30)
                        )
                        SELECT
                            server_instance_id AS server_id,
                            avg(decode_tps) AS decode_tps,
                            count(decode_tps) AS decode_runs,
                            avg(prefill_tps) AS prefill_tps,
                            count(prefill_tps) AS prefill_runs
                        FROM ranked
                        WHERE rn <= :live_runs
                        GROUP BY server_instance_id
                        """
                    ),
                    {"live_runs": LIVE_RUNS},
                )
            )
            .mappings()
            .all()
        )

    live_by_server: dict[Any, dict[str, Any]] = {
        row["server_id"]: row for row in live_rows
    }

    window_totals: dict[str, dict[str, int]] = {
        "last_24h": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "last_7d": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "last_30d": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    servers: list[dict[str, Any]] = []

    for row in rows:
        prompt_24h = int(row["prompt_24h"])
        completion_24h = int(row["completion_24h"])
        prompt_7d = int(row["prompt_7d"])
        completion_7d = int(row["completion_7d"])
        prompt_30d = int(row["prompt_30d"])
        completion_30d = int(row["completion_30d"])
        window_totals["last_24h"]["prompt_tokens"] += prompt_24h
        window_totals["last_24h"]["completion_tokens"] += completion_24h
        window_totals["last_24h"]["total_tokens"] += prompt_24h + completion_24h
        window_totals["last_7d"]["prompt_tokens"] += prompt_7d
        window_totals["last_7d"]["completion_tokens"] += completion_7d
        window_totals["last_7d"]["total_tokens"] += prompt_7d + completion_7d
        window_totals["last_30d"]["prompt_tokens"] += prompt_30d
        window_totals["last_30d"]["completion_tokens"] += completion_30d
        window_totals["last_30d"]["total_tokens"] += prompt_30d + completion_30d

        if row["server_id"] is None:
            continue
        live = live_by_server.get(row["server_id"])
        servers.append(
            {
                "id": str(row["server_id"]),
                "alias": row["alias"] or "unknown",
                "decode_tokens_per_second": (
                    float(live["decode_tps"]) if live and live["decode_tps"] else None
                ),
                "prefill_tokens_per_second": (
                    float(live["prefill_tps"]) if live and live["prefill_tps"] else None
                ),
                "last_7d": {
                    "prompt_tokens": prompt_7d,
                    "completion_tokens": completion_7d,
                    "total_tokens": prompt_7d + completion_7d,
                },
                "last_30d": {
                    "prompt_tokens": prompt_30d,
                    "completion_tokens": completion_30d,
                    "total_tokens": prompt_30d + completion_30d,
                },
            }
        )

    # Global live rates: run-weighted mean of per-server averages over
    # each server's last LIVE_RUNS runs.
    decode_weighted = 0.0
    decode_runs = 0
    prefill_weighted = 0.0
    prefill_runs = 0
    for live in live_by_server.values():
        if live["decode_tps"] is not None and live["decode_runs"]:
            decode_weighted += float(live["decode_tps"]) * int(live["decode_runs"])
            decode_runs += int(live["decode_runs"])
        if live["prefill_tps"] is not None and live["prefill_runs"]:
            prefill_weighted += float(live["prefill_tps"]) * int(live["prefill_runs"])
            prefill_runs += int(live["prefill_runs"])

    servers.sort(
        key=lambda item: (
            item["decode_tokens_per_second"] or 0.0,
            item["prefill_tokens_per_second"] or 0.0,
            item["last_30d"]["total_tokens"],
        ),
        reverse=True,
    )

    return {
        "global": {
            "live": {
                "runs": LIVE_RUNS,
                "decode_tokens_per_second": (
                    decode_weighted / decode_runs if decode_runs else None
                ),
                "prefill_tokens_per_second": (
                    prefill_weighted / prefill_runs if prefill_runs else None
                ),
            },
            "last_24h": window_totals["last_24h"],
            "last_7d": window_totals["last_7d"],
            "last_30d": window_totals["last_30d"],
        },
        "servers": servers,
    }


async def prune_old_samples() -> int:
    """Delete samples past the retention window. One worker at a time."""
    from app.services.scheduler_locks import (
        TOKEN_STATS_PRUNE_LOCK_KEY,
        release_session_advisory_lock,
    )

    async with async_engine.connect() as connection:
        # The try begins before the lock can be observed as held so no
        # cancellation window can leave the session-level advisory lock
        # attached to the pooled connection. The shielded unlock in the
        # finally is a safe no-op when the lock was never acquired.
        try:
            locked = (
                await connection.execute(
                    text("SELECT pg_try_advisory_lock(:key)"),
                    {"key": TOKEN_STATS_PRUNE_LOCK_KEY},
                )
            ).scalar_one()
            await connection.commit()
            if not locked:
                return 0
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
            await release_session_advisory_lock(connection, TOKEN_STATS_PRUNE_LOCK_KEY)


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

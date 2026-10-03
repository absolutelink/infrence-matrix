"""Prometheus metrics endpoint."""

import asyncio
import logging

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

router = APIRouter(tags=["metrics"])

# Metrics definitions
AGENT_COUNT = Gauge(
    "inference_matrix_agents_total", "Total number of registered agents", ["status"]
)

SERVER_COUNT = Gauge(
    "inference_matrix_servers_total",
    "Total number of running servers",
    ["agent_id", "status"],
)

INFERENCE_REQUESTS = Counter(
    "inference_matrix_inference_requests_total",
    "Total number of inference requests",
    ["model", "agent_id", "status"],
)

INFERENCE_LATENCY = Histogram(
    "inference_matrix_inference_latency_seconds",
    "Inference request latency",
    ["model", "agent_id"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float("inf")),
)

TOKENS_GENERATED = Counter(
    "inference_matrix_tokens_generated_total",
    "Total tokens generated",
    ["model", "agent_id"],
)

VRAM_USAGE = Gauge(
    "inference_matrix_vram_usage_bytes", "VRAM usage in bytes", ["agent_id", "gpu_id"]
)

MODEL_CACHE_SIZE = Gauge(
    "inference_matrix_cache_size_bytes",
    "Prompt cache size in bytes",
    ["model", "agent_id"],
)


@router.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


logger = logging.getLogger(__name__)

METRICS_SNAPSHOT_INTERVAL_SECONDS = 15


def update_agent_metrics(agents: list) -> None:
    """Update agent-related metrics.

    Full snapshot: clears stale label sets first so agents that went away
    or changed status do not leave frozen series behind.
    """
    status_counts: dict[str, int] = {}
    for agent in agents:
        status = agent.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    AGENT_COUNT.clear()
    for status, count in status_counts.items():
        AGENT_COUNT.labels(status=status).set(count)


def update_server_metrics(servers: list) -> None:
    """Update server-related metrics.

    Full snapshot: counts servers per (agent, status) and clears stale
    label sets from deleted servers or previous statuses.
    """
    counts: dict[tuple[str, str], int] = {}
    for server in servers:
        key = (
            str(server.get("agent_id", "unknown")),
            server.get("status", "unknown"),
        )
        counts[key] = counts.get(key, 0) + 1

    SERVER_COUNT.clear()
    for (agent_id, status), count in counts.items():
        SERVER_COUNT.labels(agent_id=agent_id, status=status).set(count)


def record_inference_request(
    model: str, agent_id: str, status: str, latency: float, tokens: int
) -> None:
    """Record inference request metrics."""
    INFERENCE_REQUESTS.labels(model=model, agent_id=agent_id, status=status).inc()

    INFERENCE_LATENCY.labels(model=model, agent_id=agent_id).observe(latency)

    if tokens > 0:
        TOKENS_GENERATED.labels(model=model, agent_id=agent_id).inc(tokens)


def update_vram_metrics(agent_id: str, gpu_id: int, vram_bytes: int) -> None:
    """Update VRAM usage metrics."""
    VRAM_USAGE.labels(agent_id=agent_id, gpu_id=str(gpu_id)).set(vram_bytes)


def snapshot_from_db() -> None:
    """Refresh agent/server gauges from the database."""
    from sqlmodel import Session, select

    from app.core.db import engine
    from app.models import Agent, ServerInstance

    with Session(engine) as session:
        agents = [
            {"status": agent.status} for agent in session.exec(select(Agent)).all()
        ]
        servers = [
            {"agent_id": str(server.agent_id), "status": server.status}
            for server in session.exec(select(ServerInstance)).all()
        ]
    update_agent_metrics(agents)
    update_server_metrics(servers)


async def metrics_snapshot_loop() -> None:
    """Periodically refresh database-backed gauges."""
    while True:
        try:
            await asyncio.to_thread(snapshot_from_db)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Metrics snapshot failed", exc_info=True)
        await asyncio.sleep(METRICS_SNAPSHOT_INTERVAL_SECONDS)

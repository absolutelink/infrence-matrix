"""Prometheus metrics endpoint."""

from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Gauge, Counter, Histogram

router = APIRouter(tags=["metrics"])

# Metrics definitions
AGENT_COUNT = Gauge(
    "inference_matrix_agents_total",
    "Total number of registered agents",
    ["status"]
)

SERVER_COUNT = Gauge(
    "inference_matrix_servers_total",
    "Total number of running servers",
    ["agent_id", "status"]
)

INFERENCE_REQUESTS = Counter(
    "inference_matrix_inference_requests_total",
    "Total number of inference requests",
    ["model", "agent_id", "status"]
)

INFERENCE_LATENCY = Histogram(
    "inference_matrix_inference_latency_seconds",
    "Inference request latency",
    ["model", "agent_id"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float("inf"))
)

TOKENS_GENERATED = Counter(
    "inference_matrix_tokens_generated_total",
    "Total tokens generated",
    ["model", "agent_id"]
)

VRAM_USAGE = Gauge(
    "inference_matrix_vram_usage_bytes",
    "VRAM usage in bytes",
    ["agent_id", "gpu_id"]
)

MODEL_CACHE_SIZE = Gauge(
    "inference_matrix_cache_size_bytes",
    "Prompt cache size in bytes",
    ["model", "agent_id"]
)


@router.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


def update_agent_metrics(agents: list) -> None:
    """Update agent-related metrics."""
    # Count agents by status
    status_counts = {}
    for agent in agents:
        status = agent.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    for status, count in status_counts.items():
        AGENT_COUNT.labels(status=status).set(count)


def update_server_metrics(servers: list) -> None:
    """Update server-related metrics."""
    for server in servers:
        SERVER_COUNT.labels(
            agent_id=server.get("agent_id", "unknown"),
            status=server.get("status", "unknown")
        ).set(1)


def record_inference_request(
    model: str,
    agent_id: str,
    status: str,
    latency: float,
    tokens: int
) -> None:
    """Record inference request metrics."""
    INFERENCE_REQUESTS.labels(
        model=model,
        agent_id=agent_id,
        status=status
    ).inc()
    
    INFERENCE_LATENCY.labels(
        model=model,
        agent_id=agent_id
    ).observe(latency)
    
    if tokens > 0:
        TOKENS_GENERATED.labels(
            model=model,
            agent_id=agent_id
        ).inc(tokens)


def update_vram_metrics(agent_id: str, gpu_id: int, vram_bytes: int) -> None:
    """Update VRAM usage metrics."""
    VRAM_USAGE.labels(
        agent_id=agent_id,
        gpu_id=str(gpu_id)
    ).set(vram_bytes)

"""Tests for Prometheus metric updaters and snapshot loops."""

import uuid

from prometheus_client import REGISTRY

from app.api.routes.metrics import (
    snapshot_from_db,
    update_agent_metrics,
    update_server_metrics,
    update_vram_metrics,
)


def test_update_agent_metrics_clears_stale_status_labels() -> None:
    update_agent_metrics([{"status": "online"}, {"status": "online"}])
    online = REGISTRY.get_sample_value(
        "inference_matrix_agents_total", {"status": "online"}
    )
    assert online == 2

    update_agent_metrics([{"status": "offline"}])
    online = REGISTRY.get_sample_value(
        "inference_matrix_agents_total", {"status": "online"}
    )
    offline = REGISTRY.get_sample_value(
        "inference_matrix_agents_total", {"status": "offline"}
    )
    # Cleared, not left frozen at the previous value.
    assert online is None
    assert offline == 1


def test_update_server_metrics_counts_per_agent_status() -> None:
    agent_a = str(uuid.uuid4())
    agent_b = str(uuid.uuid4())
    update_server_metrics(
        [
            {"agent_id": agent_a, "status": "running"},
            {"agent_id": agent_a, "status": "running"},
            {"agent_id": agent_b, "status": "stopped"},
        ]
    )
    assert (
        REGISTRY.get_sample_value(
            "inference_matrix_servers_total",
            {"agent_id": agent_a, "status": "running"},
        )
        == 2
    )
    assert (
        REGISTRY.get_sample_value(
            "inference_matrix_servers_total",
            {"agent_id": agent_b, "status": "stopped"},
        )
        == 1
    )

    # Snapshot semantics: the previous label set disappears entirely.
    update_server_metrics([{"agent_id": agent_b, "status": "error"}])
    assert (
        REGISTRY.get_sample_value(
            "inference_matrix_servers_total",
            {"agent_id": agent_a, "status": "running"},
        )
        is None
    )
    assert (
        REGISTRY.get_sample_value(
            "inference_matrix_servers_total",
            {"agent_id": agent_b, "status": "error"},
        )
        == 1
    )


def test_update_vram_metrics_sets_gauge() -> None:
    agent_id = str(uuid.uuid4())
    update_vram_metrics(agent_id, 1, 1234)
    assert (
        REGISTRY.get_sample_value(
            "inference_matrix_vram_usage_bytes",
            {"agent_id": agent_id, "gpu_id": "1"},
        )
        == 1234
    )


def test_snapshot_from_db_reads_agents_and_servers() -> None:
    from sqlmodel import Session

    from app.core.db import engine
    from app.models import Agent, Model, ServerInstance

    agent = Agent(
        name=f"metrics-agent-{uuid.uuid4().hex}",
        host="127.0.0.1",
        port=8080,
        status="online",
    )
    with Session(engine) as session:
        session.add(agent)
        session.commit()
        session.refresh(agent)
        agent_id = str(agent.id)
        model = Model(
            name=f"metrics-model-{uuid.uuid4().hex}",
            path="/models/metrics.gguf",
            size_bytes=1_000,
            architecture="llama",
            quantization="Q4_K_M",
            source="huggingface",
        )
        session.add(model)
        session.commit()
        session.refresh(model)
        instance = ServerInstance(
            model_id=model.id,
            agent_id=agent.id,
            alias=f"mx-{uuid.uuid4().hex[:6]}",
            process_command="llama-server",
            status="running",
            health_status="healthy",
        )
        session.add(instance)
        session.commit()

    snapshot_from_db()

    assert (
        REGISTRY.get_sample_value(
            "inference_matrix_servers_total",
            {"agent_id": agent_id, "status": "running"},
        )
        == 1
    )
    assert (
        REGISTRY.get_sample_value("inference_matrix_agents_total", {"status": "online"})
        >= 1
    )

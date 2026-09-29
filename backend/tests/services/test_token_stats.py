"""Tests for token usage sample recording and retention."""

import asyncio
import threading
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import text
from sqlmodel import Session, select

from app.core.config import settings
from app.core.db import engine
from app.db.session import AsyncSessionMaker
from app.db.session import engine as async_engine
from app.models import Agent, Model, ServerInstance, TokenUsageSample
from app.services.inference_scheduler import InferenceLeaseHandle
from app.services.token_stats import (
    TOKEN_SAMPLE_RETENTION_DAYS,
    extract_usage_fields,
    parse_llama_metrics,
    prune_old_samples,
    record_request_telemetry,
    record_usage,
    token_stats_snapshot,
)


def test_parse_llama_metrics_extracts_engine_reported_rates() -> None:
    metrics = """# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed
llamacpp:prompt_tokens_total 29701
llamacpp:prompt_seconds_total 19.1784
llamacpp:tokens_predicted_total 983
llamacpp:tokens_predicted_seconds_total 20.54
"""

    assert parse_llama_metrics(metrics) == {
        "prompt_per_second": pytest.approx(29701 / 19.1784),
        "predicted_per_second": pytest.approx(983 / 20.54),
    }


def test_parse_llama_metrics_ignores_zero_duration_ratios() -> None:
    metrics = """llamacpp:prompt_tokens_total 200
llamacpp:prompt_seconds_total 0
llamacpp:tokens_predicted_total 10
llamacpp:tokens_predicted_seconds_total 0
"""

    assert parse_llama_metrics(metrics) == {}


@pytest.mark.asyncio
async def test_async_request_telemetry_does_not_block_event_loop(monkeypatch):
    from app.services import token_stats

    started = threading.Event()
    unblock = threading.Event()
    finished = threading.Event()

    def blocked_persistence(*_args) -> None:
        started.set()
        unblock.wait(timeout=2.0)
        finished.set()

    monkeypatch.setattr(
        token_stats, "_record_request_telemetry_sync", blocked_persistence
    )
    loop = asyncio.get_running_loop()
    loop_tick = asyncio.Event()
    loop.call_later(0.02, loop_tick.set)
    release_timer = threading.Timer(0.2, unblock.set)
    release_timer.start()

    started_at = loop.time()
    record_request_telemetry(None, {"prompt_tokens": 1}, status="success")
    await asyncio.wait_for(loop_tick.wait(), timeout=0.1)
    elapsed = loop.time() - started_at
    await asyncio.to_thread(finished.wait, 1.0)
    release_timer.cancel()

    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_completed_request_scrapes_agent_metrics_once(monkeypatch):
    from app.services import agent_manager, token_stats

    persisted: dict[str, Any] = {}

    def capture_persistence(_server, _usage, timings, *_args) -> None:
        persisted.update(timings or {})

    scrape = AsyncMock(
        return_value={
            "metrics": "llamacpp:prompt_tokens_total 29701\n"
            "llamacpp:prompt_seconds_total 19.1784\n"
            "llamacpp:tokens_predicted_total 983\n"
            "llamacpp:tokens_predicted_seconds_total 20.54\n"
        }
    )
    monkeypatch.setattr(agent_manager.agent_manager, "send_to_agent", scrape)
    monkeypatch.setattr(
        token_stats, "_record_request_telemetry_sync", capture_persistence
    )

    await token_stats._record_request_telemetry_after_scrape(
        SimpleNamespace(id="server-1", agent_id="agent-1"),
        {"prompt_tokens": 10, "completion_tokens": 2},
        {"prompt_ms": 100, "predicted_ms": 50},
        150,
        "success",
        None,
        None,
        "request-1",
    )

    scrape.assert_awaited_once_with(
        "agent-1", "GET", "/proxy/server-1/metrics", timeout=5.0
    )
    assert persisted["prompt_per_second"] == pytest.approx(29701 / 19.1784)
    assert persisted["predicted_per_second"] == pytest.approx(983 / 20.54)
    assert persisted["prompt_ms"] == 100
    assert persisted["predicted_ms"] == 50


@pytest.mark.asyncio
async def test_blocked_telemetry_persistence_does_not_delay_lease_release(monkeypatch):
    from app.services import token_stats

    started = threading.Event()
    unblock = threading.Event()

    def blocked_write(*_args):
        started.set()
        unblock.wait(timeout=3)

    monkeypatch.setattr(token_stats, "_record_request_telemetry_sync", blocked_write)
    session = AsyncMock()
    session.execute.return_value.rowcount = 1
    context = AsyncMock()
    context.__aenter__.return_value = session
    handle = InferenceLeaseHandle(
        "telemetry-blocked-id", SimpleNamespace(), uuid.uuid4()
    )
    try:
        record_request_telemetry(
            None, {"prompt_tokens": 1}, request_id=handle.request_id
        )
        assert await asyncio.to_thread(started.wait, 1)
        with (
            patch(
                "app.services.inference_scheduler.AsyncSessionMaker",
                return_value=context,
            ),
            patch(
                "app.services.inference_scheduler.UPSTREAM_COMPLETION_COOLDOWN_SECONDS",
                0,
            ),
        ):
            await asyncio.wait_for(handle.release(), timeout=0.5)
        session.commit.assert_awaited_once()
        assert not unblock.is_set()
    finally:
        unblock.set()


def _make_server(db: Session) -> ServerInstance:
    agent = Agent(
        name=f"tokstats-agent-{uuid.uuid4().hex}",
        host="127.0.0.1",
        port=8080,
        status="online",
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    model = Model(
        name=f"tokstats-model-{uuid.uuid4().hex}",
        path="/models/tokstats.gguf",
        size_bytes=1_000_000,
        architecture="llama",
        quantization="Q4_K_M",
        source="huggingface",
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    server = ServerInstance(
        model_id=model.id,
        agent_id=agent.id,
        alias=f"tok-{uuid.uuid4().hex[:8]}",
        process_command="llama-server",
        status="running",
        health_status="healthy",
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    return server


@pytest.mark.asyncio
async def test_database_engines_bound_idle_transactions() -> None:
    with Session(engine) as session:
        timeout_ms = session.exec(
            text(
                "SELECT extract(epoch from current_setting("
                "'idle_in_transaction_session_timeout')::interval) * 1000"
            )
        ).one()[0]
    async with AsyncSessionMaker() as session:
        async_timeout_ms = (
            await session.execute(
                text(
                    "SELECT extract(epoch from current_setting("
                    "'idle_in_transaction_session_timeout')::interval) * 1000"
                )
            )
        ).scalar_one()

    assert timeout_ms == pytest.approx(settings.DB_IDLE_TRANSACTION_TIMEOUT_MS)
    assert async_timeout_ms == pytest.approx(settings.DB_IDLE_TRANSACTION_TIMEOUT_MS)


def test_extract_openai_style_usage_with_timings() -> None:
    fields = extract_usage_fields(
        {
            "prompt_tokens": 100,
            "completion_tokens": 42,
            "prompt_tokens_details": {"cached_tokens": 64},
        },
        {"prompt_ms": 500.0, "predicted_ms": 1050.0},
    )
    assert fields == {
        "prompt_tokens": 100,
        "completion_tokens": 42,
        "cached_tokens": 64,
        "prompt_ms": 500.0,
        "predicted_ms": 1050.0,
        "prompt_per_second": 200.0,
        "predicted_per_second": 40.0,
    }


def test_extract_spec_style_usage_and_cache_n_fallback() -> None:
    fields = extract_usage_fields(
        {"input_tokens": 80, "output_tokens": 20},
        {"cache_n": 55},
    )
    assert fields["prompt_tokens"] == 80
    assert fields["completion_tokens"] == 20
    assert fields["cached_tokens"] == 55


def test_extract_derives_ms_from_rates_when_durations_missing() -> None:
    fields = extract_usage_fields(
        {"prompt_tokens": 200, "completion_tokens": 50},
        {"prompt_per_second": 100.0, "predicted_per_second": 25.0},
    )
    assert fields["prompt_ms"] == 2000.0
    assert fields["predicted_ms"] == 2000.0


def test_extract_empty_usage() -> None:
    fields = extract_usage_fields(None, None)
    assert not any(fields.values())


def test_record_usage_persists_sample(db: Session) -> None:
    server = _make_server(db)

    record_usage(
        server,
        {"prompt_tokens": 12, "completion_tokens": 34},
        {"prompt_ms": 30.0, "predicted_ms": 120.0},
    )

    samples = list(
        db.exec(
            select(TokenUsageSample).where(
                TokenUsageSample.server_instance_id == server.id
            )
        )
    )
    assert len(samples) == 1
    assert samples[0].prompt_tokens == 12
    assert samples[0].completion_tokens == 34
    assert samples[0].prompt_ms == 30.0
    assert samples[0].predicted_ms == 120.0
    assert samples[0].agent_id == server.agent_id
    assert samples[0].model_id == server.model_id


def test_record_usage_skips_empty_payloads() -> None:
    before = _sample_count()
    record_usage(None, {}, {})
    record_usage(None, None, None)
    assert _sample_count() == before


def test_record_usage_never_raises_on_garbage_input() -> None:
    before = _sample_count()
    record_usage(None, {"prompt_tokens": "not-a-number"})
    assert _sample_count() == before


def test_record_request_telemetry_increments_server_counters(db: Session) -> None:
    server = _make_server(db)

    record_request_telemetry(
        server,
        {"prompt_tokens": 10, "completion_tokens": 25},
        {"prompt_ms": 40.0, "predicted_ms": 160.0},
        latency_ms=500.0,
        status="success",
    )

    db.expire_all()
    db.refresh(server)
    assert server.total_requests == 1
    assert server.total_tokens_generated == 35
    assert server.average_response_time_ms == pytest.approx(500.0)


def test_record_request_telemetry_running_average(db: Session) -> None:
    server = _make_server(db)

    record_request_telemetry(
        server,
        {"prompt_tokens": 1, "completion_tokens": 1},
        latency_ms=100.0,
        status="success",
    )
    record_request_telemetry(
        server,
        {"prompt_tokens": 1, "completion_tokens": 1},
        latency_ms=300.0,
        status="success",
    )

    db.expire_all()
    db.refresh(server)
    assert server.total_requests == 2
    assert server.total_tokens_generated == 4
    # (100 * 1 + 300) / 2 == 200
    assert server.average_response_time_ms == pytest.approx(200.0)


def test_record_request_telemetry_error_does_not_increment_counters(
    db: Session,
) -> None:
    server = _make_server(db)

    record_request_telemetry(
        server,
        None,
        None,
        latency_ms=250.0,
        status="error",
    )

    db.expire_all()
    db.refresh(server)
    assert server.total_requests == 0
    assert server.total_tokens_generated == 0
    assert server.average_response_time_ms == 0.0


def test_record_request_telemetry_updates_prometheus(db: Session) -> None:
    from prometheus_client import REGISTRY

    server = _make_server(db)
    label = {
        "model": server.alias,
        "agent_id": str(server.agent_id),
        "status": "success",
    }
    before = REGISTRY.get_sample_value(
        "inference_matrix_inference_requests_total", label
    )
    before_tokens = REGISTRY.get_sample_value(
        "inference_matrix_tokens_generated_total",
        {"model": server.alias, "agent_id": str(server.agent_id)},
    )

    record_request_telemetry(
        server,
        {"prompt_tokens": 4, "completion_tokens": 6},
        latency_ms=25.0,
        status="success",
    )

    after = REGISTRY.get_sample_value(
        "inference_matrix_inference_requests_total", label
    )
    after_tokens = REGISTRY.get_sample_value(
        "inference_matrix_tokens_generated_total",
        {"model": server.alias, "agent_id": str(server.agent_id)},
    )
    assert (after or 0) - (before or 0) == 1
    assert (after_tokens or 0) - (before_tokens or 0) == 10


def test_record_request_telemetry_never_raises_without_server() -> None:
    before = _sample_count()
    record_request_telemetry(
        None,
        {"prompt_tokens": 3, "completion_tokens": 3},
        latency_ms=10.0,
        status="success",
        model_name="detached-model",
        agent_id="detached-agent",
    )
    assert _sample_count() == before + 1


async def test_snapshot_aggregates_windows_and_rates(db: Session) -> None:
    server_a = _make_server(db)
    server_b = _make_server(db)
    now = datetime.now(UTC)

    with Session(engine) as session:
        # server_a: fast decode + big prefill, inside the live window.
        # Engine rates cover the whole prompt stage (cached included),
        # so prompt_per_second = prompt_tokens / prompt_ms, not the
        # uncached-only count.
        session.add(
            TokenUsageSample(
                server_instance_id=server_a.id,
                prompt_tokens=1000,
                cached_tokens=400,
                completion_tokens=100,
                prompt_ms=2000.0,
                predicted_ms=2000.0,
                prompt_per_second=500.0,
                predicted_per_second=50.0,
                created_at=now - timedelta(seconds=10),
            )
        )
        # A more recent idle/unsupported metrics scrape must not erase the
        # last non-zero rates reported by either llama.cpp or Halogen Flash.
        session.add(
            TokenUsageSample(
                server_instance_id=server_a.id,
                prompt_tokens=0,
                completion_tokens=0,
                prompt_per_second=0.0,
                predicted_per_second=0.0,
                created_at=now - timedelta(seconds=1),
            )
        )
        # server_a: older, outside the live window but inside 24h
        session.add(
            TokenUsageSample(
                server_instance_id=server_a.id,
                prompt_tokens=500,
                cached_tokens=0,
                completion_tokens=50,
                prompt_ms=1000.0,
                predicted_ms=5000.0,
                prompt_per_second=500.0,
                predicted_per_second=10.0,
                created_at=now - timedelta(hours=2),
            )
        )
        # server_b: 8 days old (excluded from 7d window, included in 30d)
        session.add(
            TokenUsageSample(
                server_instance_id=server_b.id,
                prompt_tokens=300,
                cached_tokens=0,
                completion_tokens=30,
                prompt_ms=3000.0,
                predicted_ms=6000.0,
                prompt_per_second=100.0,
                predicted_per_second=5.0,
                created_at=now - timedelta(days=8),
            )
        )
        # orphan sample with no server row
        session.add(
            TokenUsageSample(
                server_instance_id=None,
                prompt_tokens=7,
                completion_tokens=7,
                created_at=now,
            )
        )
        # ancient sample outside every window
        session.add(
            TokenUsageSample(
                server_instance_id=server_a.id,
                prompt_tokens=999,
                completion_tokens=999,
                created_at=now - timedelta(days=40),
            )
        )
        session.commit()

    snapshot = await token_stats_snapshot()
    glob = snapshot["global"]
    assert glob["live"]["runs"] == 1
    # Per-server rates use each server's latest metric scrape; global rates
    # are the mean of those current per-server gauges.
    assert glob["live"]["decode_tokens_per_second"] == pytest.approx(27.5)
    assert glob["live"]["prefill_tokens_per_second"] == pytest.approx(300)

    assert glob["last_24h"]["prompt_tokens"] == 1107
    assert glob["last_24h"]["completion_tokens"] == 157
    assert glob["last_24h"]["total_tokens"] == 1264

    assert glob["last_7d"]["prompt_tokens"] == 1107
    assert glob["last_30d"]["prompt_tokens"] == 1407
    assert glob["last_30d"]["completion_tokens"] == 187

    servers = snapshot["servers"]
    # orphan excluded; server_a first (only one with a live rate)
    assert [s["id"] for s in servers] == [str(server_a.id), str(server_b.id)]
    assert servers[0]["alias"] == server_a.alias
    assert servers[0]["decode_tokens_per_second"] == 50.0
    assert servers[0]["prefill_tokens_per_second"] == 500.0
    assert servers[0]["last_7d"]["total_tokens"] == 1250
    assert servers[1]["last_7d"]["total_tokens"] == 0
    assert servers[1]["last_30d"]["total_tokens"] == 330
    # Latest available per-server scrape is used (this fixture is 8 days old).
    assert servers[1]["decode_tokens_per_second"] == 5.0
    assert servers[1]["prefill_tokens_per_second"] == 100.0


async def test_prune_removes_old_samples() -> None:
    cutoff = datetime.now(UTC) - timedelta(days=TOKEN_SAMPLE_RETENTION_DAYS)
    with Session(engine) as session:
        session.add(
            TokenUsageSample(
                prompt_tokens=1,
                completion_tokens=1,
                created_at=cutoff - timedelta(days=1),
            )
        )
        session.add(
            TokenUsageSample(
                prompt_tokens=1,
                completion_tokens=1,
                created_at=cutoff + timedelta(days=1),
            )
        )
        session.commit()

    removed = await prune_old_samples()
    assert removed >= 1
    with Session(engine) as session:
        old = list(
            session.exec(
                select(TokenUsageSample).where(TokenUsageSample.created_at < cutoff)
            )
        )
    assert old == []
    await async_engine.dispose()


def _sample_count() -> int:
    with Session(engine) as session:
        return len(list(session.exec(select(TokenUsageSample))))

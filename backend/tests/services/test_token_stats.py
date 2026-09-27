"""Tests for token usage sample recording and retention."""

import uuid
from datetime import UTC, datetime, timedelta

from sqlmodel import Session, select

from app.core.db import engine
from app.db.session import engine as async_engine
from app.models import Agent, Model, ServerInstance, TokenUsageSample
from app.services.token_stats import (
    LIVE_WINDOW_SECONDS,
    TOKEN_SAMPLE_RETENTION_DAYS,
    extract_usage_fields,
    prune_old_samples,
    record_usage,
    token_stats_snapshot,
)


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


async def test_snapshot_aggregates_windows_and_rates(db: Session) -> None:
    server_a = _make_server(db)
    server_b = _make_server(db)
    now = datetime.now(UTC)

    with Session(engine) as session:
        # server_a: fast decode + big prefill, inside the live window
        session.add(
            TokenUsageSample(
                server_instance_id=server_a.id,
                prompt_tokens=1000,
                cached_tokens=400,
                completion_tokens=100,
                prompt_ms=2000.0,
                predicted_ms=2000.0,
                created_at=now - timedelta(seconds=10),
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
    assert glob["live"]["window_seconds"] == LIVE_WINDOW_SECONDS
    # live decode: only server_a's 10s-old sample: 100 tok / 2 s
    assert glob["live"]["decode_tokens_per_second"] == 50.0
    # live prefill excludes cached tokens: (1000 - 400) / 2 s
    assert glob["live"]["prefill_tokens_per_second"] == 300.0

    assert glob["last_24h"]["prompt_tokens"] == 1507
    assert glob["last_24h"]["completion_tokens"] == 157
    assert glob["last_24h"]["total_tokens"] == 1664

    assert glob["last_7d"]["prompt_tokens"] == 1507
    assert glob["last_30d"]["prompt_tokens"] == 1807
    assert glob["last_30d"]["completion_tokens"] == 187

    servers = snapshot["servers"]
    # orphan excluded; server_a first (only one with a live rate)
    assert [s["id"] for s in servers] == [str(server_a.id), str(server_b.id)]
    assert servers[0]["alias"] == server_a.alias
    assert servers[0]["decode_tokens_per_second"] == 50.0
    assert servers[0]["prefill_tokens_per_second"] == 300.0
    assert servers[0]["last_7d"]["total_tokens"] == 1650
    assert servers[1]["last_7d"]["total_tokens"] == 0
    assert servers[1]["last_30d"]["total_tokens"] == 330
    assert servers[1]["decode_tokens_per_second"] is None


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

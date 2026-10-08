"""Slice 1 — per-instance token/throughput statistics.

Covers:

- ``GET /admin/api/stats/instances`` — cumulative 24h/7d/30d windows, live
  throughput, mean rates, cache-hit rate, generation-tps percentiles, and a
  ``ResponseRecord``-derived error rate; embedding-modality blanking; every
  instance present (even with zero samples).
- ``GET /admin/api/instances`` now carries ``modality``.

Timestamps are seeded relative to ``datetime.now(UTC)`` so window membership is
deterministic. ``TokenUsageSample.created_at`` is ``timestamptz`` (seeded aware);
``ResponseRecord.created_at`` is a naive timestamp (seeded with ``tzinfo=None``).
"""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.models import ResponseRecord, TokenUsageSample
from tests.helpers import (
    get_or_create_machine,
    make_agent,
    make_definition,
    make_instance,
)


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


def _llm_stack(session: Session, *, alias: str, modality: str = "llm"):
    machine = get_or_create_machine(session, uid=f"st-{alias}")
    agent = make_agent(session, machine, provider_type="mock")
    definition = make_definition(session, alias=alias, provider_type="mock", modality=modality)
    instance = make_instance(session, agent, definition)
    return machine, agent, definition, instance


def _sample(
    session: Session,
    instance,
    definition,
    *,
    age: timedelta,
    prompt: int = 0,
    cached: int = 0,
    completion: int = 0,
    pps: float = 0.0,
    predps: float = 0.0,
    predms: float = 0.0,
) -> None:
    session.add(
        TokenUsageSample(
            provider_instance_id=instance.id,
            provider_definition_id=definition.id,
            prompt_tokens=prompt,
            cached_tokens=cached,
            completion_tokens=completion,
            prompt_per_second=pps,
            predicted_per_second=predps,
            predicted_ms=predms,
            created_at=datetime.now(UTC) - age,
        )
    )


def _record(
    session: Session,
    instance,
    definition,
    *,
    response_id: str,
    age: timedelta,
    status: str = "completed",
) -> None:
    session.add(
        ResponseRecord(
            response_id=response_id,
            provider_definition_id=definition.id,
            provider_instance_id=instance.id,
            status=status,
            created_at=(datetime.now(UTC) - age).replace(tzinfo=None),
        )
    )


# ---------------------------------------------------------------------------
# LLM windows / live / rates / percentiles
# ---------------------------------------------------------------------------


def test_instance_stats_llm_windows(client: TestClient, session: Session) -> None:
    _, _, definition, instance = _llm_stack(session, alias="llm-win")
    # A second instance with zero samples must still appear.
    _, _, _, empty = _llm_stack(session, alias="llm-empty")

    _sample(session, instance, definition, age=timedelta(minutes=1), prompt=100, cached=25, completion=10, pps=1000.0, predps=50.0, predms=200.0)
    _sample(session, instance, definition, age=timedelta(hours=12), prompt=200, cached=50, completion=20, pps=2000.0, predps=100.0, predms=400.0)
    _sample(session, instance, definition, age=timedelta(days=3), prompt=300, cached=0, completion=30, pps=3000.0, predps=150.0, predms=600.0)
    _sample(session, instance, definition, age=timedelta(days=10), prompt=400, cached=100, completion=40, pps=4000.0, predps=200.0, predms=800.0)
    # Older than 30d: excluded from every window.
    _sample(session, instance, definition, age=timedelta(days=60), prompt=999, cached=999, completion=999, pps=9999.0, predps=9999.0, predms=9999.0)
    session.commit()

    body = client.get("/admin/api/stats/instances").json()
    assert set(body) == {str(instance.id), str(empty.id)}

    entry = body[str(instance.id)]
    assert entry["modality"] == "llm"
    # Only the 1-minute sample is within the 5-minute live window.
    assert entry["live_throughput_tps"] == 50.0

    w24 = entry["windows"]["24h"]
    assert w24["completion_tokens"] == 30
    assert w24["prompt_tokens"] == 300
    assert w24["cached_tokens"] == 75
    assert w24["request_count"] == 2
    assert w24["avg_prompt_tps"] == 1500.0
    assert w24["avg_gen_tps"] == 75.0
    assert w24["cache_hit_rate"] == 0.25
    assert w24["avg_predicted_ms"] == 300.0
    assert w24["p50_gen_tps"] == 75.0
    assert w24["p95_gen_tps"] == 97.5

    w7 = entry["windows"]["7d"]
    assert w7["completion_tokens"] == 60
    assert w7["prompt_tokens"] == 600
    assert w7["cached_tokens"] == 75
    assert w7["request_count"] == 3
    assert w7["avg_prompt_tps"] == 2000.0
    assert w7["avg_gen_tps"] == 100.0
    assert w7["cache_hit_rate"] == 0.125
    assert w7["avg_predicted_ms"] == 400.0
    assert w7["p50_gen_tps"] == 100.0
    assert w7["p95_gen_tps"] == 145.0

    w30 = entry["windows"]["30d"]
    assert w30["completion_tokens"] == 100
    assert w30["prompt_tokens"] == 1000
    assert w30["cached_tokens"] == 175
    assert w30["request_count"] == 4
    assert w30["avg_prompt_tps"] == 2500.0
    assert w30["avg_gen_tps"] == 125.0
    assert w30["cache_hit_rate"] == 0.175
    assert w30["avg_predicted_ms"] == 500.0
    assert w30["p50_gen_tps"] == 125.0
    assert w30["p95_gen_tps"] == 192.5

    # Zero-sample instance: all sums zero, all rates null.
    empty_entry = body[str(empty.id)]
    assert empty_entry["live_throughput_tps"] is None
    for win in empty_entry["windows"].values():
        assert win["request_count"] == 0
        assert win["completion_tokens"] == 0
        assert win["avg_gen_tps"] is None
        assert win["p50_gen_tps"] is None
        assert win["error_rate"] is None


# ---------------------------------------------------------------------------
# Window boundary discrimination (>= cutoff semantics)
# ---------------------------------------------------------------------------


def test_instance_stats_window_boundary(client: TestClient, session: Session) -> None:
    _, _, definition, instance = _llm_stack(session, alias="llm-edge")
    # Just inside vs just outside the 24h edge. A small (30s) margin is used
    # instead of literally ``now - 24h`` because the endpoint recomputes ``now``
    # a few ms after seeding, which would push an exactly-at-edge sample onto the
    # excluded side; the margin keeps the assertion deterministic while still
    # proving the cutoff sits at 24h (not 23h/25h).
    _sample(
        session,
        instance,
        definition,
        age=timedelta(hours=24) - timedelta(seconds=30),
        completion=11,
        predps=10.0,
    )
    _sample(
        session,
        instance,
        definition,
        age=timedelta(hours=24) + timedelta(seconds=30),
        completion=22,
        predps=20.0,
    )
    session.commit()

    entry = client.get("/admin/api/stats/instances").json()[str(instance.id)]
    # Only the just-inside sample is in the 24h window.
    assert entry["windows"]["24h"]["request_count"] == 1
    assert entry["windows"]["24h"]["completion_tokens"] == 11
    # Both are within 7d (the just-outside one falls into the wider window).
    assert entry["windows"]["7d"]["request_count"] == 2
    assert entry["windows"]["7d"]["completion_tokens"] == 33


# ---------------------------------------------------------------------------
# Live throughput is a mean over multiple in-window samples
# ---------------------------------------------------------------------------


def test_instance_stats_live_is_mean_of_window(
    client: TestClient, session: Session
) -> None:
    _, _, definition, instance = _llm_stack(session, alias="llm-live")
    # Two samples inside the 5-minute live window with different rates, plus one
    # outside it (but inside 24h) that must not affect the live mean.
    _sample(session, instance, definition, age=timedelta(minutes=1), predps=40.0)
    _sample(session, instance, definition, age=timedelta(minutes=2), predps=60.0)
    _sample(session, instance, definition, age=timedelta(minutes=10), predps=999.0)
    session.commit()

    entry = client.get("/admin/api/stats/instances").json()[str(instance.id)]
    # mean(40, 60) = 50.0 — a single-sample pick would yield 40.0 or 60.0.
    assert entry["live_throughput_tps"] == 50.0


# ---------------------------------------------------------------------------
# Embedding blanking
# ---------------------------------------------------------------------------


def test_instance_stats_embedding_blanking(
    client: TestClient, session: Session
) -> None:
    _, _, definition, instance = _llm_stack(
        session, alias="emb-1", modality="embedding"
    )
    _sample(
        session,
        instance,
        definition,
        age=timedelta(hours=1),
        prompt=500,
        cached=0,
        completion=0,
        pps=1234.0,
        predps=0.0,
        predms=0.0,
    )
    session.commit()

    entry = client.get("/admin/api/stats/instances").json()[str(instance.id)]
    assert entry["modality"] == "embedding"
    assert entry["live_throughput_tps"] is None
    for win in entry["windows"].values():
        assert win["prompt_tokens"] == 500
        assert win["request_count"] == 1
        assert win["completion_tokens"] == 0
        assert win["cached_tokens"] == 0
        # LLM-only fields blanked even though prompt_per_second > 0.
        assert win["avg_prompt_tps"] is None
        assert win["avg_gen_tps"] is None
        assert win["cache_hit_rate"] is None
        assert win["avg_predicted_ms"] is None
        assert win["p50_gen_tps"] is None
        assert win["p95_gen_tps"] is None


# ---------------------------------------------------------------------------
# Error rate from ResponseRecord
# ---------------------------------------------------------------------------


def test_instance_stats_error_rate(client: TestClient, session: Session) -> None:
    _, _, definition, instance = _llm_stack(session, alias="err-1")
    _record(session, instance, definition, response_id="r-completed", age=timedelta(hours=2), status="completed")
    _record(session, instance, definition, response_id="r-failed", age=timedelta(hours=3), status="failed")
    # Outside 30d: excluded from every window.
    _record(session, instance, definition, response_id="r-old", age=timedelta(days=40), status="incomplete")
    session.commit()

    entry = client.get("/admin/api/stats/instances").json()[str(instance.id)]
    for name in ("24h", "7d", "30d"):
        assert entry["windows"][name]["error_rate"] == 0.5


# ---------------------------------------------------------------------------
# instances list carries modality
# ---------------------------------------------------------------------------


def test_instances_list_carries_modality(client: TestClient, session: Session) -> None:
    _, _, _, llm_inst = _llm_stack(session, alias="mod-llm", modality="llm")
    _, _, _, emb_inst = _llm_stack(session, alias="mod-emb", modality="embedding")

    items = {i["id"]: i for i in client.get("/admin/api/instances").json()}
    assert items[str(llm_inst.id)]["modality"] == "llm"
    assert items[str(emb_inst.id)]["modality"] == "embedding"

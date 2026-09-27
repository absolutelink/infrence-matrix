import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlmodel import select

from app.models import Agent, InferenceLease, Model, ServerInstance
from app.services.inference_scheduler import (
    InferenceLeaseHandle,
    InferenceLeaseLost,
    InferenceRequestCancelled,
    InferenceScheduler,
    _vram_requirements_fit,
    server_capacity,
)


def _make_model(db, name: str) -> Model:
    model = Model(
        name=name,
        path=f"/models/{name}",
        size_bytes=1_000_000,
        architecture="llama",
        quantization="Q4_K_M",
        source="huggingface",
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def _make_server(
    db, model: Model, *, alias: str, effective_capacity: int = 1
) -> ServerInstance:
    agent = Agent(
        name=f"scheduler-agent-{uuid.uuid4().hex}",
        host="127.0.0.1",
        port=8080,
        status="online",
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    server = ServerInstance(
        model_id=model.id,
        agent_id=agent.id,
        alias=alias,
        process_command="llama-server",
        status="running",
        health_status="healthy",
        effective_capacity=effective_capacity,
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    return server


def _make_lease(
    db,
    model: Model,
    request_id: str,
    *,
    queued_at: datetime | None = None,
    status: str = "queued",
    server: ServerInstance | None = None,
    expires_at: datetime | None = None,
) -> InferenceLease:
    lease = InferenceLease(
        request_id=request_id,
        model_id=model.id,
        status=status,
        server_instance_id=server.id if server else None,
        slot_generation=server.slot_generation if server else None,
        queued_at=queued_at or datetime.now(UTC),
        started_at=datetime.now(UTC) if status == "active" else None,
        lease_expires_at=expires_at or datetime.now(UTC) + timedelta(minutes=5),
    )
    db.add(lease)
    db.commit()
    db.refresh(lease)
    return lease


def test_server_capacity_uses_effective_capacity():
    server = SimpleNamespace(
        engine="llamacpp",
        engine_options={},
        server_options={"parallel": 8},
        effective_capacity=3,
    )

    assert server_capacity(server) == 3


def test_flash_uses_kv_slots_for_capacity():
    server = SimpleNamespace(
        engine="halogen-flash",
        engine_options={"kv_slots": 2},
        server_options={},
        effective_capacity=2,
    )
    assert server_capacity(server) == 2


def test_vram_admission_uses_configured_requirements():
    assert not _vram_requirements_fit(113, 30, 110)
    assert _vram_requirements_fit(113, 30, 0)


def test_default_server_capacity_is_one():
    server = SimpleNamespace(engine="llamacpp", engine_options={}, server_options={})

    assert server_capacity(server) == 1


@pytest.mark.asyncio
async def test_release_completes_when_request_task_is_cancelled():
    commit_started = asyncio.Event()
    allow_commit = asyncio.Event()
    session = AsyncMock()
    session.execute.return_value.rowcount = 1

    async def commit() -> None:
        commit_started.set()
        await allow_commit.wait()

    session.commit.side_effect = commit
    session.add = Mock()
    session_context = AsyncMock()
    session_context.__aenter__.return_value = session
    session_context.__aexit__.return_value = None

    handle = InferenceLeaseHandle("request-1", SimpleNamespace(), uuid.uuid4())
    with patch(
        "app.services.inference_scheduler.AsyncSessionMaker",
        return_value=session_context,
    ):
        release_task = asyncio.create_task(handle.release())
        await commit_started.wait()
        release_task.cancel()
        allow_commit.set()

        with pytest.raises(asyncio.CancelledError):
            await release_task

    session.execute.assert_awaited_once()
    session.commit.assert_awaited_once()


async def test_guard_cancels_upstream_when_lease_is_lost():
    operation_cancelled = asyncio.Event()

    async def upstream() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            operation_cancelled.set()

    handle = InferenceLeaseHandle("request-1", SimpleNamespace(), uuid.uuid4())
    guarded = asyncio.create_task(handle.guard(upstream()))
    await asyncio.sleep(0)
    handle.lost.set()

    with pytest.raises(InferenceLeaseLost):
        await guarded

    assert operation_cancelled.is_set()


async def test_renewal_does_not_extend_unstarted_lease():
    renewed = Mock(rowcount=1)
    session = AsyncMock()
    session.execute.return_value = renewed
    session_context = AsyncMock()
    session_context.__aenter__.return_value = session
    session_context.__aexit__.return_value = None
    handle = InferenceLeaseHandle(
        "request-1",
        SimpleNamespace(id=uuid.uuid4()),
        uuid.uuid4(),
        slot_generation=3,
    )

    with (
        patch(
            "app.services.inference_scheduler.AsyncSessionMaker",
            return_value=session_context,
        ),
        patch("app.services.inference_scheduler.LEASE_RENEWAL_INTERVAL_SECONDS", 0.001),
        patch("app.services.inference_scheduler.ACTIVE_LEASE_TTL_SECONDS", 0.01),
    ):
        handle.start_renewal()
        await asyncio.sleep(0.02)
        assert not handle._upstream_started.is_set()
        assert session.execute.await_count == 0
        await handle.release()


async def test_unexpected_acquisition_exception_terminalizes_queued_lease(db):
    model = _make_model(db, "scheduler-acquisition-error.gguf")
    scheduler = InferenceScheduler()
    request_id = f"request-{uuid.uuid4()}"

    with patch.object(
        scheduler,
        "_admission_open",
        new=AsyncMock(side_effect=RuntimeError("admission failed")),
    ):
        with pytest.raises(RuntimeError, match="admission failed"):
            await scheduler.acquire(model.id, request_id, timeout=5)

    db.expire_all()
    lease = db.exec(
        select(InferenceLease).where(InferenceLease.request_id == request_id)
    ).one()
    assert lease.status == "failed"
    assert lease.terminal_reason == "admission_error"
    assert lease.released_at is not None


@pytest.mark.asyncio
async def test_disconnect_after_claim_cancels_before_returning_lease():
    scheduler = InferenceScheduler()
    queued = SimpleNamespace(id=uuid.uuid4())
    server = SimpleNamespace(id=uuid.uuid4(), status="running", health_status="healthy")
    lease = SimpleNamespace(cancel=AsyncMock())
    is_cancelled = AsyncMock(side_effect=[False, True])

    with (
        patch.object(scheduler, "_queue", new=AsyncMock(return_value=queued)),
        patch.object(scheduler, "_ensure_queued", new=AsyncMock()),
        patch.object(scheduler, "_admission_open", new=AsyncMock(return_value=True)),
        patch.object(scheduler, "_candidates", new=AsyncMock(return_value=[server])),
        patch.object(scheduler, "_active_leases", new=AsyncMock(return_value=0)),
        patch.object(scheduler, "_claim", new=AsyncMock(return_value=lease)),
        patch.object(scheduler, "_finish_queued", new=AsyncMock()),
    ):
        with pytest.raises(InferenceRequestCancelled):
            await scheduler.acquire(
                uuid.uuid4(),
                "disconnected-before-dispatch",
                is_cancelled=is_cancelled,
            )

    lease.cancel.assert_awaited_once()


async def test_compatible_fifo_does_not_block_unrelated_models(db):
    now = datetime.now(UTC)
    blocked_model = _make_model(db, "scheduler-blocked-model.gguf")
    available_model = _make_model(db, "scheduler-available-model.gguf")
    server = _make_server(db, available_model, alias="scheduler-unrelated-model-server")
    older = _make_lease(
        db,
        blocked_model,
        f"older-{uuid.uuid4()}",
        queued_at=now - timedelta(seconds=1),
    )
    compatible = _make_lease(
        db, available_model, f"compatible-{uuid.uuid4()}", queued_at=now
    )

    handle = await InferenceScheduler()._claim(compatible.id, server, capacity=1)

    assert handle is not None
    try:
        db.expire_all()
        assert db.get(InferenceLease, older.id).status == "queued"
        assert db.get(InferenceLease, compatible.id).status == "active"
    finally:
        await handle.release()


async def test_only_one_concurrent_claim_gets_capacity_one_server(db):
    model = _make_model(db, "scheduler-concurrent-model.gguf")
    server = _make_server(db, model, alias="scheduler-capacity-one-server")
    first = _make_lease(db, model, f"first-{uuid.uuid4()}")
    second = _make_lease(
        db,
        model,
        f"second-{uuid.uuid4()}",
        queued_at=first.queued_at + timedelta(microseconds=1),
    )

    claims = await asyncio.gather(
        InferenceScheduler()._claim(first.id, server, capacity=1),
        InferenceScheduler()._claim(second.id, server, capacity=1),
    )

    assert sum(claim is not None for claim in claims) == 1
    handle = next(claim for claim in claims if claim is not None)
    try:
        db.expire_all()
        statuses = {
            db.get(InferenceLease, first.id).status,
            db.get(InferenceLease, second.id).status,
        }
        assert statuses == {"active", "queued"}
    finally:
        await handle.release()


@pytest.mark.parametrize(
    "terminal_status", ["cancelled", "expired", "failed", "released"]
)
async def test_terminal_lease_transitions_are_immutable(db, terminal_status):
    model = _make_model(db, f"scheduler-terminal-{terminal_status}.gguf")
    lease = _make_lease(
        db,
        model,
        f"terminal-{terminal_status}-{uuid.uuid4()}",
        status=terminal_status,
    )
    lease.terminal_reason = "original_reason"
    lease.released_at = datetime.now(UTC)
    original_released_at = lease.released_at
    db.add(lease)
    db.commit()

    changed = await InferenceScheduler()._finish_queued(
        lease.id, "failed", "replacement_reason"
    )

    db.expire_all()
    persisted = db.get(InferenceLease, lease.id)
    assert changed is False
    assert persisted.status == terminal_status
    assert persisted.terminal_reason == "original_reason"
    assert persisted.released_at == original_released_at


async def test_persisted_reconciliation_preserves_healthy_active_lease(db):
    model = _make_model(db, "scheduler-healthy-active.gguf")
    server = _make_server(db, model, alias="scheduler-healthy-active-server")
    lease = _make_lease(
        db,
        model,
        f"healthy-{uuid.uuid4()}",
        status="active",
        server=server,
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    scheduler = InferenceScheduler()

    with patch(
        "app.services.inference_scheduler.agent_manager.send_to_agent",
        new=AsyncMock(),
    ) as send_to_agent:
        reconciled = await scheduler.reconcile_persisted_leases()

    db.expire_all()
    assert reconciled == 0
    assert db.get(InferenceLease, lease.id).status == "active"
    assert db.get(ServerInstance, server.id).status == "running"
    send_to_agent.assert_not_awaited()


async def test_reconciliation_does_not_stop_healthy_server_for_expired_lease(db):
    model = _make_model(db, "scheduler-expired-healthy.gguf")
    server = _make_server(db, model, alias="scheduler-expired-healthy-server")
    lease = _make_lease(
        db,
        model,
        f"expired-healthy-{uuid.uuid4()}",
        status="active",
        server=server,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    scheduler = InferenceScheduler()

    with patch(
        "app.services.inference_scheduler.agent_manager.send_to_agent",
        new=AsyncMock(),
    ) as send_to_agent:
        reconciled = await scheduler.reconcile_stale_leases()

    db.expire_all()
    persisted_lease = db.get(InferenceLease, lease.id)
    persisted_server = db.get(ServerInstance, server.id)
    assert reconciled == 1
    assert persisted_lease.status == "failed"
    assert persisted_lease.terminal_reason == "lease_expired_or_server_unavailable"
    assert persisted_server.status == "running"
    send_to_agent.assert_not_awaited()


async def test_reconciliation_preserves_a_renewed_healthy_lease(db):
    model = _make_model(db, "scheduler-renewed-active.gguf")
    server = _make_server(db, model, alias="scheduler-renewed-active-server")
    lease = _make_lease(
        db,
        model,
        f"renewed-{uuid.uuid4()}",
        status="active",
        server=server,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    lease.lease_expires_at = datetime.now(UTC) + timedelta(minutes=5)
    db.add(lease)
    db.commit()

    reconciled = await InferenceScheduler().reconcile_stale_leases()

    db.expire_all()
    assert reconciled == 0
    assert db.get(InferenceLease, lease.id).status == "active"


async def test_reconciliation_expires_queued_lease_past_deadline(db):
    model = _make_model(db, "scheduler-expired-queued.gguf")
    lease = _make_lease(
        db,
        model,
        f"expired-{uuid.uuid4()}",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    reconciled = await InferenceScheduler().reconcile_stale_leases()

    db.expire_all()
    persisted = db.get(InferenceLease, lease.id)
    assert reconciled == 1
    assert persisted.status == "expired"
    assert persisted.terminal_reason == "queue_timeout"
    assert persisted.released_at is not None

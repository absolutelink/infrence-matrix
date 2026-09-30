import asyncio
import hashlib
import logging
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy import text
from sqlmodel import select

from app.db.session import engine as async_engine
from app.models import Agent, InferenceLease, Model, ServerInstance
from app.services.inference_scheduler import (
    InferenceLeaseHandle,
    InferenceLeaseLost,
    InferenceRequestCancelled,
    InferenceScheduler,
    _vram_requirements_fit,
    server_capacity,
)
from app.services.scheduler_locks import release_session_advisory_lock


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
    db,
    model: Model,
    *,
    alias: str,
    effective_capacity: int = 1,
    inference_slot_protocol: int = 1,
) -> ServerInstance:
    agent = Agent(
        name=f"scheduler-agent-{uuid.uuid4().hex}",
        host="127.0.0.1",
        port=8080,
        status="online",
        inference_slot_protocol=inference_slot_protocol,
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


def _resource_key(host: str, gpu_id: object) -> int:
    resource_name = f"{host}:{gpu_id}"
    return int.from_bytes(
        hashlib.blake2b(resource_name.encode(), digest_size=8).digest(),
        "big",
    ) & ((1 << 63) - 1)


async def _advisory_lock_pids(key: int) -> list[int]:
    async with async_engine.connect() as connection:
        rows = (
            await connection.execute(
                text(
                    "SELECT pid FROM pg_locks WHERE locktype = 'advisory' "
                    "AND ((classid::bigint << 32) | "
                    "(objid::bigint & 4294967295)) = :key"
                ),
                {"key": key},
            )
        ).all()
        await connection.commit()
    return [int(row[0]) for row in rows]


async def _try_advisory_lock_elsewhere(key: int) -> bool:
    async with async_engine.connect() as connection:
        acquired = bool(
            (
                await connection.execute(
                    text("SELECT pg_try_advisory_lock(:key)"), {"key": key}
                )
            ).scalar_one()
        )
        await connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})
        await connection.commit()
    return acquired


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
async def test_release_completes_when_request_task_is_cancelled(caplog):
    caplog.set_level(logging.INFO)
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
    with (
        patch(
            "app.services.inference_scheduler.AsyncSessionMaker",
            return_value=session_context,
        ),
        patch(
            "app.services.inference_scheduler.UPSTREAM_COMPLETION_COOLDOWN_SECONDS", 0
        ),
    ):
        release_task = asyncio.create_task(handle.release())
        await commit_started.wait()
        release_task.cancel()
        allow_commit.set()

        with pytest.raises(asyncio.CancelledError):
            await release_task

    session.execute.assert_awaited_once()
    session.commit.assert_awaited_once()
    assert "inference_lease_release_begin request_id=request-1" in caplog.text
    assert "inference_lease_release_complete request_id=request-1" in caplog.text
    assert "request_cancelled=true" in caplog.text


@pytest.mark.asyncio
async def test_release_keeps_slot_active_during_completion_cooldown():
    cooldown_started = asyncio.Event()
    allow_cooldown = asyncio.Event()
    session = AsyncMock()
    session.execute.return_value.rowcount = 1
    session_context = AsyncMock()
    session_context.__aenter__.return_value = session
    session_context.__aexit__.return_value = None

    async def sleep(_seconds: float) -> None:
        cooldown_started.set()
        await allow_cooldown.wait()

    handle = InferenceLeaseHandle(
        "request-1", SimpleNamespace(), uuid.uuid4(), legacy_capacity=True
    )
    with (
        patch(
            "app.services.inference_scheduler.AsyncSessionMaker",
            return_value=session_context,
        ),
        patch("app.services.inference_scheduler.asyncio.sleep", side_effect=sleep),
    ):
        release_task = asyncio.create_task(handle.release())
        await cooldown_started.wait()
        session.execute.assert_not_awaited()
        allow_cooldown.set()
        await release_task

    session.execute.assert_awaited_once()


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
    lease = SimpleNamespace(cancel=AsyncMock(), monitor_disconnect=Mock())
    disconnected = asyncio.Event()

    async def is_cancelled() -> bool:
        return disconnected.is_set()

    async def claim(*_args):
        disconnected.set()
        await asyncio.sleep(0.11)
        return lease

    with (
        patch.object(scheduler, "_queue", new=AsyncMock(return_value=queued)),
        patch.object(scheduler, "_ensure_queued", new=AsyncMock()),
        patch.object(scheduler, "_admission_open", new=AsyncMock(return_value=True)),
        patch.object(scheduler, "_candidates", new=AsyncMock(return_value=[server])),
        patch.object(scheduler, "_active_leases", new=AsyncMock(return_value=0)),
        patch.object(scheduler, "_claim", new=claim),
        patch.object(scheduler, "_finish_queued", new=AsyncMock()),
    ):
        with pytest.raises(InferenceRequestCancelled):
            await scheduler.acquire(
                uuid.uuid4(),
                "disconnected-before-dispatch",
                is_cancelled=is_cancelled,
            )

    lease.cancel.assert_awaited_once()


@pytest.mark.asyncio
async def test_disconnect_watcher_removes_queued_lease():
    scheduler = InferenceScheduler()
    disconnected = asyncio.Event()

    async def is_cancelled() -> bool:
        return disconnected.is_set()

    with patch.object(scheduler, "_cancel", new=AsyncMock()) as cancel:
        disconnected_event = asyncio.Event()
        watcher = asyncio.create_task(
            scheduler._cancel_when_disconnected(
                uuid.uuid4(), is_cancelled, disconnected_event
            )
        )
        disconnected.set()
        await watcher

    cancel.assert_awaited_once()
    assert disconnected_event.is_set()


@pytest.mark.asyncio
async def test_active_disconnect_marks_lease_lost():
    handle = InferenceLeaseHandle("request-1", SimpleNamespace(), uuid.uuid4())
    disconnected = asyncio.Event()

    async def is_cancelled() -> bool:
        return disconnected.is_set()

    with patch.object(handle, "_mark_cancelled", new=AsyncMock()) as mark_cancelled:
        handle.monitor_disconnect(is_cancelled)
        disconnected.set()
        await asyncio.sleep(0.11)

    assert handle.lost.is_set()
    mark_cancelled.assert_awaited_once()
    await handle._stop_disconnect_monitor()


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

    handle = await InferenceScheduler()._claim(compatible.id, server)

    assert handle is not None
    try:
        db.expire_all()
        assert db.get(InferenceLease, older.id).status == "queued"
        assert db.get(InferenceLease, compatible.id).status == "active"
    finally:
        await handle.release()


async def test_expired_active_lease_does_not_block_new_claim(db):
    model = _make_model(db, f"scheduler-expired-slot-{uuid.uuid4().hex}")
    server = _make_server(db, model, alias=f"scheduler-expired-slot-{uuid.uuid4().hex}")
    _make_lease(
        db,
        model,
        f"expired-active-{uuid.uuid4()}",
        status="active",
        server=server,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    queued = _make_lease(db, model, f"queued-{uuid.uuid4()}")

    handle = await InferenceScheduler()._claim(queued.id, server)

    assert handle is not None
    await handle.release()


async def test_newly_ready_server_waits_before_claiming(db):
    model = _make_model(db, "scheduler-ready-cooldown.gguf")
    server = _make_server(db, model, alias="scheduler-ready-cooldown-server")
    lease = _make_lease(db, model, f"cooldown-{uuid.uuid4()}")
    server.started_at = datetime.now(UTC) - timedelta(seconds=1)
    db.add(server)
    db.commit()

    scheduler = InferenceScheduler()
    assert await scheduler._claim(lease.id, server) is None

    server.started_at = datetime.now(UTC) - timedelta(seconds=4)
    db.add(server)
    db.commit()
    handle = await scheduler._claim(lease.id, server)

    assert handle is not None
    await handle.release()


async def test_concurrent_claims_queue_at_agent_for_capacity_one_server(db):
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
        InferenceScheduler()._claim(first.id, server),
        InferenceScheduler()._claim(second.id, server),
    )

    assert all(claim is not None for claim in claims)
    try:
        db.expire_all()
        statuses = {
            db.get(InferenceLease, first.id).status,
            db.get(InferenceLease, second.id).status,
        }
        assert statuses == {"active"}
    finally:
        await asyncio.gather(*(claim.release() for claim in claims if claim))


async def test_cross_server_claim_waits_for_older_locked_compatible_request(db):
    model = _make_model(db, f"scheduler-cross-server-fifo-{uuid.uuid4().hex}")
    server_a = _make_server(
        db, model, alias=f"scheduler-cross-server-a-{uuid.uuid4().hex}"
    )
    server_b = _make_server(
        db, model, alias=f"scheduler-cross-server-b-{uuid.uuid4().hex}"
    )
    older = _make_lease(db, model, f"cross-server-older-{uuid.uuid4()}")
    newer = _make_lease(
        db,
        model,
        f"cross-server-newer-{uuid.uuid4()}",
        queued_at=older.queued_at + timedelta(microseconds=1),
    )
    newer.preferred_server_id = server_b.id
    db.add(newer)
    db.commit()

    connection = await async_engine.connect()
    transaction = await connection.begin()
    await connection.execute(
        select(InferenceLease.id).where(InferenceLease.id == older.id).with_for_update()
    )
    worker_a = InferenceScheduler()
    worker_b = InferenceScheduler()
    newer_on_b = asyncio.create_task(worker_b._claim(newer.id, server_b))
    try:
        await asyncio.sleep(0.05)
        assert not newer_on_b.done()
    finally:
        await transaction.rollback()
        await connection.close()

    assert await newer_on_b is None
    older_on_a = await worker_a._claim(older.id, server_a)
    newer_on_b = await worker_b._claim(newer.id, server_b)
    assert older_on_a is not None
    assert newer_on_b is not None
    await asyncio.gather(older_on_a.release(), newer_on_b.release())


async def test_protocol_one_claim_never_waits_on_server_row_capacity_lock(db):
    model = _make_model(db, f"scheduler-agent-lock-{uuid.uuid4().hex}")
    server = _make_server(db, model, alias=f"agent-lock-{uuid.uuid4().hex}")
    queued = _make_lease(db, model, f"agent-lock-request-{uuid.uuid4()}")
    assert db.get(Agent, server.agent_id).inference_slot_protocol == 1

    connection = await async_engine.connect()
    transaction = await connection.begin()
    await connection.execute(
        select(ServerInstance.id)
        .where(ServerInstance.id == server.id)
        .with_for_update(key_share=True)
    )
    try:
        handle = await asyncio.wait_for(
            InferenceScheduler()._claim(queued.id, server), timeout=1.0
        )
        assert handle is not None
    finally:
        await transaction.rollback()
        await connection.close()
    await handle.release()


async def test_protocol_three_claim_waits_for_acknowledged_agent_capacity(db):
    model = _make_model(db, f"scheduler-reservation-{uuid.uuid4().hex}")
    server = _make_server(
        db,
        model,
        alias=f"scheduler-reservation-{uuid.uuid4().hex}",
        inference_slot_protocol=3,
    )
    first = _make_lease(db, model, f"first-{uuid.uuid4()}")
    second = _make_lease(
        db,
        model,
        f"second-{uuid.uuid4()}",
        queued_at=first.queued_at + timedelta(microseconds=1),
    )
    scheduler = InferenceScheduler()
    reservations = iter([False, True, True])

    async def respond(_agent_id, _method, path, **_kwargs):
        if "/reservations/" in path:
            return {"accepted": next(reservations)}
        return {"cancelled": True}

    with patch(
        "app.services.inference_scheduler.agent_manager.send_to_agent",
        new_callable=AsyncMock,
        side_effect=respond,
    ) as send:
        assert await scheduler._claim(first.id, server) is None
        db.expire_all()
        assert db.get(InferenceLease, first.id).status == "queued"
        assert await scheduler._claim(second.id, server) is None
        first_handle = await scheduler._claim(first.id, server)
        assert first_handle is not None
        assert first_handle.reservation_protocol
        assert (
            first_handle.dispatch_headers()["X-Inference-Reservation"]
            == first.request_id
        )
        db.expire_all()
        assert db.get(InferenceLease, first.id).status == "active"
        assert (
            db.get(InferenceLease, first.id).slot_generation == server.slot_generation
        )
        assert await scheduler._claim(second.id, server) is None
        assert (
            sum("/reservations/" in call.args[2] for call in send.await_args_list) == 2
        )
        await first_handle.release()
        second_handle = await scheduler._claim(second.id, server)
        assert second_handle is not None
        await second_handle.release()
    db.rollback()


async def test_protocol_three_concurrent_workers_reserve_only_once(db):
    model = _make_model(db, f"scheduler-parallel-{uuid.uuid4().hex}")
    server = _make_server(
        db,
        model,
        alias=f"scheduler-parallel-{uuid.uuid4().hex}",
        inference_slot_protocol=3,
    )
    queued = _make_lease(db, model, f"parallel-{uuid.uuid4()}")

    async def acknowledge(*_args, **_kwargs):
        await asyncio.sleep(0.02)
        return {"accepted": True}

    with patch(
        "app.services.inference_scheduler.agent_manager.send_to_agent",
        new_callable=AsyncMock,
        side_effect=acknowledge,
    ) as send:
        handles = await asyncio.gather(
            InferenceScheduler()._claim(queued.id, server),
            InferenceScheduler()._claim(queued.id, server),
        )
        assert sum(handle is not None for handle in handles) == 1
        assert send.await_count == 1
        await next(handle for handle in handles if handle is not None).release()


async def test_reserving_lease_can_be_cancelled_while_agent_is_slow(db):
    model = _make_model(db, f"scheduler-slow-{uuid.uuid4().hex}")
    server = _make_server(
        db, model, alias=f"slow-{uuid.uuid4().hex}", inference_slot_protocol=3
    )
    queued = _make_lease(db, model, f"slow-{uuid.uuid4()}")
    called = asyncio.Event()
    release = asyncio.Event()
    tokens = []

    async def agent_call(_agent_id, _method, path, *, headers=None, **_kwargs):
        tokens.append(headers["X-Inference-Reservation-ID"])
        if "/reservations/" in path:
            called.set()
            await release.wait()
            return {"accepted": True}
        return {"cancelled": True}

    scheduler = InferenceScheduler()
    with patch(
        "app.services.inference_scheduler.agent_manager.send_to_agent",
        new_callable=AsyncMock,
        side_effect=agent_call,
    ):
        pending = asyncio.create_task(scheduler._claim(queued.id, server))
        await asyncio.wait_for(called.wait(), 1)
        db.expire_all()
        assert db.get(InferenceLease, queued.id).status == "reserving"
        db.rollback()
        await scheduler._cancel(queued.id)
        release.set()
        assert await pending is None
        db.expire_all()
        assert db.get(InferenceLease, queued.id).status == "cancelled"
        assert len(tokens) == 2
        assert tokens[0] == tokens[1]
        db.rollback()


async def test_cancelled_claim_after_commit_does_not_leave_active_lease(db):
    model = _make_model(db, f"scheduler-post-commit-{uuid.uuid4().hex}")
    server = _make_server(
        db,
        model,
        alias=f"post-commit-{uuid.uuid4().hex}",
        inference_slot_protocol=3,
    )
    queued = _make_lease(db, model, f"post-commit-{uuid.uuid4()}")
    cancelled_tokens = []

    async def agent_call(_agent_id, _method, path, *, headers=None, **_kwargs):
        if "/reservations/" in path:
            return {"accepted": True}
        cancelled_tokens.append(headers["X-Inference-Reservation-ID"])
        return {"cancelled": True}

    with (
        patch(
            "app.services.inference_scheduler.agent_manager.send_to_agent",
            new_callable=AsyncMock,
            side_effect=agent_call,
        ),
        patch.object(
            InferenceLeaseHandle, "start_renewal", side_effect=asyncio.CancelledError
        ),
    ):
        with pytest.raises(asyncio.CancelledError):
            await InferenceScheduler()._claim(queued.id, server)
    db.expire_all()
    finished = db.get(InferenceLease, queued.id)
    assert finished.status == "cancelled"
    assert cancelled_tokens == [str(finished.reservation_id)]
    db.rollback()


async def test_lost_agent_ack_cancels_attempt_and_requeues(db):
    model = _make_model(db, f"scheduler-lost-ack-{uuid.uuid4().hex}")
    server = _make_server(
        db,
        model,
        alias=f"lost-ack-{uuid.uuid4().hex}",
        inference_slot_protocol=3,
    )
    queued = _make_lease(db, model, f"lost-ack-{uuid.uuid4()}")
    tokens = []

    async def agent_call(_agent_id, _method, path, *, headers=None, **_kwargs):
        tokens.append(headers["X-Inference-Reservation-ID"])
        if "/reservations/" in path:
            raise TimeoutError("Agent accepted but reply was lost")
        return {"cancelled": True}

    with patch(
        "app.services.inference_scheduler.agent_manager.send_to_agent",
        new_callable=AsyncMock,
        side_effect=agent_call,
    ):
        assert await InferenceScheduler()._claim(queued.id, server) is None
    db.expire_all()
    assert db.get(InferenceLease, queued.id).status == "queued"
    assert len(tokens) == 2 and tokens[0] == tokens[1]
    db.rollback()


async def test_abandoned_reservation_rejoins_fifo_after_recovery(db):
    model = _make_model(db, f"scheduler-recovery-{uuid.uuid4().hex}")
    server = _make_server(
        db,
        model,
        alias=f"recovery-{uuid.uuid4().hex}",
        inference_slot_protocol=3,
    )
    lease = _make_lease(db, model, f"recovery-{uuid.uuid4()}")
    lease.status = "reserving"
    lease.server_instance_id = server.id
    lease.slot_generation = server.slot_generation
    lease.reservation_id = uuid.uuid4()
    lease.reservation_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.add(lease)
    db.commit()

    assert await InferenceScheduler()._ensure_queued(lease.id)
    db.expire_all()
    recovered = db.get(InferenceLease, lease.id)
    assert recovered.status == "queued"
    assert recovered.reservation_id is None
    assert recovered.server_instance_id is None
    db.rollback()


async def test_legacy_agent_uses_database_capacity_fallback(db):
    model = _make_model(db, f"scheduler-legacy-agent-{uuid.uuid4().hex}")
    server = _make_server(
        db,
        model,
        alias=f"scheduler-legacy-agent-{uuid.uuid4().hex}",
        inference_slot_protocol=0,
    )
    first = _make_lease(db, model, f"legacy-first-{uuid.uuid4()}")
    second = _make_lease(
        db,
        model,
        f"legacy-second-{uuid.uuid4()}",
        queued_at=first.queued_at + timedelta(microseconds=1),
    )
    scheduler = InferenceScheduler()

    first_handle = await scheduler._claim(first.id, server)
    second_handle = await scheduler._claim(second.id, server)

    assert first_handle is not None
    assert second_handle is None
    await first_handle.release()
    second_handle = await scheduler._claim(second.id, server)
    assert second_handle is not None
    await second_handle.release()


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
        new=AsyncMock(return_value={"complete": True, "operations": []}),
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
        new=AsyncMock(return_value={"complete": True, "operations": []}),
    ) as send_to_agent:
        reconciled = await scheduler.reconcile_stale_leases()

    db.expire_all()
    persisted_lease = db.get(InferenceLease, lease.id)
    persisted_server = db.get(ServerInstance, server.id)
    assert reconciled == 1
    assert persisted_lease.status == "failed"
    assert persisted_lease.terminal_reason == "agent_operation_unknown"
    assert persisted_server.status == "running"
    send_to_agent.assert_awaited_once_with(
        str(server.agent_id),
        "GET",
        f"/proxy/{server.id}/operations",
        timeout=5.0,
    )


@pytest.mark.parametrize(
    "snapshot",
    [
        {"complete": False, "operations": []},
        {"complete": True, "operations": None},
    ],
)
async def test_reconciliation_defers_lease_for_incomplete_agent_snapshot(db, snapshot):
    model = _make_model(db, f"scheduler-incomplete-agent-snapshot-{uuid.uuid4().hex}")
    server = _make_server(
        db, model, alias=f"scheduler-incomplete-agent-snapshot-{uuid.uuid4().hex}"
    )
    lease = _make_lease(
        db,
        model,
        f"incomplete-agent-snapshot-{uuid.uuid4()}",
        status="active",
        server=server,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    with patch(
        "app.services.inference_scheduler.agent_manager.send_to_agent",
        new=AsyncMock(return_value=snapshot),
    ):
        reconciled = await InferenceScheduler().reconcile_stale_leases()

    db.expire_all()
    persisted = db.get(InferenceLease, lease.id)
    assert reconciled == 0
    assert persisted.status == "active"
    assert persisted.lease_expires_at > datetime.now(UTC)


async def test_legacy_agent_reconciles_expired_lease_without_operation_api(db):
    model = _make_model(db, f"scheduler-legacy-reconcile-{uuid.uuid4().hex}")
    server = _make_server(
        db,
        model,
        alias=f"scheduler-legacy-reconcile-{uuid.uuid4().hex}",
        inference_slot_protocol=0,
    )
    lease = _make_lease(
        db,
        model,
        f"legacy-reconcile-{uuid.uuid4()}",
        status="active",
        server=server,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    query = AsyncMock()

    with patch(
        "app.services.inference_scheduler.agent_manager.send_to_agent", new=query
    ):
        reconciled = await InferenceScheduler().reconcile_stale_leases()

    db.expire_all()
    persisted = db.get(InferenceLease, lease.id)
    assert reconciled == 1
    assert persisted.status == "failed"
    assert persisted.terminal_reason == "lease_expired_or_server_unavailable"
    query.assert_not_awaited()


async def test_reconciliation_fails_active_lease_from_previous_generation(db):
    model = _make_model(db, f"scheduler-old-generation-{uuid.uuid4().hex}")
    server = _make_server(
        db, model, alias=f"scheduler-old-generation-{uuid.uuid4().hex}"
    )
    lease = _make_lease(
        db,
        model,
        f"old-generation-{uuid.uuid4()}",
        status="active",
        server=server,
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )
    server.slot_generation += 1
    db.add(server)
    db.commit()

    with patch(
        "app.services.inference_scheduler.agent_manager.send_to_agent",
        new=AsyncMock(),
    ) as send_to_agent:
        reconciled = await InferenceScheduler().reconcile_stale_leases()

    db.expire_all()
    persisted = db.get(InferenceLease, lease.id)
    assert reconciled == 1
    assert persisted.status == "failed"
    assert persisted.terminal_reason == "server_generation_changed"
    send_to_agent.assert_not_awaited()


async def test_reconciliation_renews_lease_for_live_agent_operation(db):
    model = _make_model(db, "scheduler-live-agent-operation.gguf")
    server = _make_server(db, model, alias="scheduler-live-agent-operation-server")
    lease = _make_lease(
        db,
        model,
        f"live-agent-operation-{uuid.uuid4()}",
        status="active",
        server=server,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    scheduler = InferenceScheduler()
    query = AsyncMock(
        return_value={
            "complete": True,
            "operations": [
                {
                    "request_id": lease.request_id,
                    "status": "active",
                    "slot_generation": server.slot_generation,
                }
            ],
        }
    )

    with patch(
        "app.services.inference_scheduler.agent_manager.send_to_agent", new=query
    ):
        reconciled = await scheduler.reconcile_stale_leases()

    db.expire_all()
    persisted = db.get(InferenceLease, lease.id)
    assert reconciled == 0
    assert persisted.status == "active"
    assert persisted.lease_expires_at > datetime.now(UTC)
    query.assert_awaited_once_with(
        str(server.agent_id),
        "GET",
        f"/proxy/{server.id}/operations",
        timeout=5.0,
    )


@pytest.mark.parametrize(
    ("operation_status", "expected_status", "expected_reason"),
    [
        ("completed", "released", "completed"),
        ("cancelled", "cancelled", "agent_cancelled"),
        ("failed", "failed", "agent_operation_failed"),
    ],
)
async def test_reconciliation_maps_terminal_agent_operation_status(
    db, operation_status, expected_status, expected_reason
):
    model = _make_model(db, f"scheduler-terminal-operation-{uuid.uuid4().hex}")
    server = _make_server(
        db, model, alias=f"scheduler-terminal-operation-{uuid.uuid4().hex}"
    )
    lease = _make_lease(
        db,
        model,
        f"terminal-operation-{uuid.uuid4()}",
        status="active",
        server=server,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    query = AsyncMock(
        return_value={
            "complete": True,
            "operations": [
                {
                    "request_id": lease.request_id,
                    "status": operation_status,
                    "slot_generation": server.slot_generation,
                }
            ],
        }
    )

    with patch(
        "app.services.inference_scheduler.agent_manager.send_to_agent", new=query
    ):
        await InferenceScheduler().reconcile_stale_leases()

    db.expire_all()
    persisted = db.get(InferenceLease, lease.id)
    assert persisted.status == expected_status
    assert persisted.terminal_reason == expected_reason


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


def _make_stopped_target(db, name: str, gpu_id: str) -> ServerInstance:
    model = _make_model(db, f"{name}.gguf")
    server = _make_server(db, model, alias=f"{name}-server")
    agent = db.get(Agent, server.agent_id)
    agent.gpu_info = {"id": gpu_id}
    db.add(agent)
    server.status = "stopped"
    db.add(server)
    db.commit()
    db.refresh(server)
    return server


class _ScalarResult:
    def __init__(self, value: bool) -> None:
        self._value = value

    def scalar_one(self) -> bool:
        return self._value


class _CommitHangsAfterAcquire:
    """Connection proxy that hangs inside the commit right after the lock is granted.

    Models a client disconnect landing in the await window between acquiring the
    session-level advisory lock and the commit that follows it.
    """

    def __init__(self, connection) -> None:
        self._connection = connection
        self._armed = False
        self._hung = False
        self.commit_hanging = asyncio.Event()

    async def execute(self, statement, parameters=None):
        result = await self._connection.execute(statement, parameters)
        if "pg_try_advisory_lock" in str(statement):
            acquired = bool(result.scalar_one())
            if acquired:
                self._armed = True
            return _ScalarResult(acquired)
        return result

    async def commit(self) -> None:
        if self._armed and not self._hung:
            self._armed = False
            self._hung = True
            self.commit_hanging.set()
            await asyncio.Event().wait()
        await self._connection.commit()

    async def close(self) -> None:
        await self._connection.close()

    def __getattr__(self, name):
        return getattr(self._connection, name)


class _ConnectContext:
    def __init__(self, connection) -> None:
        self._connection = connection

    async def __aenter__(self):
        return self._connection

    async def __aexit__(self, *exc_info):
        await self._connection.close()


@pytest.mark.asyncio
async def test_cancel_during_post_acquire_commit_releases_advisory_lock(db):
    server = _make_stopped_target(db, "scheduler-commit-window", "gpu-commit")
    key = _resource_key("127.0.0.1", "gpu-commit")

    async with async_engine.connect() as inner:
        proxy = _CommitHangsAfterAcquire(inner)
        engine = SimpleNamespace(connect=lambda: _ConnectContext(proxy))
        with patch("app.services.inference_scheduler.async_engine", engine):
            task = asyncio.create_task(
                InferenceScheduler()._prepare_target(
                    server, asyncio.get_running_loop().time() + 30
                )
            )
            await asyncio.wait_for(proxy.commit_hanging.wait(), timeout=10)
            assert await _advisory_lock_pids(key)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    assert await _advisory_lock_pids(key) == []
    assert await _try_advisory_lock_elsewhere(key)


@pytest.mark.asyncio
async def test_cancel_after_lock_acquisition_releases_advisory_lock(db):
    server = _make_stopped_target(db, "scheduler-lock-leak", "gpu-leak")
    key = _resource_key("127.0.0.1", "gpu-leak")

    entered = asyncio.Event()
    hang = asyncio.Event()

    async def make_room(*_args, **_kwargs) -> None:
        entered.set()
        await hang.wait()

    scheduler = InferenceScheduler()
    with patch.object(scheduler, "_make_room", new=make_room):
        task = asyncio.create_task(
            scheduler._prepare_target(server, asyncio.get_running_loop().time() + 30)
        )
        await asyncio.wait_for(entered.wait(), timeout=10)
        assert await _advisory_lock_pids(key)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert await _advisory_lock_pids(key) == []
    assert await _try_advisory_lock_elsewhere(key)


@pytest.mark.asyncio
async def test_cancelled_preparer_does_not_stall_next_preparer(db):
    server = _make_stopped_target(db, "scheduler-lock-handoff", "gpu-handoff")
    key = _resource_key("127.0.0.1", "gpu-handoff")

    entered = asyncio.Event()
    hang = asyncio.Event()

    async def make_room(*_args, **_kwargs) -> None:
        entered.set()
        await hang.wait()

    async def make_room_now(*_args, **_kwargs) -> None:
        return None

    async def ready(current, **_kwargs):
        return current

    blocked = InferenceScheduler()
    second = InferenceScheduler()
    with (
        patch.object(blocked, "_make_room", new=make_room),
        patch("app.services.server_startup.ensure_server_ready", new=ready),
    ):
        first = asyncio.create_task(
            blocked._prepare_target(server, asyncio.get_running_loop().time() + 30)
        )
        await asyncio.wait_for(entered.wait(), timeout=10)

        with patch.object(second, "_make_room", new=make_room_now):
            handoff = asyncio.create_task(
                second._prepare_target(server, asyncio.get_running_loop().time() + 30)
            )
            await asyncio.sleep(0.2)
            assert len(await _advisory_lock_pids(key)) == 1

            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

            await asyncio.wait_for(handoff, timeout=15)
            assert len(await _advisory_lock_pids(key)) == 0


@pytest.mark.asyncio
async def test_prepare_target_deadline_raises_timeout_without_holding_lock(db):
    server = _make_stopped_target(db, "scheduler-lock-timeout", "gpu-timeout")
    key = _resource_key("127.0.0.1", "gpu-timeout")
    scheduler = InferenceScheduler()

    with pytest.raises(TimeoutError):
        await scheduler._prepare_target(server, asyncio.get_running_loop().time() - 1)

    assert await _advisory_lock_pids(key) == []


@pytest.mark.asyncio
async def test_release_helper_unlocks_even_when_caller_cancelled():
    key = _resource_key("127.0.0.1", "gpu-helper")
    connection = await async_engine.connect()
    acquired = (
        await connection.execute(
            text("SELECT pg_try_advisory_lock(:key)"), {"key": key}
        )
    ).scalar_one()
    assert acquired
    await connection.commit()
    assert await _advisory_lock_pids(key)

    task = asyncio.create_task(release_session_advisory_lock(connection, key))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await _advisory_lock_pids(key) == []
    assert await _try_advisory_lock_elsewhere(key)


@pytest.mark.asyncio
async def test_release_helper_is_noop_when_lock_never_held():
    key = _resource_key("127.0.0.1", "gpu-noop")
    connection = await async_engine.connect()

    await release_session_advisory_lock(connection, key)

    assert connection.closed
    assert await _advisory_lock_pids(key) == []


@pytest.mark.asyncio
async def test_status_snapshot_separates_expired_active_leases(db):
    model = _make_model(db, f"scheduler-status-{uuid.uuid4().hex}")
    server = _make_server(db, model, alias=f"scheduler-status-{uuid.uuid4().hex}")
    _make_lease(
        db,
        model,
        f"stale-{uuid.uuid4().hex}",
        status="active",
        server=server,
        expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )

    snapshot = await InferenceScheduler().status_snapshot()
    data = snapshot["data"]
    server_status = next(
        item for item in data["servers"] if item["id"] == str(server.id)
    )

    assert data["stale_active"] == 1
    assert data["reconciliation_running"] is False
    assert data["last_reconciliation_at"] is None
    assert server_status["stale_active"] == 1
    assert server_status["active"] == 1
    assert server_status["available"] == 0

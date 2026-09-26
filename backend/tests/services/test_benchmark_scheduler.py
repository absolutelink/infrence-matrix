"""Scheduler coordination tests for persisted benchmark runs."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.models import (
    Agent,
    BenchmarkDefinition,
    BenchmarkRun,
    InferenceLease,
    Model,
    ServerInstance,
)
from app.services import benchmark


def _benchmark_run(db, *, status: str = "queued") -> tuple[BenchmarkRun, Model]:
    suffix = uuid.uuid4().hex
    model = Model(
        name=f"benchmark-{suffix}.gguf",
        path=f"/models/benchmark-{suffix}.gguf",
        size_bytes=100,
        architecture="llama",
        quantization="Q4_K_M",
        source="huggingface",
    )
    agent = Agent(
        name=f"benchmark-agent-{suffix}",
        host="localhost",
        port=8080,
        status="online",
    )
    db.add(model)
    db.add(agent)
    db.flush()
    definition = BenchmarkDefinition(
        name=f"benchmark-{suffix}",
        config={"model_id": str(model.id), "agent_id": str(agent.id)},
    )
    db.add(definition)
    db.flush()
    run = BenchmarkRun(definition_id=definition.id, status=status)
    db.add(run)
    db.commit()
    db.refresh(run)
    return run, model


async def test_wait_for_idle_counts_persisted_active_leases(db) -> None:
    run, model = _benchmark_run(db)
    lease = InferenceLease(
        request_id=f"benchmark-active-{uuid.uuid4()}",
        model_id=model.id,
        status="active",
        started_at=datetime.now(UTC),
        lease_expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    db.add(lease)
    db.commit()
    run_id = str(run.id)
    control = benchmark._run_controls.setdefault(run_id, benchmark.asyncio.Event())
    control.set()
    benchmark._run_decisions[run_id] = "abort"

    try:
        with (
            patch.object(benchmark.settings, "BENCHMARK_IDLE_TIMEOUT", 0),
            patch.object(benchmark, "_set_run", new=AsyncMock()) as set_run,
        ):
            idle = await benchmark._wait_for_idle(run_id)
    finally:
        benchmark._run_controls.pop(run_id, None)
        benchmark._run_decisions.pop(run_id, None)

    assert idle is False
    set_run.assert_awaited_once_with(run_id, status="idle_timeout")


async def test_force_stop_cancels_idle_timeout_run(db) -> None:
    run, _model = _benchmark_run(db, status="idle_timeout")

    with patch.object(benchmark, "agent_manager") as manager:
        manager.send_to_agent = AsyncMock()
        await benchmark.stop_run(str(run.id), force=True)

    db.expire_all()
    persisted = db.get(BenchmarkRun, run.id)
    assert persisted.status == "cancelled"
    assert persisted.error == "Cancelled by user"
    manager.send_to_agent.assert_not_awaited()


async def test_stop_servers_reports_dispatch_failures() -> None:
    server = ServerInstance(
        id=uuid.uuid4(),
        model_id=uuid.uuid4(),
        agent_id=uuid.uuid4(),
        alias="benchmark-stop-failure",
        process_command="llama-server",
        status="running",
    )
    result = Mock()
    result.scalars.return_value.all.return_value = [server]
    session = AsyncMock()
    session.execute.return_value = result
    session_context = AsyncMock()
    session_context.__aenter__.return_value = session
    session_context.__aexit__.return_value = None

    with (
        patch.object(benchmark, "AsyncSessionMaker", return_value=session_context),
        patch.object(
            benchmark,
            "request_server_stop",
            new=AsyncMock(side_effect=RuntimeError("dispatch failed")),
        ) as stop_server,
        pytest.raises(RuntimeError, match="Unable to stop 1 inference server"),
    ):
        await benchmark._stop_servers()

    stop_server.assert_awaited_once_with(
        server.id,
        force=True,
        reason="benchmark_stop",
    )


async def test_execute_does_not_launch_when_server_stop_fails(db) -> None:
    run, _model = _benchmark_run(db)
    run_id = str(run.id)
    acquired = Mock()
    acquired.scalar_one.return_value = True
    unlocked = Mock()
    connection = AsyncMock()
    connection.execute = AsyncMock(side_effect=[acquired, unlocked])
    connection_context = AsyncMock()
    connection_context.__aenter__.return_value = connection
    connection_context.__aexit__.return_value = None
    engine = Mock()
    engine.connect.return_value = connection_context

    with (
        patch.object(benchmark, "async_engine", engine),
        patch.object(benchmark, "_wait_for_idle", new=AsyncMock(return_value=True)),
        patch.object(
            benchmark,
            "_stop_servers",
            new=AsyncMock(side_effect=RuntimeError("stop dispatch failed")),
        ),
        patch.object(benchmark.agent_manager, "send_to_agent", new=AsyncMock()) as send,
    ):
        await benchmark._execute(run_id)

    db.expire_all()
    persisted = db.get(BenchmarkRun, run.id)
    assert persisted.status == "failed"
    assert persisted.error == "stop dispatch failed"
    send.assert_not_awaited()


async def test_concurrent_execute_claims_benchmark_run_once(db) -> None:
    run, _model = _benchmark_run(db)
    run_id = str(run.id)
    entered_idle_check = asyncio.Event()
    release_worker = asyncio.Event()
    idle_checks = 0

    async def fail_after_claim(_run_id: str) -> bool:
        nonlocal idle_checks
        idle_checks += 1
        entered_idle_check.set()
        await release_worker.wait()
        raise RuntimeError("stop claimed worker")

    with patch.object(benchmark, "_wait_for_idle", new=fail_after_claim):
        first_worker = asyncio.create_task(benchmark._execute(run_id))
        await asyncio.wait_for(entered_idle_check.wait(), timeout=2)
        await benchmark._execute(run_id)
        release_worker.set()
        await first_worker

    db.expire_all()
    assert idle_checks == 1
    assert db.get(BenchmarkRun, run.id).status == "failed"


async def test_execute_releases_benchmark_gate_and_cleans_task_state_on_failure(
    db,
) -> None:
    run, _model = _benchmark_run(db)
    run_id = str(run.id)
    acquired = Mock()
    acquired.scalar_one.return_value = True
    unlocked = Mock()
    connection = AsyncMock()
    connection.execute = AsyncMock(side_effect=[acquired, unlocked])
    connection_context = AsyncMock()
    connection_context.__aenter__.return_value = connection
    connection_context.__aexit__.return_value = None
    engine = Mock()
    engine.connect.return_value = connection_context
    benchmark._run_tasks[run_id] = Mock()
    benchmark._run_controls[run_id] = benchmark.asyncio.Event()

    with (
        patch.object(benchmark, "async_engine", engine),
        patch.object(
            benchmark,
            "_wait_for_idle",
            new=AsyncMock(side_effect=RuntimeError("idle check failed")),
        ),
    ):
        await benchmark._execute(run_id)

    db.expire_all()
    persisted = db.get(BenchmarkRun, run.id)
    assert persisted.status == "failed"
    assert persisted.error == "idle check failed"
    assert connection.execute.await_count == 2
    unlock_statement = str(connection.execute.await_args_list[1].args[0])
    assert "pg_advisory_unlock" in unlock_statement
    assert run_id not in benchmark._run_tasks
    assert run_id not in benchmark._run_controls

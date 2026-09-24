"""Queue and orchestrate persisted llama-bench runs."""

import asyncio
import logging
import uuid as uuid_module
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from app.core.config import settings
from app.db.session import AsyncSessionMaker
from app.models import Agent, BenchmarkDefinition, BenchmarkRun, Model, ServerInstance
from app.services.agent_manager import agent_manager
from app.services.request_activity import active_request_count

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = {"waiting_for_idle", "stopping_servers", "running", "idle_timeout"}
_run_tasks: dict[str, asyncio.Task[None]] = {}
_run_controls: dict[str, asyncio.Event] = {}
_run_decisions: dict[str, str] = {}
_execution_lock = asyncio.Lock()
_queue_task: asyncio.Task[None] | None = None


async def is_benchmark_blocking() -> bool:
    async with AsyncSessionMaker() as session:
        result = await session.execute(
            select(BenchmarkRun.id).where(BenchmarkRun.status.in_(ACTIVE_STATUSES))
        )
        return result.first() is not None


def _definition_dict(
    definition: BenchmarkDefinition,
    source: ServerInstance | None = None,
) -> dict[str, Any]:
    return {
        "id": str(definition.id),
        "name": definition.name,
        "description": definition.description,
        "source_server_instance_id": (
            str(definition.source_server_instance_id)
            if definition.source_server_instance_id
            else None
        ),
        "config": definition.config,
        "created_at": definition.created_at.isoformat(),
        "updated_at": definition.updated_at.isoformat(),
        "server_id": str(source.id) if source else None,
        "server_alias": source.alias if source else None,
    }


def _run_dict(
    run: BenchmarkRun,
    definition: BenchmarkDefinition | None = None,
    agent: Agent | None = None,
) -> dict[str, Any]:
    return {
        "id": str(run.id),
        "definition_id": str(run.definition_id),
        "definition_name": definition.name if definition else None,
        "status": run.status,
        "created_at": run.created_at.isoformat(),
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "error": run.error,
        "results": run.results,
        "agent_id": str(run.agent_id) if run.agent_id else None,
        "agent_name": agent.name if agent else None,
        "server_id": str(run.server_instance_id) if run.server_instance_id else None,
    }


async def _set_run(run_id: str, **values: Any) -> None:
    async with AsyncSessionMaker() as session:
        run = await session.get(BenchmarkRun, uuid_module.UUID(run_id))
        if not run:
            return
        for key, value in values.items():
            setattr(run, key, value)
        session.add(run)
        await session.commit()


async def _wait_for_idle(run_id: str) -> bool:
    deadline = asyncio.get_running_loop().time() + settings.BENCHMARK_IDLE_TIMEOUT
    while True:
        async with AsyncSessionMaker() as session:
            result = await session.execute(
                select(ServerInstance).where(ServerInstance.status == "running")
            )
            running = list(result.scalars().all())
        now = datetime.now(UTC)
        # The active request paths update last_request_at. Keep a short grace
        # period so a request finishing at the boundary is not interrupted.
        busy = [
            server
            for server in running
            if server.last_request_at
            and (now - server.last_request_at).total_seconds() < 5
        ]
        if not busy and await active_request_count() == 0:
            return True
        if asyncio.get_running_loop().time() >= deadline:
            await _set_run(run_id, status="idle_timeout")
            control = _run_controls.setdefault(run_id, asyncio.Event())
            await control.wait()
            return _run_decisions.pop(run_id, "abort") == "force"
        await asyncio.sleep(settings.BENCHMARK_IDLE_POLL_INTERVAL)


async def _stop_servers() -> None:
    async with AsyncSessionMaker() as session:
        result = await session.execute(
            select(ServerInstance).where(
                ServerInstance.status.in_(["starting", "running"])
            )
        )
        servers = list(result.scalars().all())
    for server in servers:
        try:
            await agent_manager.send_to_agent(
                str(server.agent_id),
                "POST",
                "/servers/stop",
                {"server_id": str(server.id)},
            )
        except Exception as exc:  # noqa: BLE001 - one agent must not block cleanup
            logger.warning("Unable to stop server %s: %s", server.id, exc)
        await _set_server_stopped(str(server.id))


async def _set_server_stopped(server_id: str) -> None:
    async with AsyncSessionMaker() as session:
        server = await session.get(ServerInstance, uuid_module.UUID(server_id))
        if server:
            server.status = "stopped"
            server.health_status = "unknown"
            session.add(server)
            await session.commit()


async def _execute(run_id: str) -> None:
    async with _execution_lock:
        try:
            async with AsyncSessionMaker() as session:
                run = await session.get(BenchmarkRun, uuid_module.UUID(run_id))
                if not run:
                    return
                definition = await session.get(BenchmarkDefinition, run.definition_id)
                if not definition:
                    await _set_run(
                        run_id,
                        status="failed",
                        error="Benchmark definition was deleted",
                        finished_at=datetime.now(UTC),
                    )
                    return
                source = (
                    await session.get(
                        ServerInstance, definition.source_server_instance_id
                    )
                    if definition.source_server_instance_id
                    else None
                )
                config = definition.config or {}
                model_id = config.get("model_id") or (
                    source.model_id if source else None
                )
                agent_id = config.get("agent_id") or (
                    source.agent_id if source else None
                )
                model = await session.get(Model, model_id) if model_id else None
                agent = await session.get(Agent, agent_id) if agent_id else None
            if not model or not agent:
                raise RuntimeError("Benchmark source server, model, or agent not found")

            await _set_run(run_id, status="waiting_for_idle")
            if not await _wait_for_idle(run_id):
                await _set_run(
                    run_id,
                    status="cancelled",
                    error="Idle timeout exceeded",
                    finished_at=datetime.now(UTC),
                )
                return

            await _set_run(run_id, status="stopping_servers")
            await _stop_servers()

            while agent.status != "online":
                await asyncio.sleep(settings.BENCHMARK_IDLE_POLL_INTERVAL)
                async with AsyncSessionMaker() as session:
                    refreshed = await session.get(Agent, agent.id)
                    if refreshed:
                        agent = refreshed

            request = {
                "run_id": run_id,
                "model_path": config.get("model_path", model.path),
                "source": (
                    {
                        "source": model.source,
                        "repo_id": model.source_repo_id or "",
                        "filename": model.source_file or model.path.rsplit("/", 1)[-1],
                    }
                    if model.source_repo_id
                    else None
                ),
                "prompt_sizes": config.get("prompt_sizes", [512]),
                "generation_sizes": config.get("generation_sizes", [128]),
                "repetitions": config.get("repetitions", 3),
                "batch_size": config.get("batch_size", 512),
                "ubatch_size": config.get("ubatch_size"),
                "context_size": config.get("context_size"),
                "gpu_layers": config.get("gpu_layers", 35),
                "flash_attn": config.get("flash_attn", True),
            }
            response = await agent_manager.send_to_agent(
                str(agent.id),
                "POST",
                "/benchmarks/run",
                request,
                timeout=900,
            )
            await _set_run(
                run_id,
                status="running",
                agent_id=agent.id,
                server_instance_id=source.id if source else None,
                command=response.get("command", []),
                started_at=datetime.now(UTC),
            )
        except Exception as exc:  # noqa: BLE001 - persist worker failures
            logger.exception("Benchmark %s failed", run_id)
            await _set_run(
                run_id,
                status="failed",
                error=str(exc),
                finished_at=datetime.now(UTC),
            )
        finally:
            _run_tasks.pop(run_id, None)
            _run_controls.pop(run_id, None)


def schedule_run(run_id: str) -> None:
    _run_tasks[run_id] = asyncio.create_task(_execute(run_id))


async def _queue_loop() -> None:
    """Resume persisted queued runs after startup or a worker interruption."""
    while True:
        try:
            async with AsyncSessionMaker() as session:
                result = await session.execute(
                    select(BenchmarkRun.id)
                    .where(BenchmarkRun.status == "queued")
                    .order_by(BenchmarkRun.created_at)
                )
                queued_ids = [str(row[0]) for row in result.all()]
            for run_id in queued_ids:
                if run_id not in _run_tasks:
                    schedule_run(run_id)
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Benchmark queue dispatcher failed")
            await asyncio.sleep(2)


def start_queue_worker() -> None:
    global _queue_task
    if _queue_task is None or _queue_task.done():
        _queue_task = asyncio.create_task(_queue_loop())


async def stop_queue_worker() -> None:
    if _queue_task and not _queue_task.done():
        _queue_task.cancel()
        await asyncio.gather(_queue_task, return_exceptions=True)


def resolve_timeout(run_id: str, decision: str) -> None:
    _run_decisions[run_id] = decision
    control = _run_controls.setdefault(run_id, asyncio.Event())
    control.set()


async def stop_run(run_id: str, *, force: bool = False) -> None:
    del force
    async with AsyncSessionMaker() as session:
        run = await session.get(BenchmarkRun, uuid_module.UUID(run_id))
        if not run:
            return
        status = run.status
        agent_id = run.agent_id
    if agent_id:
        await agent_manager.send_to_agent(
            str(agent_id),
            "POST",
            "/benchmarks/stop",
            {"run_id": run_id},
        )
    if status in {"queued", "waiting_for_idle", "idle_timeout"}:
        await _set_run(
            run_id,
            status="cancelled",
            error="Cancelled by user",
            finished_at=datetime.now(UTC),
        )

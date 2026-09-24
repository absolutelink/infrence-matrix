"""Persisted benchmark definition and run endpoints."""

import uuid as uuid_module
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.session import AsyncSessionMaker
from app.models import Agent, BenchmarkDefinition, BenchmarkRun, Model, ServerInstance
from app.services.benchmark import (
    _definition_dict,
    _run_dict,
    resolve_timeout,
    schedule_run,
    stop_run,
)

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


class BenchmarkDefinitionInput(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    source_server_instance_id: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class RunInput(BaseModel):
    definition_id: str


def _source_id(request: BenchmarkDefinitionInput) -> uuid_module.UUID | None:
    value = request.source_server_instance_id
    if not value:
        return None
    try:
        return uuid_module.UUID(str(value))
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="Invalid server instance ID"
        ) from exc


async def _snapshot(
    session: Any,
    source: ServerInstance | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    model_id = config.get("model_id") or (source.model_id if source else None)
    agent_id = config.get("agent_id") or (source.agent_id if source else None)
    if not model_id or not agent_id:
        raise HTTPException(
            status_code=422,
            detail="Select a model and agent, or copy from a server instance",
        )
    model = await session.get(Model, uuid_module.UUID(str(model_id)))
    if not model:
        raise HTTPException(status_code=404, detail="Source model not found")
    snapshot = {
        "server_instance_id": str(source.id) if source else None,
        "model_id": str(model.id),
        "agent_id": str(agent_id),
        "model_path": model.path,
        "gpu_layers": source.gpu_layers if source else 35,
        "context_size": source.context_size if source else 4096,
        "flash_attn": source.flash_attn if source else True,
        "mtp_draft_max": source.mtp_draft_max if source else None,
        "mmproj_model_id": source.mmproj_model_id if source else None,
    }
    snapshot.update(config)
    return snapshot


@router.get("/definitions")
async def list_definitions() -> list[dict[str, Any]]:
    async with AsyncSessionMaker() as session:
        result = await session.execute(
            select(BenchmarkDefinition).order_by(BenchmarkDefinition.updated_at.desc())
        )
        definitions = list(result.scalars().all())
        output = []
        for definition in definitions:
            source = (
                await session.get(ServerInstance, definition.source_server_instance_id)
                if definition.source_server_instance_id
                else None
            )
            output.append(_definition_dict(definition, source))
        return output


@router.post("/definitions")
async def create_definition(request: BenchmarkDefinitionInput) -> dict[str, Any]:
    source_id = _source_id(request)
    async with AsyncSessionMaker() as session:
        source = await session.get(ServerInstance, source_id) if source_id else None
        if request.source_server_instance_id and not source:
            raise HTTPException(status_code=404, detail="Source server not found")
        definition = BenchmarkDefinition(
            name=request.name.strip(),
            description=request.description,
            source_server_instance_id=source.id if source else None,
            config=await _snapshot(session, source, request.config),
        )
        session.add(definition)
        await session.commit()
        await session.refresh(definition)
        return _definition_dict(definition, source)


@router.patch("/definitions/{definition_id}")
async def update_definition(
    definition_id: str, request: BenchmarkDefinitionInput
) -> dict[str, Any]:
    async with AsyncSessionMaker() as session:
        definition = await session.get(BenchmarkDefinition, definition_id)
        if not definition:
            raise HTTPException(
                status_code=404, detail="Benchmark definition not found"
            )
        source_id = _source_id(request)
        source = await session.get(ServerInstance, source_id) if source_id else None
        if request.source_server_instance_id and not source:
            raise HTTPException(status_code=404, detail="Source server not found")
        definition.name = request.name.strip()
        definition.description = request.description
        definition.source_server_instance_id = source.id if source else None
        definition.config = await _snapshot(session, source, request.config)
        definition.updated_at = datetime.now(UTC)
        session.add(definition)
        await session.commit()
        await session.refresh(definition)
        return _definition_dict(definition, source)


@router.delete("/definitions/{definition_id}", status_code=204)
async def delete_definition(definition_id: str) -> None:
    async with AsyncSessionMaker() as session:
        definition = await session.get(BenchmarkDefinition, definition_id)
        if not definition:
            raise HTTPException(
                status_code=404, detail="Benchmark definition not found"
            )
        await session.delete(definition)
        await session.commit()


@router.get("/runs")
async def list_runs() -> list[dict[str, Any]]:
    async with AsyncSessionMaker() as session:
        result = await session.execute(
            select(BenchmarkRun).order_by(BenchmarkRun.created_at.desc())
        )
        runs = list(result.scalars().all())
        output = []
        for run in runs:
            definition = await session.get(BenchmarkDefinition, run.definition_id)
            agent = await session.get(Agent, run.agent_id) if run.agent_id else None
            output.append(_run_dict(run, definition, agent))
        return output


@router.post("/runs")
async def create_run(request: RunInput) -> dict[str, Any]:
    async with AsyncSessionMaker() as session:
        definition = await session.get(BenchmarkDefinition, request.definition_id)
        if not definition:
            raise HTTPException(
                status_code=404, detail="Benchmark definition not found"
            )
        source = (
            await session.get(ServerInstance, definition.source_server_instance_id)
            if definition.source_server_instance_id
            else None
        )
        agent_id = (definition.config or {}).get("agent_id")
        run = BenchmarkRun(
            definition_id=definition.id,
            agent_id=uuid_module.UUID(str(agent_id))
            if agent_id
            else (source.agent_id if source else None),
            server_instance_id=source.id if source else None,
            status="queued",
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        response = _run_dict(run, definition)
        schedule_run(str(run.id))
        return response


@router.get("/runs/{run_id}/results")
async def get_results(run_id: str) -> dict[str, Any] | None:
    async with AsyncSessionMaker() as session:
        run = await session.get(BenchmarkRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Benchmark run not found")
        return run.results


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict[str, Any]:
    await stop_run(run_id)
    return await _get_run_response(run_id)


@router.post("/runs/{run_id}/abort")
async def abort_run(run_id: str) -> dict[str, Any]:
    resolve_timeout(run_id, "abort")
    await stop_run(run_id)
    return await _get_run_response(run_id)


@router.post("/runs/{run_id}/force-stop")
async def force_stop_run(run_id: str) -> dict[str, Any]:
    resolve_timeout(run_id, "force")
    await stop_run(run_id, force=True)
    return await _get_run_response(run_id)


async def _get_run_response(run_id: str) -> dict[str, Any]:
    async with AsyncSessionMaker() as session:
        run = await session.get(BenchmarkRun, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Benchmark run not found")
        definition = await session.get(BenchmarkDefinition, run.definition_id)
        agent = await session.get(Agent, run.agent_id) if run.agent_id else None
        return _run_dict(run, definition, agent)

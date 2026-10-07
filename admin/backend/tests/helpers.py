"""Shared test builders for the Phase 16 agent model.

A "stack" is Machine -> ProviderAgent -> ProviderInstance (one backend) for a
ProviderDefinition. Agents hold the container-level connection state
(websocket_connected/epoch/version); instances are the per-backend rows the
scheduler and lifecycle act on.
"""

import uuid as _uuid
from typing import Any

from sqlmodel import Session, select

from app.models import (
    Machine,
    ProviderAgent,
    ProviderDefinition,
    ProviderInstance,
)


def get_or_create_machine(
    session: Session,
    *,
    uid: str,
    name: str | None = None,
    host: str = "127.0.0.1",
    total_vram: int = 0,
    registration_secret: str = "test-secret",
) -> Machine:
    machine = session.exec(select(Machine).where(Machine.uid == uid)).first()
    if machine is None:
        machine = Machine(
            uid=uid,
            name=name or f"name-{uid}",
            host=host,
            total_vram_bytes=total_vram,
            registration_secret=registration_secret,
        )
        session.add(machine)
    else:
        if total_vram:
            machine.total_vram_bytes = total_vram
        session.add(machine)
    session.commit()
    session.refresh(machine)
    return machine


def make_agent(
    session: Session,
    machine: Machine,
    *,
    provider_type: str = "mock",
    agent_id: str | None = None,
    base_port: int = 8081,
    version: str = "dev",
    connected: bool = True,
    epoch: int = 0,
    agent_status: str = "running",
    **kwargs: Any,
) -> ProviderAgent:
    agent = ProviderAgent(
        machine_id=machine.id,
        provider_type=provider_type,
        agent_id=agent_id or f"agent-{_uuid.uuid4().hex[:8]}",
        base_port=base_port,
        version=version,
        websocket_connected=connected,
        epoch=epoch,
        agent_status=agent_status,
        **kwargs,
    )
    session.add(agent)
    session.commit()
    session.refresh(agent)
    return agent


def make_definition(
    session: Session,
    *,
    alias: str,
    provider_type: str = "mock",
    backend_config: dict[str, Any] | None = None,
    vram_required: int = 0,
    capacity: int = 1,
    idle_timeout_seconds: int = 300,
    enabled: bool = True,
    **kwargs: Any,
) -> ProviderDefinition:
    definition = ProviderDefinition(
        alias=alias,
        provider_type=provider_type,
        backend_config=backend_config if backend_config is not None else {},
        vram_required_bytes=vram_required,
        capacity=capacity,
        idle_timeout_seconds=idle_timeout_seconds,
        enabled=enabled,
        **kwargs,
    )
    session.add(definition)
    session.commit()
    session.refresh(definition)
    return definition


def make_instance(
    session: Session,
    agent: ProviderAgent,
    definition: ProviderDefinition,
    *,
    port: int | None = None,
    backend_status: str = "stopped",
    **kwargs: Any,
) -> ProviderInstance:
    instance = ProviderInstance(
        agent_id=agent.id,
        provider_definition_id=definition.id,
        port=port if port is not None else agent.base_port,
        backend_status=backend_status,
        **kwargs,
    )
    session.add(instance)
    session.commit()
    session.refresh(instance)
    return instance


def make_stack(
    session: Session,
    *,
    machine_uid: str,
    alias: str,
    provider_type: str = "mock",
    total_vram: int = 1000,
    vram_required: int = 0,
    capacity: int = 1,
    backend_status: str = "stopped",
    connected: bool = True,
    port: int = 8081,
    backend_config: dict[str, Any] | None = None,
) -> tuple[Machine, ProviderAgent, ProviderDefinition, ProviderInstance]:
    """One machine + one agent + one definition + one backend."""
    machine = get_or_create_machine(session, uid=machine_uid, total_vram=total_vram)
    agent = make_agent(
        session,
        machine,
        provider_type=provider_type,
        base_port=port,
        connected=connected,
    )
    definition = make_definition(
        session,
        alias=alias,
        provider_type=provider_type,
        backend_config=(
            backend_config
            if backend_config is not None
            else {"model": {"file": "m.gguf"}}
        ),
        vram_required=vram_required,
        capacity=capacity,
    )
    instance = make_instance(
        session,
        agent,
        definition,
        port=port,
        backend_status=backend_status,
    )
    return machine, agent, definition, instance

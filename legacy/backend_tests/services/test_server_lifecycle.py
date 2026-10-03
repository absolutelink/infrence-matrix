"""Focused tests for generation-fenced server lifecycle transitions."""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from app.models import Agent, InferenceLease, Model, ServerInstance
from app.services.agent_manager import AgentManager
from app.services.server_lifecycle import ServerBusyError, request_server_stop


def _server(db, *, status: str = "running", generation: int = 4) -> ServerInstance:
    suffix = uuid.uuid4().hex
    model = Model(
        name=f"lifecycle-{suffix}.gguf",
        path=f"/models/lifecycle-{suffix}.gguf",
        size_bytes=100,
        architecture="llama",
        quantization="Q4_K_M",
        source="huggingface",
    )
    agent = Agent(
        name=f"lifecycle-agent-{suffix}",
        host="localhost",
        port=8080,
        status="online",
    )
    db.add(model)
    db.add(agent)
    db.flush()
    server = ServerInstance(
        model_id=model.id,
        agent_id=agent.id,
        alias=f"lifecycle-server-{suffix}",
        process_command="llama-server",
        status=status,
        health_status="healthy",
        slot_generation=generation,
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    return server


def _active_lease(
    db, server: ServerInstance, *, generation: int | None = None
) -> InferenceLease:
    lease = InferenceLease(
        request_id=f"lifecycle-request-{uuid.uuid4()}",
        model_id=server.model_id,
        server_instance_id=server.id,
        status="active",
        started_at=datetime.now(UTC),
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        slot_generation=(server.slot_generation if generation is None else generation),
    )
    db.add(lease)
    db.commit()
    db.refresh(lease)
    return lease


@pytest.mark.parametrize(
    ("handler_name", "initial_status"),
    [
        ("_handle_server_started", "starting"),
        ("_handle_server_stopped", "running"),
        ("_handle_server_error", "running"),
    ],
)
async def test_server_events_ignore_stale_generations(
    db, handler_name: str, initial_status: str
) -> None:
    server = _server(db, status=initial_status)
    lease = _active_lease(db, server)
    manager = AgentManager()
    handler = getattr(manager, handler_name)

    await handler(
        str(server.agent_id),
        {
            "server_id": str(server.id),
            "slot_generation": server.slot_generation - 1,
            "error": "stale failure",
            "effective_capacity": 9,
        },
    )

    db.expire_all()
    persisted_server = db.get(ServerInstance, server.id)
    persisted_lease = db.get(InferenceLease, lease.id)
    assert persisted_server.status == initial_status
    assert persisted_server.health_status == "healthy"
    assert persisted_server.error_message is None
    assert persisted_server.effective_capacity == 1
    assert persisted_lease.status == "active"


async def test_graceful_stop_rejects_server_with_active_lease(db) -> None:
    server = _server(db)
    lease = _active_lease(db, server)

    with patch(
        "app.services.server_lifecycle.agent_manager.send_to_agent",
        new=AsyncMock(),
    ) as send_to_agent:
        with pytest.raises(ServerBusyError, match="1 active inference request"):
            await request_server_stop(
                server.id, force=False, reason="graceful_test_stop"
            )

    db.expire_all()
    assert db.get(ServerInstance, server.id).status == "running"
    assert db.get(InferenceLease, lease.id).status == "active"
    send_to_agent.assert_not_awaited()


async def test_forced_stop_terminalizes_all_server_leases_and_dispatches_generation(
    db,
) -> None:
    server = _server(db)
    current_lease = _active_lease(db, server)
    stale_lease = _active_lease(db, server, generation=server.slot_generation - 1)

    with patch(
        "app.services.server_lifecycle.agent_manager.send_to_agent",
        new=AsyncMock(return_value={"status": "stopped"}),
    ) as send_to_agent:
        stopped = await request_server_stop(
            server.id, force=True, reason="forced_test_stop"
        )

    db.expire_all()
    current = db.get(InferenceLease, current_lease.id)
    stale = db.get(InferenceLease, stale_lease.id)
    assert stopped.status == "stopped"
    assert current.status == "failed"
    assert current.terminal_reason == "forced_test_stop"
    assert current.released_at is not None
    assert stale.status == "failed"
    assert stale.terminal_reason == "forced_test_stop"
    send_to_agent.assert_awaited_once_with(
        str(server.agent_id),
        "POST",
        "/servers/stop",
        {
            "server_id": str(server.id),
            "slot_generation": server.slot_generation,
        },
        timeout=60.0,
    )


async def test_stop_dispatch_failure_marks_server_error(db) -> None:
    server = _server(db)

    with patch(
        "app.services.server_lifecycle.agent_manager.send_to_agent",
        new=AsyncMock(side_effect=RuntimeError("agent unavailable")),
    ):
        with pytest.raises(RuntimeError, match="agent unavailable"):
            await request_server_stop(server.id, force=True, reason="stop_failure")

    db.expire_all()
    persisted = db.get(ServerInstance, server.id)
    assert persisted.status == "error"
    assert persisted.error_message == "agent unavailable"

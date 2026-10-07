"""Manual backend control + reinitialize routes (provider_lib.ops surface).

Covers the admin half of the contract (Phase 16: commands are addressed to the
owning agent and carry the target ``instance_id``):

* the 202 accept-style defaults (``wait_for_running: false`` + the short ack
  window) versus the opt-in blocking form (boot budget),
* provider NAKs surfacing as 502 with ``step`` / ``retry_after``,
* ``backend.metadata`` ingest onto the definition,
* the registration port-clash refusal that keeps two backends on one machine
  addressable as separate targets.

Routes are called directly (no HTTP layer) with the ack intercepted, so the
command payload — the part the provider contract actually depends on — is
asserted exactly.
"""

import uuid
from typing import Any

import pytest
from sqlmodel import Session

from app.api.admin import instances as routes
from app.api.admin import providers as provider_routes
from app.api.admin.instances import BackendActionBody, _send_action
from app.models import (
    Machine,
    ProviderAgent,
    ProviderDefinition,
    ProviderInstance,
)
from app.services.connection_manager import manager
from app.services.wire import Frame


def _seed(
    session: Session,
    *,
    connected: bool = True,
    port: int = 8083,
    machine_uid: str = "mach-actions",
) -> tuple[ProviderInstance, ProviderDefinition, Machine]:
    machine = Machine(
        uid=machine_uid,
        name=f"name-{machine_uid}-{uuid.uuid4().hex[:8]}",
        host="10.0.0.5",
        dns="box.local",
    )
    session.add(machine)
    session.commit()
    session.refresh(machine)
    agent = ProviderAgent(
        machine_id=machine.id,
        provider_type="halogen-flash",
        agent_id=f"agent-{uuid.uuid4().hex[:8]}",
        base_port=port,
        websocket_connected=connected,
        agent_status="running",
    )
    session.add(agent)
    session.commit()
    session.refresh(agent)
    definition = ProviderDefinition(
        alias=f"alias-{uuid.uuid4().hex[:8]}",
        provider_type="halogen-flash",
        backend_config={},
        vram_required_bytes=0,
        idle_timeout_seconds=300,
        capacity=1,
        enabled=True,
    )
    session.add(definition)
    session.commit()
    session.refresh(definition)
    inst = ProviderInstance(
        agent_id=agent.id,
        provider_definition_id=definition.id,
        port=port,
        backend_status="stopped",
    )
    session.add(inst)
    session.commit()
    session.refresh(inst)
    return inst, definition, machine


class _AckCapture:
    """Stand-in for ConnectionManager.send_command (agent-addressed)."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str, dict[str, Any], float | None]] = []

    async def __call__(
        self,
        agent_id: str,
        type_: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> Frame:
        self.calls.append((agent_id, type_, payload, timeout))
        if "raise" in self.payload:
            raise RuntimeError(self.payload["raise"])
        return Frame(type="ack", id="a1", payload=dict(self.payload))

    @property
    def last(self) -> tuple[str, str, dict[str, Any], float | None]:
        return self.calls[-1]


@pytest.fixture
def ack(monkeypatch: pytest.MonkeyPatch) -> _AckCapture:
    capture = _AckCapture({"ok": True, "detail": {"backend_status": "initializing"}})
    monkeypatch.setattr(manager, "send_command", capture)
    return capture


# ---------------------------------------------------------------------------
# backend start / stop / restart
# ---------------------------------------------------------------------------


async def test_start_acks_accepted_with_the_short_window(
    session: Session, ack: _AckCapture
) -> None:
    """Default 202 path: `wait_for_running: false`, and the ack window is the
    action timeout — the provider starts the boot in the background."""
    inst, _definition, _machine = _seed(session)
    out = await routes.start_backend(str(inst.id), None, session)
    assert out["ok"] is True and out["backend_status"] == "initializing"
    agent_id, command, payload, timeout = ack.last
    assert agent_id == str(inst.agent_id)
    assert command == "backend.start"
    assert payload == {"wait_for_running": False, "instance_id": str(inst.id)}
    assert timeout == routes.ACTION_TIMEOUT_SECONDS


async def test_start_can_block_on_the_boot_budget(
    session: Session, ack: _AckCapture
) -> None:
    from app.core.config import settings

    inst, _definition, _machine = _seed(session)
    await routes.start_backend(
        str(inst.id), BackendActionBody(wait_for_running=True), session
    )
    _id, _command, payload, timeout = ack.last
    assert payload == {"wait_for_running": True, "instance_id": str(inst.id)}
    assert timeout == settings.BACKEND_BOOT_TIMEOUT_SECONDS


async def test_start_requires_a_live_socket(session: Session) -> None:
    from fastapi import HTTPException

    inst, _definition, _machine = _seed(session, connected=False)
    with pytest.raises(HTTPException) as exc:
        await routes.start_backend(str(inst.id), None, session)
    assert exc.value.status_code == 409


async def test_stop_always_awaits(session: Session, ack: _AckCapture) -> None:
    from app.core.config import settings

    inst, _definition, _machine = _seed(session)
    out = await routes.stop_backend(str(inst.id), session)
    assert out["ok"] is True
    _id, command, payload, timeout = ack.last
    assert command == "backend.stop"
    assert payload == {"instance_id": str(inst.id)}
    assert timeout == settings.BACKEND_STOP_TIMEOUT_SECONDS


async def test_restart_accepts(session: Session, ack: _AckCapture) -> None:
    inst, _definition, _machine = _seed(session)
    out = await routes.restart_backend(str(inst.id), None, session)
    assert out["ok"] is True
    assert ack.last[1] == "backend.restart"
    assert ack.last[2] == {"wait_for_running": False, "instance_id": str(inst.id)}


# ---------------------------------------------------------------------------
# provider.initialize
# ---------------------------------------------------------------------------


async def test_initialize_accepts_and_reports(
    session: Session, ack: _AckCapture
) -> None:
    inst, _definition, _machine = _seed(session)
    ack.payload = {"ok": True, "detail": {"accepted": True}}
    out = await routes.initialize_instance(str(inst.id), None, session)
    assert out["accepted"] is True
    assert ack.last[1] == "provider.initialize"
    assert ack.last[2] == {"wait_for_running": False, "instance_id": str(inst.id)}


async def test_initialize_nak_becomes_502_with_retry_after(
    session: Session, ack: _AckCapture
) -> None:
    from fastapi import HTTPException

    inst, _definition, _machine = _seed(session)
    ack.payload = {
        "ok": False,
        "error": "boot_in_progress",
        "detail": {"step": "initialize", "retry_after": 10},
    }
    with pytest.raises(HTTPException) as exc:
        await routes.initialize_instance(str(inst.id), None, session)
    assert exc.value.status_code == 502
    detail = exc.value.detail
    assert detail["error"] == "boot_in_progress"
    assert detail["step"] == "initialize"
    assert detail["retry_after"] == 10


async def test_send_action_maps_a_delivery_failure_to_502(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException

    inst, _definition, _machine = _seed(session)

    async def boom(*_a: Any, **_k: Any) -> Frame:
        raise RuntimeError("websocket is not connected")

    monkeypatch.setattr(manager, "send_command", boom)
    with pytest.raises(HTTPException) as exc:
        await _send_action(inst, "backend.start", {})
    assert exc.value.status_code == 502
    assert "delivery failed" in str(exc.value.detail)


# ---------------------------------------------------------------------------
# backend.metadata ingest
# ---------------------------------------------------------------------------


async def test_backend_metadata_persists_on_the_definition(
    session: Session,
) -> None:
    inst, definition, _machine = _seed(session)
    manager._persist_model_metadata(
        str(inst.id),
        {"models": [{"id": "rocinante", "object": "model"}]},
        agent_id=str(inst.agent_id),
    )
    session.expire_all()
    fresh = session.get(ProviderDefinition, definition.id)
    assert fresh is not None
    assert fresh.model_metadata == {"models": [{"id": "rocinante", "object": "model"}]}


async def test_backend_metadata_ignores_empty_and_unknown(
    session: Session,
) -> None:
    inst, definition, _machine = _seed(session)
    manager._persist_model_metadata(
        str(inst.id), {"models": []}, agent_id=str(inst.agent_id)
    )
    manager._persist_model_metadata(
        str(uuid.uuid4()), {"models": [{"id": "x"}]}, agent_id=str(inst.agent_id)
    )
    session.expire_all()
    fresh = session.get(ProviderDefinition, definition.id)
    assert fresh is not None
    # Unset means the column default ({}): an empty or unknown-instance
    # event must not create a phantom roster.
    assert not fresh.model_metadata


# ---------------------------------------------------------------------------
# registration port clash (agent-scoped)
# ---------------------------------------------------------------------------


def _peer_agent(
    session: Session, machine: Machine, *, connected: bool
) -> ProviderAgent:
    agent = ProviderAgent(
        machine_id=machine.id,
        provider_type="mock",
        agent_id=f"peer-{uuid.uuid4().hex[:8]}",
        websocket_connected=connected,
    )
    session.add(agent)
    session.commit()
    session.refresh(agent)
    return agent


def test_port_clash_rejects_a_connected_peer(session: Session) -> None:
    from fastapi import HTTPException

    inst, _definition, machine = _seed(session, port=8083)
    agent = session.get(ProviderAgent, inst.agent_id)
    peer_agent = _peer_agent(session, machine, connected=True)
    other_def = ProviderDefinition(
        alias=f"other-{uuid.uuid4().hex[:6]}",
        provider_type="mock",
        backend_config={},
        vram_required_bytes=0,
        idle_timeout_seconds=300,
        capacity=1,
        enabled=True,
    )
    session.add(other_def)
    session.commit()
    session.refresh(other_def)
    session.add(
        ProviderInstance(
            agent_id=peer_agent.id,
            provider_definition_id=other_def.id,
            port=8083,
        )
    )
    session.commit()

    with pytest.raises(HTTPException) as exc:
        provider_routes._reject_port_clash(session, agent, [8083])
    assert exc.value.status_code == 409
    detail = exc.value.detail
    assert detail["error"] == "port_conflict"
    assert detail["port"] == 8083
    assert other_def.alias in detail["message"]


def test_port_clash_ignores_disconnected_peer(session: Session) -> None:
    inst, _definition, machine = _seed(session, port=8083)
    agent = session.get(ProviderAgent, inst.agent_id)
    peer_agent = _peer_agent(session, machine, connected=False)
    other_def = ProviderDefinition(
        alias=f"stale-{uuid.uuid4().hex[:6]}",
        provider_type="mock",
        backend_config={},
        vram_required_bytes=0,
        idle_timeout_seconds=300,
        capacity=1,
        enabled=True,
    )
    session.add(other_def)
    session.commit()
    session.refresh(other_def)
    session.add(
        ProviderInstance(
            agent_id=peer_agent.id,
            provider_definition_id=other_def.id,
            port=8083,
        )
    )
    session.commit()
    # Disconnected peer holds no port: no clash.
    provider_routes._reject_port_clash(session, agent, [8083])


def test_port_clash_is_scoped_to_the_machine(session: Session) -> None:
    inst, _definition, _machine = _seed(session, port=8083, machine_uid="mach-a")
    agent = session.get(ProviderAgent, inst.agent_id)
    other = Machine(
        uid="mach-b",
        name=f"mach-b-{uuid.uuid4().hex[:8]}",
        host="10.0.0.6",
        dns="other.local",
    )
    session.add(other)
    session.commit()
    session.refresh(other)
    peer_agent = _peer_agent(session, other, connected=True)
    other_def = ProviderDefinition(
        alias=f"other-mach-{uuid.uuid4().hex[:6]}",
        provider_type="mock",
        backend_config={},
        vram_required_bytes=0,
        idle_timeout_seconds=300,
        capacity=1,
        enabled=True,
    )
    session.add(other_def)
    session.commit()
    session.refresh(other_def)
    session.add(
        ProviderInstance(
            agent_id=peer_agent.id,
            provider_definition_id=other_def.id,
            port=8083,
        )
    )
    session.commit()
    # Same port on another machine must not collide with mach-a.
    provider_routes._reject_port_clash(session, agent, [8083])

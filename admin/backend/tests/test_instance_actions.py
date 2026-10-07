"""Manual backend control + reinitialize routes (provider_lib.ops surface).

Covers the admin half of the contract:

* the shell fence (no authored `backend_config` = 409, never a 502 from the
  far end),
* the 202 accept-style defaults (`wait_for_running: false` + the short ack
  window) versus the opt-in blocking form (boot budget),
* provider NAKs surfacing as 502 with `step` / `retry_after`,
* `backend.metadata` ingest onto the definition,
* the registration port-clash refusal that keeps two instances on one
  machine addressable as separate targets.

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
    ProviderDefinition,
    ProviderInstance,
)
from app.services.connection_manager import manager
from app.services.wire import Frame


def _seed(
    session: Session,
    *,
    connected: bool = True,
    authored: bool = True,
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
    definition = ProviderDefinition(
        alias=f"alias-{uuid.uuid4().hex[:8]}",
        provider_type="halogen-flash",
        registration_token=f"tok-{uuid.uuid4().hex[:8]}",
        backend_config={} if authored else None,
        vram_required_bytes=0,
        idle_timeout_seconds=300,
        capacity=1,
        enabled=True,
    )
    session.add(definition)
    session.commit()
    session.refresh(definition)
    inst = ProviderInstance(
        machine_id=machine.id,
        provider_definition_id=definition.id,
        port=port,
        websocket_connected=connected,
        instance_status="running",
        backend_status="stopped",
    )
    session.add(inst)
    session.commit()
    session.refresh(inst)
    return inst, definition, machine


class _AckCapture:
    """Stand-in for ConnectionManager.send_command."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str, dict[str, Any], float | None]] = []

    async def __call__(
        self,
        instance_id: str,
        type_: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> Frame:
        self.calls.append((instance_id, type_, payload, timeout))
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
    _id, command, payload, timeout = ack.last
    assert command == "backend.start"
    assert payload == {"wait_for_running": False}
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
    assert payload == {"wait_for_running": True}
    assert timeout == settings.BACKEND_BOOT_TIMEOUT_SECONDS


async def test_start_refuses_a_shell_definition(
    session: Session, ack: _AckCapture
) -> None:
    """Phase 14: an unauthored config is a 409 here, not a `no_config` NAK
    round-trip (and the provider would refuse it anyway)."""
    from fastapi import HTTPException

    inst, definition, _machine = _seed(session, authored=False)
    with pytest.raises(HTTPException) as exc:
        await routes.start_backend(str(inst.id), None, session)
    assert exc.value.status_code == 409
    assert "backend_config" in str(exc.value.detail)
    assert ack.calls == []
    assert definition.backend_config is None


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
    assert payload == {}
    assert timeout == settings.BACKEND_STOP_TIMEOUT_SECONDS
    # Stop is allowed on a shell too (nothing to boot, and unloading is safe).
    shell = _seed(session, authored=False, machine_uid="mach-actions-shell")[0]
    await routes.stop_backend(str(shell.id), session)


async def test_restart_fences_and_accepts(session: Session, ack: _AckCapture) -> None:
    from fastapi import HTTPException

    inst, _definition, _machine = _seed(session)
    out = await routes.restart_backend(str(inst.id), None, session)
    assert out["ok"] is True
    assert ack.last[1] == "backend.restart"
    assert ack.last[2] == {"wait_for_running": False}

    shell = _seed(session, authored=False, machine_uid="mach-actions-shell")[0]
    with pytest.raises(HTTPException):
        await routes.restart_backend(str(shell.id), None, session)


# ---------------------------------------------------------------------------
# provider.initialize
# ---------------------------------------------------------------------------


async def test_initialize_boots_a_shell_refresh_but_never_rejects_it(
    session: Session, ack: _AckCapture
) -> None:
    """`initialize` is the escape hatch for an instance stuck in
    `awaiting_config`: the route must not fence it (the provider reports
    `no_config` in the ack instead of booting on defaults)."""
    inst, _definition, _machine = _seed(session, authored=False)
    ack.payload = {"ok": True, "detail": {"accepted": True, "no_config": True}}
    out = await routes.initialize_instance(str(inst.id), None, session)
    assert out["no_config"] is True
    assert ack.last[1] == "provider.initialize"


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException

    async def boom(*_a: Any, **_k: Any) -> Frame:
        raise RuntimeError("websocket is not connected")

    monkeypatch.setattr(manager, "send_command", boom)
    with pytest.raises(HTTPException) as exc:
        await _send_action(str(uuid.uuid4()), "backend.start", {})
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
        str(inst.id), {"models": [{"id": "rocinante", "object": "model"}]}
    )
    session.expire_all()
    fresh = session.get(ProviderDefinition, definition.id)
    assert fresh is not None
    assert fresh.model_metadata == {"models": [{"id": "rocinante", "object": "model"}]}


async def test_backend_metadata_ignores_empty_and_unknown(
    session: Session,
) -> None:
    inst, definition, _machine = _seed(session)
    manager._persist_model_metadata(str(inst.id), {"models": []})
    manager._persist_model_metadata(str(uuid.uuid4()), {"models": [{"id": "x"}]})
    session.expire_all()
    fresh = session.get(ProviderDefinition, definition.id)
    assert fresh is not None
    # Unset means the column default ({}): an empty or unknown-instance
    # event must not create a phantom roster.
    assert not fresh.model_metadata


# ---------------------------------------------------------------------------
# registration port clash
# ---------------------------------------------------------------------------


def test_port_clash_rejects_a_different_connected_instance(
    session: Session,
) -> None:
    from fastapi import HTTPException

    inst, definition, machine = _seed(session, port=8083)
    other_def = ProviderDefinition(
        alias=f"other-{uuid.uuid4().hex[:6]}",
        provider_type="mock",
        registration_token=f"tok-{uuid.uuid4().hex[:8]}",
        backend_config={},
        vram_required_bytes=0,
        idle_timeout_seconds=300,
        capacity=1,
        enabled=True,
    )
    session.add(other_def)
    session.commit()
    session.refresh(other_def)
    peer = ProviderInstance(
        machine_id=machine.id,
        provider_definition_id=other_def.id,
        port=8083,
        websocket_connected=True,
    )
    session.add(peer)
    session.commit()

    with pytest.raises(HTTPException) as exc:
        provider_routes._reject_port_clash(session, machine.id, definition.id, 8083)
    assert exc.value.status_code == 409
    detail = exc.value.detail
    assert detail["error"] == "port_conflict"
    assert detail["port"] == 8083
    assert other_def.alias in detail["message"]


def test_port_clash_ignores_disconnected_and_self(session: Session) -> None:
    inst, definition, machine = _seed(session, port=8083)
    # Same definition re-registering (the provider.initialize path): self.
    provider_routes._reject_port_clash(session, machine.id, definition.id, 8083)
    # A different definition, same port, but NOT connected: free port.
    other_def = ProviderDefinition(
        alias=f"stale-{uuid.uuid4().hex[:6]}",
        provider_type="mock",
        registration_token=f"tok-{uuid.uuid4().hex[:8]}",
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
            machine_id=machine.id,
            provider_definition_id=other_def.id,
            port=8083,
            websocket_connected=False,
        )
    )
    session.commit()
    provider_routes._reject_port_clash(session, machine.id, definition.id, 8083)
    assert inst.port == 8083


def test_port_clash_is_scoped_to_the_machine(session: Session) -> None:
    inst, definition, _machine = _seed(session, port=8083, machine_uid="mach-a")
    other = Machine(
        uid="mach-b",
        name=f"mach-b-{uuid.uuid4().hex[:8]}",
        host="10.0.0.6",
        dns="other.local",
    )
    session.add(other)
    session.commit()
    session.refresh(other)
    session.add(
        ProviderInstance(
            machine_id=other.id,
            provider_definition_id=definition.id,
            port=8083,
            websocket_connected=True,
        )
    )
    session.commit()
    # Same definition row on another machine must not collide with mach-a.
    provider_routes._reject_port_clash(session, inst.machine_id, definition.id, 8083)

"""Phase 16 slice 5: ``agent.assignments.update`` push + placement reconcile.

Covers the admin half of the event-driven placement propagation:
``app.services.assignments`` reconciles an agent's ``ProviderInstance`` rows
against its resolved placement and pushes the new set over the agent socket.
The provider WS is simulated by monkeypatching the connection-manager singleton
``send_command`` (same pattern as ``test_config_update_flow.py``); warm-up is
observed through a stub scheduler.

Key contracts under test:
* a newly-placed definition creates a stopped backend row and is pushed
  (``added``), and the scheduler is asked to warm the agent;
* a de-placed backend that is still ``running``/``in_use`` is NOT deleted and
  is reported busy-safe (``refused``) — the row survives for a later prune;
* a disconnected agent still gets its ghost rows pruned but receives no frame;
* the definition CREATE route fans the push out to every connected agent of
  the type (``any_of_type`` now includes the new backend).
"""

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.api.admin.providers import compute_config_fingerprint
from app.core.db import engine
from app.models import (
    DefinitionAgent,
    ProviderAgent,
    ProviderDefinition,
    ProviderInstance,
    ProviderType,
)
from app.services import assignments as asg
from app.services.connection_manager import manager
from app.services.wire import Frame, FrameKind


async def asyncio_sleep() -> None:
    """Let a fire-and-forget warm-up task scheduled by push run to completion."""
    for _ in range(5):
        await asyncio.sleep(0)


class _StubScheduler:
    """Records warm-up triggers (the slice-4 re-warm seam)."""

    def __init__(self) -> None:
        self.warmed: list[str] = []

    async def warm_up_agent(self, agent_id: str) -> None:
        self.warmed.append(agent_id)


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _seed_mock_type(session: Session) -> None:
    if (
        session.exec(select(ProviderType).where(ProviderType.name == "mock")).first()
        is None
    ):
        session.add(
            ProviderType(
                name="mock",
                schema={"type": "object"},
                schema_fingerprint=compute_config_fingerprint({}),
                status="active",
                max_running_backends=0,
            )
        )
        session.commit()


def _machine(session: Session, *, uid: str) -> ProviderAgent:
    from tests.helpers import get_or_create_machine

    get_or_create_machine(session, uid=uid, total_vram=1000, registration_secret="s")


def _agent(session: Session, *, uid: str, connected: bool) -> ProviderAgent:
    from tests.helpers import get_or_create_machine, make_agent

    machine = get_or_create_machine(session, uid=uid, total_vram=1000)
    return make_agent(session, machine, provider_type="mock", connected=connected)


def _def(
    session: Session,
    *,
    alias: str,
    placement: str = "any_of_type",
    enabled: bool = True,
    provider_type: str = "mock",
) -> ProviderDefinition:
    from tests.helpers import make_definition

    return make_definition(
        session,
        alias=alias,
        provider_type=provider_type,
        backend_config={"delta_count": 3},
        enabled=enabled,
        agent_placement=placement,
    )


def _row(
    session: Session,
    agent: ProviderAgent,
    definition: ProviderDefinition,
    *,
    status: str,
) -> ProviderInstance:
    from tests.helpers import make_instance

    return make_instance(
        session, agent, definition, backend_status=status
    )


def _fake_send(record: list, ack: dict | None = None):
    async def send(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        record.append((agent_id, type_, dict(payload)))
        return Frame(type="ack", payload=ack or {"ok": True})

    return send


def _echo_added_send(record: list):
    """A provider that hosts everything the admin pushed (the mock): its ack
    echoes every assignment instance_id as ``added``."""

    async def send(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        record.append((agent_id, type_, dict(payload)))
        added = [e["instance_id"] for e in payload.get("assignments", [])]
        return Frame(
            type="ack",
            payload={"ok": True, "added": added, "removed": [], "refused": []},
        )

    return send


async def test_push_creates_new_row_and_warms(session: Session, monkeypatch) -> None:
    """A newly-placed definition on a connected agent creates a stopped row,
    is pushed in the assignment set, and triggers warm-up."""
    agent = _agent(session, uid="push-add", connected=True)
    kept = _def(session, alias="kept")
    _row(session, agent, kept, status="stopped")
    new = _def(session, alias="brand-new")  # no row yet -> should be created

    sent: list = []
    monkeypatch.setattr(manager, "send_command", _echo_added_send(sent))
    sched = _StubScheduler()

    diff = await asg.push_agent_assignments(str(agent.id), scheduler=sched)
    await asyncio_sleep()

    assert diff is not None
    # The brand-new def got a row; the kept def's row is untouched.
    session.expire_all()
    rows = {
        i.provider_definition_id: i
        for i in session.exec(
            select(ProviderInstance).where(ProviderInstance.agent_id == agent.id)
        ).all()
    }
    assert new.id in rows and rows[new.id].backend_status == "stopped"
    assert kept.id in rows
    assert str(rows[new.id].id) in diff.added

    # Exactly one assignments.update frame, carrying BOTH placed backends.
    frames = [s for s in sent if s[1] == FrameKind.AGENT_ASSIGNMENTS_UPDATE]
    assert len(frames) == 1
    payload = frames[0][2]
    assert {e["alias"] for e in payload["assignments"]} == {"kept", "brand-new"}
    assert payload["max_running_backends"] == 0
    entry = next(e for e in payload["assignments"] if e["alias"] == "brand-new")
    assert entry["provider_definition_id"] == str(new.id)
    assert entry["modality"] == "llm"  # Phase 18 slice 3: default modality
    assert entry["backend_config"] == {"delta_count": 3}
    assert entry["config_fingerprint"] == compute_config_fingerprint({"delta_count": 3})
    assert entry["capacity"] == 1
    assert entry["idle_timeout_seconds"] == 300
    assert entry["vram_required_bytes"] == 0
    assert "instance_id" in entry
    assert "port" not in entry  # port model overhaul: no per-backend port

    # Warm-up was triggered (a backend was added).
    assert sched.warmed == [str(agent.id)]


async def test_push_busy_removal_is_refused_and_row_kept(
    session: Session, monkeypatch
) -> None:
    """De-placing a still-running backend must NOT delete its row (busy-safe);
    it is reported in the admin diff's refused list."""
    agent = _agent(session, uid="push-busy", connected=True)
    running = _def(session, alias="running-def")
    inst = _row(session, agent, running, status="running")

    # De-place it: flip to specific on a DIFFERENT agent (so it is no longer
    # placed on `agent`).
    other = _agent(session, uid="push-busy-other", connected=False)
    session.add(DefinitionAgent(provider_definition_id=running.id, agent_id=other.id))
    running.agent_placement = "specific"
    session.add(running)
    session.commit()

    sent: list = []
    monkeypatch.setattr(
        manager,
        "send_command",
        _fake_send(sent, {"ok": True, "added": [], "removed": [], "refused": []}),
    )
    sched = _StubScheduler()

    diff = await asg.push_agent_assignments(str(agent.id), scheduler=sched)
    await asyncio_sleep()

    assert diff is not None
    # The running row survives and is refused (not removed).
    session.expire_all()
    assert session.get(ProviderInstance, inst.id) is not None
    assert str(inst.id) not in diff.removed
    assert {"instance_id": str(inst.id), "reason": "backend_busy"} in diff.refused
    # The pushed assignment set no longer includes the de-placed backend.
    frames = [s for s in sent if s[1] == FrameKind.AGENT_ASSIGNMENTS_UPDATE]
    assert frames, sent
    assert all(e["alias"] != "running-def" for e in frames[0][2]["assignments"])
    # Nothing added -> no warm-up churn.
    assert sched.warmed == []


async def test_push_stops_idle_dropped_backend(session: Session, monkeypatch) -> None:
    """A de-placed backend that is idle (stopped) IS removed from the rows and
    reported in the admin diff's removed list."""
    agent = _agent(session, uid="push-drop", connected=True)
    ghost = _def(session, alias="ghost-def")
    inst = _row(session, agent, ghost, status="stopped")
    # De-place: disable the definition (no longer placed anywhere).
    ghost.enabled = False
    session.add(ghost)
    session.commit()

    sent: list = []
    monkeypatch.setattr(manager, "send_command", _echo_added_send(sent))
    sched = _StubScheduler()

    diff = await asg.push_agent_assignments(str(agent.id), scheduler=sched)
    await asyncio_sleep()

    assert diff is not None
    assert str(inst.id) in diff.removed
    session.expunge_all()
    assert session.get(ProviderInstance, inst.id) is None


async def test_push_skips_socket_when_disconnected(
    session: Session, monkeypatch
) -> None:
    """A disconnected agent still gets ghost rows pruned but receives no frame
    (the next registration re-resolves)."""
    agent = _agent(session, uid="push-disc", connected=False)
    ghost = _def(session, alias="disc-ghost")
    inst = _row(session, agent, ghost, status="stopped")
    ghost.enabled = False
    session.add(ghost)
    session.commit()

    sent: list = []
    monkeypatch.setattr(manager, "send_command", _fake_send(sent, {"ok": True}))
    await asg.push_agent_assignments(str(agent.id), scheduler=_StubScheduler())
    await asyncio_sleep()

    session.expunge_all()
    assert session.get(ProviderInstance, inst.id) is None  # pruned
    assert sent == []  # no socket push


async def test_create_definition_pushes_to_connected_agents_of_type(
    client: TestClient, session: Session, monkeypatch
) -> None:
    """Definition CREATE of type T pushes agent.assignments.update to every
    connected agent of T (any_of_type now includes the new backend)."""
    agent = _agent(session, uid="create-push", connected=True)

    sent: list = []
    monkeypatch.setattr(manager, "send_command", _echo_added_send(sent))
    # Neutralize warm-up (boots would issue backend.start frames).
    from app.main import app

    sched = _StubScheduler()
    monkeypatch.setattr(app.state, "scheduler", sched)

    resp = client.post(
        "/admin/api/definitions",
        json={
            "alias": "created-any",
            "provider_type": "mock",
            "backend_config": {"delta_count": 5},
            "agent_placement": "any_of_type",
        },
    )
    assert resp.status_code == 200, resp.text
    await asyncio_sleep()

    def_id = uuid.UUID(resp.json()["id"])
    session.expire_all()
    row = session.exec(
        select(ProviderInstance).where(
            ProviderInstance.agent_id == agent.id,
            ProviderInstance.provider_definition_id == def_id,
        )
    ).first()
    assert row is not None  # created on the connected agent
    frames = [s for s in sent if s[1] == FrameKind.AGENT_ASSIGNMENTS_UPDATE]
    assert frames and any(
        e["alias"] == "created-any" for e in frames[0][2]["assignments"]
    )
    assert sched.warmed == [str(agent.id)]


async def test_reconcile_is_the_single_diff_path(session: Session) -> None:
    """The registration prune wrapper and the push share reconcile_agent_
    placement: a busy ghost is refused identically (no duplicated logic)."""
    agent = _agent(session, uid="shared-diff", connected=True)
    running = _def(session, alias="shared-running")
    _row(session, agent, running, status="running")
    # De-place running.
    running.enabled = False
    session.add(running)
    session.commit()

    from app.api.admin.providers import _placed_definitions, prune_unplaced_backends

    with Session(engine) as s:
        ag = s.get(ProviderAgent, agent.id)
        placed = _placed_definitions(s, ag)
        # prune_unplaced_backends delegates to the shared reconcile -> the
        # running ghost is NOT deleted (busy-safe), returns 0.
        removed = prune_unplaced_backends(s, ag, placed)
        s.commit()
    assert removed == 0
    session.expunge_all()
    assert (
        session.exec(
            select(ProviderInstance).where(
                ProviderInstance.provider_definition_id == running.id
            )
        ).first()
        is not None
    )


async def test_push_no_warm_when_provider_refuses_add(
    session: Session, monkeypatch
) -> None:
    """M2: a single-backend provider refuses the add (ack added=[]) even though
    the admin created a row — warm-up must NOT fire (no futile backend.start
    NAK churn for an unhostable id)."""
    agent = _agent(session, uid="push-refuse", connected=True)
    new = _def(session, alias="refused-def")  # no row -> admin creates it

    sent: list = []
    # Provider acks ok but added=[] (it refused the add — no make_handle).
    monkeypatch.setattr(
        manager,
        "send_command",
        _fake_send(sent, {"ok": True, "added": [], "removed": [], "refused": []}),
    )
    sched = _StubScheduler()

    diff = await asg.push_agent_assignments(str(agent.id), scheduler=sched)
    await asyncio_sleep()

    assert diff is not None
    # The admin DID create a row for the new def (its id is in diff.added)...
    session.expire_all()
    row = session.exec(
        select(ProviderInstance).where(
            ProviderInstance.agent_id == agent.id,
            ProviderInstance.provider_definition_id == new.id,
        )
    ).first()
    assert row is not None
    assert str(row.id) in diff.added
    # ...but the provider refused the add (ack added=[]) -> no warm-up fires.
    assert sched.warmed == []


async def test_push_allows_shared_base_port_no_clash(
    session: Session, monkeypatch
) -> None:
    """Port model overhaul: two connected agents on one machine may share the
    same ``base_port`` (each routes ``/v1`` by model internally). A placement
    push to the second agent now simply creates its backend row and pushes it —
    there is no cross-agent ``port_conflict`` refusal anymore."""
    from tests.helpers import get_or_create_machine, make_agent

    machine = get_or_create_machine(session, uid="clash-m", total_vram=1000)
    # Peer agent A (connected) already hosts a running backend on base_port 8081.
    peer = make_agent(session, machine, provider_type="mock", base_port=8081)
    peer_def = _def(session, alias="peer-def")
    _row(session, peer, peer_def, status="running")
    # Target agent B shares base_port 8081, has no rows, and is a DIFFERENT type
    # so its only placed def is the one that used to collide (peer_def is not
    # placed on it).
    target = make_agent(
        session,
        machine,
        provider_type="llama-cpp",
        base_port=8081,
        agent_id="target-b",
    )
    new = _def(session, alias="collide-def", provider_type="llama-cpp")

    sent: list = []
    monkeypatch.setattr(manager, "send_command", _echo_added_send(sent))
    diff = await asg.push_agent_assignments(str(target.id), scheduler=_StubScheduler())
    await asyncio_sleep()

    assert diff is not None
    # The formerly-colliding row IS created and pushed (no refusal, no port).
    assert diff.added and all(r["reason"] != "port_conflict" for r in diff.refused)
    assert any(a["alias"] == "collide-def" for a in diff.assignments)
    session.expire_all()
    row = session.exec(
        select(ProviderInstance).where(
            ProviderInstance.agent_id == target.id,
            ProviderInstance.provider_definition_id == new.id,
        )
    ).first()
    assert row is not None


def test_build_assignment_entry_carries_modality(session: Session) -> None:
    """Phase 18 slice 3: the assignment entry mirrors the definition's modality
    (single source of truth = the definition row)."""
    from tests.helpers import get_or_create_machine, make_agent, make_definition

    machine = get_or_create_machine(session, uid="entry-mod", total_vram=1000)
    agent = make_agent(session, machine, provider_type="mock", connected=True)
    emb = make_definition(session, alias="emb-entry", modality="embedding")
    emb_row = _row(session, agent, emb, status="stopped")
    assert asg.build_assignment_entry(session, emb_row, emb)["modality"] == "embedding"

    llm = make_definition(session, alias="llm-entry")  # default modality llm
    llm_row = _row(session, agent, llm, status="stopped")
    assert asg.build_assignment_entry(session, llm_row, llm)["modality"] == "llm"


async def test_reconcile_creates_row_and_refuses_busy_ghost(session: Session) -> None:
    """Port model overhaul: reconcile creates a stopped row for a newly-placed
    def (no admin-facing port) while a de-placed-but-running ghost on the same
    agent is retained busy-safe (reported refused), never deleted."""
    from tests.helpers import get_or_create_machine, make_agent

    machine = get_or_create_machine(session, uid="renum-m", total_vram=1000)
    agent = make_agent(session, machine, provider_type="mock", base_port=8081)
    # A running ghost that is NOT placed (disabled def).
    ghost = _def(session, alias="ghost-running", enabled=False)
    ghost_row = _row(session, agent, ghost, status="running")
    # A placed def (any_of_type) with no row yet -> reconcile creates one.
    _def(session, alias="placed-a")

    from app.api.admin.providers import _placed_definitions

    with Session(engine) as s:
        ag = s.get(ProviderAgent, agent.id)
        placed = _placed_definitions(s, ag)
        diff = asg.reconcile_agent_placement(s, ag, placed, create_new=True)
        s.commit()
        placed_row = s.get(ProviderInstance, uuid.UUID(diff.added[0]))
        ghost_after = s.get(ProviderInstance, ghost_row.id)

    # The placed def got a new stopped row; the busy ghost is retained + refused.
    assert placed_row is not None and placed_row.backend_status == "stopped"
    assert ghost_after is not None
    assert {"instance_id": str(ghost_row.id), "reason": "backend_busy"} in diff.refused

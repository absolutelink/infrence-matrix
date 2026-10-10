"""InferenceScheduler: FIFO admission, capacity, VRAM, boot, timeout,
idempotent + cancellation-safe release.

The provider WS command path is simulated by monkeypatching
``connection_manager.manager.send_command`` (same pattern as
``test_metrics_ownership.py``). Instances are seeded directly in the UTF8
test DB with ``websocket_connected=True``.
"""

import asyncio
import contextlib
import inspect
import time
import uuid
from collections import deque

import pytest
import redis.asyncio as aioredis
from sqlmodel import Session, select

from app.core.db import engine
from app.models import (
    Machine,
    ProviderAgent,
    ProviderDefinition,
    ProviderInstance,
    ProviderType,
)
from app.services import redis_keys
from app.services.connection_manager import manager
from app.services.scheduler import (
    InferenceScheduler,
    NoProviderAvailable,
    QueueCleared,
    QueueTimeout,
)
from app.services.wire import BackendStatusValue, Frame, FrameKind
from tests.helpers import (
    get_or_create_machine,
    make_agent,
    make_definition,
    make_instance,
)

TEST_REDIS_URL = __import__("os").environ["TEST_REDIS_URL"]


def make_provider_type(
    session: Session, name: str, *, max_running: int
) -> ProviderType:
    """Seed a ProviderType row so the scheduler can read its per-agent
    ``max_running_backends`` cap (slice 4). The scheduler treats a missing
    row as unlimited, so cap tests must create one explicitly."""
    existing = session.exec(
        select(ProviderType).where(ProviderType.name == name)
    ).first()
    if existing is not None:
        existing.max_running_backends = max_running
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing
    ptype = ProviderType(
        name=name,
        schema={"type": "object"},
        schema_fingerprint=f"fp-{name}",
        max_running_backends=max_running,
        status="active",
    )
    session.add(ptype)
    session.commit()
    session.refresh(ptype)
    return ptype


def _agent(
    session: Session,
    machine: Machine,
    *,
    connected: bool = True,
    base_port: int = 8081,
) -> ProviderAgent:
    """Get-or-create a connected agent for ``machine`` (one per machine)."""
    existing = session.exec(
        select(ProviderAgent).where(ProviderAgent.machine_id == machine.id)
    ).first()
    if existing is not None:
        if existing.websocket_connected != connected:
            existing.websocket_connected = connected
            session.add(existing)
            session.commit()
            session.refresh(existing)
        return existing
    return make_agent(session, machine, connected=connected, base_port=base_port)


def make_stack(
    session: Session,
    *,
    machine_uid: str,
    alias: str,
    total_vram: int = 1000,
    vram_required: int = 0,
    capacity: int = 1,
    backend_status: str = "stopped",
    connected: bool = True,
    port: int = 8081,
) -> tuple[Machine, ProviderDefinition, ProviderInstance]:
    machine = get_or_create_machine(session, uid=machine_uid, total_vram=total_vram)
    agent = make_agent(session, machine, base_port=port, connected=connected)
    definition = make_definition(
        session,
        alias=alias,
        vram_required=vram_required,
        capacity=capacity,
        backend_config={"model": {"file": "m.gguf"}},
    )
    instance = make_instance(
        session,
        agent,
        definition,
        backend_status=backend_status,
    )
    return machine, definition, instance


@pytest.fixture
async def aredis():
    client = aioredis.from_url(TEST_REDIS_URL, decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def boot_calls(monkeypatch):
    """Record backend.start commands (keyed by target instance_id) and ack ok."""
    calls: list[tuple[str, str]] = []

    async def fake_send_command(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append((payload.get("instance_id", agent_id), type_))
        return Frame(type="ack", payload={"ok": True, "detail": {"capacity": 1}})

    monkeypatch.setattr(manager, "send_command", fake_send_command)
    return calls


async def wait_until(predicate, timeout: float = 3.0):
    """Poll a sync or async predicate until truthy or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if inspect.isawaitable(result):
            result = await result
        if result:
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met in time")


# ---------------------------------------------------------------------------
# No candidates
# ---------------------------------------------------------------------------
async def test_acquire_no_connected_instance_raises(session: Session, aredis) -> None:
    make_stack(session, machine_uid="sch-none", alias="none-a", connected=False)
    scheduler = InferenceScheduler(aredis)
    with pytest.raises(NoProviderAvailable):
        await scheduler.acquire("none-a", "req-1")
    # Nothing leaked into Redis.
    assert await aredis.smembers(redis_keys.sched_active_key("none-a")) == set()


async def test_acquire_unknown_alias_raises(aredis) -> None:
    scheduler = InferenceScheduler(aredis)
    with pytest.raises(NoProviderAvailable):
        await scheduler.acquire("ghost-alias", "req-x")


# ---------------------------------------------------------------------------
# Capacity + FIFO + boot
# ---------------------------------------------------------------------------
async def test_capacity_one_fifo_with_boot(
    session: Session, aredis, boot_calls
) -> None:
    _, _, instance = make_stack(
        session, machine_uid="sch-cap", alias="cap-a", capacity=1
    )
    scheduler = InferenceScheduler(aredis)

    admission = await scheduler.acquire("cap-a", "r1")
    assert admission.instance_id == str(instance.id)
    assert admission.base_url == "http://127.0.0.1:8081"
    assert admission.machine_uid == "sch-cap"
    # Stopped backend was booted.
    assert (str(instance.id), "backend.start") in boot_calls
    assert scheduler.active_count("cap-a") == 1
    assert await aredis.smembers(redis_keys.sched_active_key("cap-a")) == {"r1"}

    # Second acquire queues (head is r2 now that r1 is admitted).
    r2_task = asyncio.create_task(scheduler.acquire("cap-a", "r2"))
    await wait_until(lambda: scheduler._states["cap-a"].waiters == deque(["r2"]))
    assert scheduler.active_count("cap-a") == 1
    assert await aredis.smembers(redis_keys.sched_active_key("cap-a")) == {"r1"}
    await wait_until(lambda: aredis.lrange(redis_keys.sched_queue_key("cap-a"), 0, -1))

    # Third queues behind the second: FIFO order visible in the mirror.
    r3_task = asyncio.create_task(scheduler.acquire("cap-a", "r3"))

    async def queue_has_r2_r3() -> bool:
        return await aredis.lrange(redis_keys.sched_queue_key("cap-a"), 0, -1) == [
            "r2",
            "r3",
        ]

    await wait_until(queue_has_r2_r3)

    # Release r1 -> r2 admitted (still no extra capacity for r3).
    await scheduler.release("cap-a", "r1")
    admission2 = await asyncio.wait_for(r2_task, timeout=3)
    assert admission2.instance_id == str(instance.id)
    assert scheduler.active_count("cap-a") == 1
    assert await aredis.smembers(redis_keys.sched_active_key("cap-a")) == {"r2"}

    await scheduler.release("cap-a", "r2")
    admission3 = await asyncio.wait_for(r3_task, timeout=3)
    assert admission3.instance_id == str(instance.id)
    await scheduler.release("cap-a", "r3")
    assert scheduler.active_count("cap-a") == 0
    assert await aredis.smembers(redis_keys.sched_active_key("cap-a")) == set()


async def test_capacity_two_admits_two_without_queueing(
    session: Session, aredis, boot_calls
) -> None:
    _, _, instance = make_stack(
        session, machine_uid="sch-cap2", alias="cap2-a", capacity=2
    )
    scheduler = InferenceScheduler(aredis)
    a1 = await scheduler.acquire("cap2-a", "r1")
    a2 = await scheduler.acquire("cap2-a", "r2")
    assert a1.instance_id == a2.instance_id == str(instance.id)
    assert scheduler.active_count("cap2-a") == 2
    # One boot for the shared instance.
    assert boot_calls.count((str(instance.id), "backend.start")) == 1


# ---------------------------------------------------------------------------
# VRAM admission (per-booted-instance ledger)
#
# SEMANTIC SHIFT (Phase 6 close-out): VRAM is held per *booted instance*,
# not per request slot. The old ledger had ``release`` clear the hold, so a
# running-but-idle backend appeared to hold 0 VRAM — physically false (a
# booted llama-server keeps its weights resident) and it made eviction
# impossible (nothing to free). Under the new model an idle loaded backend
# still holds its VRAM, and a request that needs the space EVICTS it
# (backend.stop) instead of waiting for a "free" that never arrives.
# ---------------------------------------------------------------------------
async def test_vram_second_alias_evicts_idle_first_alias(
    session: Session, aredis, boot_calls
) -> None:
    machine = Machine(uid="sch-vram", name="sch-vram", host="127.0.0.1")
    machine.total_vram_bytes = 100
    session.add(machine)
    session.commit()
    agent = make_agent(session, machine, connected=True)
    def_a = ProviderDefinition(
        alias="va",
        provider_type="mock",
        vram_required_bytes=80,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    def_b = ProviderDefinition(
        alias="vb",
        provider_type="mock",
        vram_required_bytes=80,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    session.add_all([def_a, def_b])
    session.commit()
    inst_a = ProviderInstance(
        agent_id=agent.id,
        provider_definition_id=def_a.id,
        backend_status="running",
    )
    inst_b = ProviderInstance(
        agent_id=agent.id,
        provider_definition_id=def_b.id,
        backend_status="stopped",
    )
    session.add_all([inst_a, inst_b])
    session.commit()

    scheduler = InferenceScheduler(aredis, queue_timeout=0.3)

    # A is admitted (already running; no boot). The running instance holds
    # 80 in the mirror (adopted into the per-booted ledger on admission).
    admission = await scheduler.acquire("va", "ra")
    assert admission.instance_id == str(inst_a.id)
    assert boot_calls == []
    used = await aredis.hgetall(redis_keys.vram_used_key("sch-vram"))
    assert used == {str(inst_a.id): "80"}

    # B needs 80 but A's LOADED backend holds 80 and A's slot is ACTIVE:
    # no idle victim exists, so B queues and times out (never evict busy).
    with pytest.raises(QueueTimeout):
        await scheduler.acquire("vb", "rb")
    used = await aredis.hgetall(redis_keys.vram_used_key("sch-vram"))
    assert used == {str(inst_a.id): "80"}
    assert scheduler.active_count("vb") == 0

    # Release A -> the hold REMAINS (backend still loaded). B now fits only
    # by EVICTING idle A: backend.stop to A, then boot B.
    await scheduler.release("va", "ra")
    assert str(inst_a.id) in scheduler._booted, (
        "release must not clear the booted instance's VRAM hold"
    )
    admission_b = await scheduler.acquire("vb", "rb2")
    assert admission_b.instance_id == str(inst_b.id)
    assert (str(inst_a.id), "backend.stop") in boot_calls
    assert (str(inst_b.id), "backend.start") in boot_calls
    assert str(inst_a.id) not in scheduler._booted
    used = await aredis.hgetall(redis_keys.vram_used_key("sch-vram"))
    assert used == {str(inst_b.id): "80"}


# ---------------------------------------------------------------------------
# Eviction policy details
# ---------------------------------------------------------------------------
def two_alias_machine(
    session: Session,
    *,
    machine_uid: str,
    total_vram: int,
    a_alias: str,
    b_alias: str,
    a_vram: int,
    b_vram: int,
    a_status: str = "running",
    a_last_request_at=None,
) -> tuple[ProviderInstance, ProviderInstance]:
    """One machine + one agent, two aliases/backends; returns (inst_a, inst_b)."""
    machine = Machine(uid=machine_uid, name=machine_uid, host="127.0.0.1")
    machine.total_vram_bytes = total_vram
    session.add(machine)
    session.commit()
    agent = make_agent(session, machine, connected=True)
    def_a = ProviderDefinition(
        alias=a_alias,
        provider_type="mock",
        vram_required_bytes=a_vram,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    def_b = ProviderDefinition(
        alias=b_alias,
        provider_type="mock",
        vram_required_bytes=b_vram,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    session.add_all([def_a, def_b])
    session.commit()
    inst_a = ProviderInstance(
        agent_id=agent.id,
        provider_definition_id=def_a.id,
        backend_status=a_status,
        last_request_at=a_last_request_at,
    )
    inst_b = ProviderInstance(
        agent_id=agent.id,
        provider_definition_id=def_b.id,
        backend_status="stopped",
    )
    session.add_all([inst_a, inst_b])
    session.commit()
    return inst_a, inst_b


async def test_evict_never_touches_busy_victim(
    session: Session, aredis, boot_calls
) -> None:
    """A holds an ACTIVE slot -> not evictable; B waits and times out."""
    inst_a, _ = two_alias_machine(
        session,
        machine_uid="sch-ev-busy",
        total_vram=100,
        a_alias="eva",
        b_alias="evb",
        a_vram=60,
        b_vram=60,
    )
    scheduler = InferenceScheduler(aredis, queue_timeout=0.3)
    await scheduler.acquire("eva", "ra")  # boots/holds A, 1 active slot
    with pytest.raises(QueueTimeout):
        await scheduler.acquire("evb", "rb")
    assert (str(inst_a.id), "backend.stop") not in boot_calls
    assert str(inst_a.id) in scheduler._booted


async def test_evict_lru_order_oldest_first(
    session: Session, aredis, boot_calls
) -> None:
    """Two idle victims -> the oldest last_request_at is evicted first."""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    inst_a, inst_b = two_alias_machine(
        session,
        machine_uid="sch-ev-lru",
        total_vram=100,
        a_alias="lua",
        b_alias="lub",
        a_vram=40,
        b_vram=40,
    )
    # A third alias needs the space; both A and B are loaded + idle.
    def_c = ProviderDefinition(
        alias="luc",
        provider_type="mock",
        vram_required_bytes=50,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    session.add(def_c)
    session.commit()
    inst_c = ProviderInstance(
        agent_id=inst_a.agent_id,
        provider_definition_id=def_c.id,
        backend_status="stopped",
    )
    session.add(inst_c)
    session.commit()

    scheduler = InferenceScheduler(aredis, queue_timeout=1.0)
    # Adopt A and B into the ledger via a serve+release, then backdate so
    # A is the LRU (oldest last_request_at).
    await scheduler.acquire("lua", "ra")
    await scheduler.release("lua", "ra")
    await scheduler.acquire("lub", "rb")
    await scheduler.release("lub", "rb")
    with Session(engine) as s:
        ia = s.get(ProviderInstance, inst_a.id)
        ib = s.get(ProviderInstance, inst_b.id)
        ia.last_request_at = now - timedelta(seconds=100)
        ib.last_request_at = now - timedelta(seconds=10)
        s.add(ia)
        s.add(ib)
        s.commit()  # C needs 50; free = 100 - (40+40) = 20. Evict LRU (A) -> free 60
    # >= 50, so B is left alone.
    admission = await scheduler.acquire("luc", "rc")
    assert admission.instance_id == str(inst_c.id)
    assert (str(inst_a.id), "backend.stop") in boot_calls
    assert (str(inst_b.id), "backend.stop") not in boot_calls
    assert str(inst_a.id) not in scheduler._booted
    assert str(inst_b.id) in scheduler._booted


async def test_evict_only_different_alias_victims(session: Session, aredis) -> None:
    """_eviction_candidates never lists the requesting definition's instances."""
    machine = Machine(uid="sch-ev-diff", name="sch-ev-diff", host="127.0.0.1")
    machine.total_vram_bytes = 100
    session.add(machine)
    def_a = ProviderDefinition(
        alias="dif-a",
        provider_type="mock",
        vram_required_bytes=40,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    def_b = ProviderDefinition(
        alias="dif-b",
        provider_type="mock",
        vram_required_bytes=40,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    session.add_all([def_a, def_b])
    session.commit()
    inst_a = ProviderInstance(
        agent_id=_agent(session, machine).id,
        provider_definition_id=def_a.id,
        backend_status="running",
    )
    inst_b = ProviderInstance(
        agent_id=_agent(session, machine).id,
        provider_definition_id=def_b.id,
        backend_status="running",
    )
    session.add_all([inst_a, inst_b])
    session.commit()

    scheduler = InferenceScheduler(aredis)
    held = {str(inst_a.id): 40, str(inst_b.id): 40}
    # Requesting dif-b's space (target inst_b): the only candidate is inst_a
    # (dif-a, a different definition). Phase 25: exclusion is by definition id.
    victims = scheduler._eviction_candidates(
        "sch-ev-diff", def_b.id, str(inst_b.id), held
    )
    assert [v["id"] for v in victims] == [str(inst_a.id)]
    # Symmetric: a dif-a request targeting inst_a can only evict inst_b.
    victims = scheduler._eviction_candidates(
        "sch-ev-diff", def_a.id, str(inst_a.id), held
    )
    assert [v["id"] for v in victims] == [str(inst_b.id)]


async def test_evict_victim_nak_tries_next_candidate(
    session: Session, aredis, monkeypatch
) -> None:
    """First victim refuses the stop -> the next LRU victim is evicted."""
    from datetime import UTC, datetime, timedelta

    machine = Machine(uid="sch-ev-nak", name="sch-ev-nak", host="127.0.0.1")
    machine.total_vram_bytes = 100
    session.add(machine)
    def_a = ProviderDefinition(
        alias="nka",
        provider_type="mock",
        vram_required_bytes=40,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    def_b = ProviderDefinition(
        alias="nkb",
        provider_type="mock",
        vram_required_bytes=40,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    def_c = ProviderDefinition(
        alias="nkc",
        provider_type="mock",
        vram_required_bytes=50,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    session.add_all([def_a, def_b, def_c])
    session.commit()
    inst_a = ProviderInstance(
        agent_id=_agent(session, machine).id,
        provider_definition_id=def_a.id,
        backend_status="running",
        last_request_at=datetime.now(UTC) - timedelta(seconds=200),
    )
    inst_b = ProviderInstance(
        agent_id=_agent(session, machine).id,
        provider_definition_id=def_b.id,
        backend_status="running",
        last_request_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    inst_c = ProviderInstance(
        agent_id=_agent(session, machine).id,
        provider_definition_id=def_c.id,
        backend_status="stopped",
    )
    session.add_all([inst_a, inst_b, inst_c])
    session.commit()

    calls: list[tuple[str, str]] = []

    async def fake_send_command(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append((payload.get("instance_id", agent_id), type_))
        if type_ == "backend.stop" and payload.get("instance_id") == str(inst_a.id):
            # A refuses (drain race); everything else succeeds.
            return Frame(type="ack", payload={"ok": False, "error": "backend_in_use"})
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", fake_send_command)
    scheduler = InferenceScheduler(aredis, queue_timeout=1.0)
    await scheduler._mark_booted(str(inst_a.id), "sch-ev-nak", def_a)
    await scheduler._mark_booted(str(inst_b.id), "sch-ev-nak", def_b)

    admission = await scheduler.acquire("nkc", "rc")
    assert admission.instance_id == str(inst_c.id)
    # A tried first (LRU) but refused -> still held; B then evicted.
    assert (str(inst_a.id), "backend.stop") in calls
    assert (str(inst_b.id), "backend.stop") in calls
    assert str(inst_a.id) in scheduler._booted
    assert str(inst_b.id) not in scheduler._booted
    assert str(inst_c.id) in scheduler._booted


async def test_evict_no_workable_victim_stays_queued(
    session: Session, aredis, monkeypatch
) -> None:
    """Every victim refuses the stop -> request stays queued (QueueTimeout);
    nothing is force-killed."""
    machine = Machine(uid="sch-ev-none", name="sch-ev-none", host="127.0.0.1")
    machine.total_vram_bytes = 100
    session.add(machine)
    def_a = ProviderDefinition(
        alias="noa",
        provider_type="mock",
        vram_required_bytes=60,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    def_b = ProviderDefinition(
        alias="nob",
        provider_type="mock",
        vram_required_bytes=60,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    session.add_all([def_a, def_b])
    session.commit()
    inst_a = ProviderInstance(
        agent_id=_agent(session, machine).id,
        provider_definition_id=def_a.id,
        backend_status="running",
    )
    inst_b = ProviderInstance(
        agent_id=_agent(session, machine).id,
        provider_definition_id=def_b.id,
        backend_status="stopped",
    )
    session.add_all([inst_a, inst_b])
    session.commit()

    async def refuse_stops(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        if type_ == "backend.stop":
            return Frame(type="ack", payload={"ok": False, "error": "backend_in_use"})
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", refuse_stops)
    scheduler = InferenceScheduler(aredis, queue_timeout=0.3)
    await scheduler._mark_booted(str(inst_a.id), "sch-ev-none", def_a)
    with pytest.raises(QueueTimeout):
        await scheduler.acquire("nob", "rb")
    assert str(inst_a.id) in scheduler._booted  # refused -> still held
    assert scheduler.active_count("nob") == 0


# ---------------------------------------------------------------------------
# Boot failure
# ---------------------------------------------------------------------------
async def test_boot_failure_does_not_admit_or_leak(
    session: Session, aredis, monkeypatch
) -> None:
    _, _, instance = make_stack(
        session, machine_uid="sch-bootfail", alias="bf-a", capacity=1
    )

    async def refuse_send_command(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        return Frame(type="ack", payload={"ok": False, "error": "boot exploded"})

    monkeypatch.setattr(manager, "send_command", refuse_send_command)
    scheduler = InferenceScheduler(aredis, queue_timeout=0.3)

    with pytest.raises(QueueTimeout):
        await scheduler.acquire("bf-a", "r1")
    assert scheduler.active_count("bf-a") == 0
    assert await aredis.smembers(redis_keys.sched_active_key("bf-a")) == set()
    assert await aredis.hgetall(redis_keys.vram_used_key("sch-bootfail")) == {}

    # A working boot afterwards still admits.
    async def ok_send_command(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", ok_send_command)
    admission = await scheduler.acquire("bf-a", "r2")
    assert admission.instance_id == str(instance.id)
    await scheduler.release("bf-a", "r2")


async def test_boot_exception_falls_through_to_next_candidate(
    session: Session, aredis, monkeypatch
) -> None:
    make_stack(
        session,
        machine_uid="sch-b1",
        alias="ft-a",
        capacity=1,
        port=8081,
    )
    # A second connected instance on another machine for the same alias.
    machine2 = Machine(uid="sch-b2", name="sch-b2", host="10.0.0.2")
    session.add(machine2)
    definition = (
        session.query(ProviderDefinition)
        .filter(ProviderDefinition.alias == "ft-a")
        .first()
    )
    inst2 = ProviderInstance(
        agent_id=_agent(session, machine2, base_port=9091).id,
        provider_definition_id=definition.id,
        backend_status="stopped",
    )
    session.add(inst2)
    session.commit()

    async def flaky_send_command(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        if payload.get("instance_id") != str(inst2.id):
            raise ConnectionError("socket gone")
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", flaky_send_command)
    scheduler = InferenceScheduler(aredis, queue_timeout=1.0)
    admission = await scheduler.acquire("ft-a", "r1")
    assert admission.instance_id == str(inst2.id)
    assert admission.base_url == "http://10.0.0.2:9091"
    await scheduler.release("ft-a", "r1")


# ---------------------------------------------------------------------------
# Release idempotency + cancellation safety
# ---------------------------------------------------------------------------
async def test_release_is_idempotent(
    session: Session,
    aredis,
    boot_calls,  # noqa: ARG001
) -> None:
    # boot_calls: the instance starts stopped, so the acquire does boot it.
    _, _, instance = make_stack(
        session,
        machine_uid="sch-idem",
        alias="id-a",
        capacity=1,
        vram_required=42,
    )
    scheduler = InferenceScheduler(aredis)
    await scheduler.acquire("id-a", "r1")
    await scheduler.release("id-a", "r1")
    # Second and third release: no-ops, no exceptions.
    await scheduler.release("id-a", "r1")
    await scheduler.release("id-a", "never-admitted")
    assert scheduler.active_count("id-a") == 0
    assert await aredis.smembers(redis_keys.sched_active_key("id-a")) == set()
    # NEW semantics: release frees the capacity slot but NOT the booted
    # instance's VRAM hold (the backend is still loaded).
    assert str(instance.id) in scheduler._booted
    assert await aredis.hgetall(redis_keys.vram_used_key("sch-idem")) == {
        str(instance.id): "42"
    }


async def test_release_inside_cancelled_task_still_cleans_redis(
    session: Session,
    aredis,
    boot_calls,  # noqa: ARG001
) -> None:
    _, _, instance = make_stack(
        session,
        machine_uid="sch-cancel",
        alias="cx-a",
        capacity=1,
        vram_required=42,
    )
    scheduler = InferenceScheduler(aredis)

    async def worker() -> None:
        await scheduler.acquire("cx-a", "r1")
        # Hold the slot until cancelled; release must still land.
        try:
            await asyncio.sleep(10)
        finally:
            await scheduler.release("cx-a", "r1")

    task = asyncio.create_task(worker())
    await wait_until(lambda: scheduler.active_count("cx-a") == 1)
    assert await aredis.smembers(redis_keys.sched_active_key("cx-a")) == {"r1"}

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The shielded release runs as a detached inner task that keeps going
    # past the cancelled caller; poll until both the in-process slot and
    # the Redis active-set cleanup land. The booted instance's VRAM HOLD
    # stays (new per-booted-instance semantics) — release only frees the
    # capacity slot, never the loaded backend's reservation.
    async def released_everywhere() -> bool:
        return (
            scheduler.active_count("cx-a") == 0
            and await aredis.smembers(redis_keys.sched_active_key("cx-a")) == set()
        )

    await wait_until(released_everywhere)
    assert str(instance.id) in scheduler._booted
    assert await aredis.hgetall(redis_keys.vram_used_key("sch-cancel")) == {
        str(instance.id): "42"
    }


async def test_cancelled_acquire_removes_waiter(
    session: Session,
    aredis,
    boot_calls,  # noqa: ARG001
) -> None:
    make_stack(session, machine_uid="sch-cancelq", alias="cq-a", capacity=1)
    scheduler = InferenceScheduler(aredis)
    await scheduler.acquire("cq-a", "r1")

    task = asyncio.create_task(scheduler.acquire("cq-a", "r2"))
    await wait_until(lambda: "r2" in scheduler._states["cq-a"].waiters)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert list(scheduler._states["cq-a"].waiters) == []
    assert await aredis.lrange(redis_keys.sched_queue_key("cq-a"), 0, -1) == []
    # r1's slot untouched.
    assert await aredis.smembers(redis_keys.sched_active_key("cq-a")) == {"r1"}


# ---------------------------------------------------------------------------
# Queue timeout
# ---------------------------------------------------------------------------
async def test_queue_timeout_when_capacity_held(
    session: Session,
    aredis,
    boot_calls,  # noqa: ARG001
) -> None:
    make_stack(session, machine_uid="sch-to", alias="to-a", capacity=1)
    scheduler = InferenceScheduler(aredis, queue_timeout=0.2)
    await scheduler.acquire("to-a", "r1")
    start = time.monotonic()
    with pytest.raises(QueueTimeout):
        await scheduler.acquire("to-a", "r2")
    elapsed = time.monotonic() - start
    assert 0.15 <= elapsed < 2.0
    assert scheduler.active_count("to-a") == 1
    assert await aredis.smembers(redis_keys.sched_active_key("to-a")) == {"r1"}
    # Timed-out waiter removed from the mirror.
    assert await aredis.lrange(redis_keys.sched_queue_key("to-a"), 0, -1) == []
    await scheduler.release("to-a", "r1")


# ---------------------------------------------------------------------------
# Background lifecycle
# ---------------------------------------------------------------------------
async def test_start_background_and_stop_clean(aredis) -> None:
    scheduler = InferenceScheduler(aredis)
    await scheduler.start_background()
    assert scheduler._bg_task is not None and not scheduler._bg_task.done()
    await scheduler.stop()
    assert scheduler._bg_task is None


# ---------------------------------------------------------------------------
# Admission identity
# ---------------------------------------------------------------------------
async def test_admission_uses_machine_reachable_preference(
    session: Session, aredis
) -> None:
    machine = Machine(
        uid="sch-addr", name="sch-addr", host="host.local", dns="dns.local"
    )
    session.add(machine)
    session.commit()
    definition = ProviderDefinition(
        alias="ad-a",
        provider_type="mock",
        backend_config={"model": {"file": "m.gguf"}},
    )
    session.add(definition)
    session.commit()
    instance = ProviderInstance(
        agent_id=_agent(session, machine).id,
        provider_definition_id=definition.id,
        backend_status="running",
    )
    session.add(instance)
    session.commit()

    scheduler = InferenceScheduler(aredis)
    admission = await scheduler.acquire("ad-a", "r1")
    # dns preferred over host (Machine.reachable_address).
    assert admission.base_url == "http://dns.local:8081"
    assert admission.machine_uid == "sch-addr"
    # uuid sanity: the instance id round-trips for persistence.
    uuid.UUID(admission.instance_id)


# ---------------------------------------------------------------------------
# Per-booted-instance VRAM ledger
# ---------------------------------------------------------------------------
async def test_already_booted_target_admits_on_capacity_alone(
    session: Session, aredis, boot_calls
) -> None:
    """A booted instance needs no new VRAM: admission is gated only by
    capacity, even when the machine budget is already fully held."""
    machine = Machine(uid="sch-preboot", name="sch-preboot", host="127.0.0.1")
    machine.total_vram_bytes = 60
    session.add(machine)
    definition = ProviderDefinition(
        alias="pb-a",
        provider_type="mock",
        vram_required_bytes=60,
        capacity=2,
        backend_config={"model": {"file": "m.gguf"}},
    )
    session.add(definition)
    session.commit()
    instance = ProviderInstance(
        agent_id=_agent(session, machine).id,
        provider_definition_id=definition.id,
        backend_status="stopped",
    )
    session.add(instance)
    session.commit()

    scheduler = InferenceScheduler(aredis)
    # First request boots (consumes the whole 60 budget).
    await scheduler.acquire("pb-a", "r1")
    assert (str(instance.id), "backend.start") in boot_calls
    # Second request: the instance is already booted and holds the VRAM,
    # so no fresh VRAM is needed -> admits on spare capacity with NO extra
    # boot and no eviction.
    await scheduler.acquire("pb-a", "r2")
    assert boot_calls.count((str(instance.id), "backend.start")) == 1
    assert scheduler.active_count("pb-a") == 2


async def test_ledger_survives_release_and_redis_reflects_holds(
    session: Session, aredis, boot_calls
) -> None:
    """release keeps the hold; the Redis mirror lists booted-instance
    holds (not per-request), and _machine_vram_held agrees."""
    _, _, instance = make_stack(
        session,
        machine_uid="sch-ledger",
        alias="lg-a",
        total_vram=100,
        vram_required=70,
        capacity=1,
    )
    scheduler = InferenceScheduler(aredis)
    await scheduler.acquire("lg-a", "r1")
    # While active: held = 70 (booted), mirror = {instance: 70}.
    held = await scheduler._machine_vram_held(
        instance.machine
        if instance.machine is not None
        else session.get(Machine, instance.agent.machine_id)
    )
    assert held[str(instance.id)] == 70
    assert await aredis.hgetall(redis_keys.vram_used_key("sch-ledger")) == {
        str(instance.id): "70"
    }
    # After release: still held 70 (backend loaded), mirror unchanged.
    await scheduler.release("lg-a", "r1")
    assert str(instance.id) in scheduler._booted
    assert await aredis.hgetall(redis_keys.vram_used_key("sch-ledger")) == {
        str(instance.id): "70"
    }
    # Free VRAM excludes only the target's own hold: a different-alias
    # boot needing 40 fits (100-70=30 < 40 -> would evict); needing 30
    # fits. Verify the 30 case admits without eviction.
    def_small = ProviderDefinition(
        alias="lg-b",
        provider_type="mock",
        vram_required_bytes=30,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    session.add(def_small)
    session.commit()
    inst_small = ProviderInstance(
        agent_id=instance.agent_id,
        provider_definition_id=def_small.id,
        backend_status="stopped",
    )
    session.add(inst_small)
    session.commit()
    await scheduler.acquire("lg-b", "r2")
    # No stop of the lg-a instance (30 fit in the 30 remainder).
    assert (str(instance.id), "backend.stop") not in boot_calls


# ---------------------------------------------------------------------------
# Idle-timeout reaper
# ---------------------------------------------------------------------------
def make_reap_target(
    session: Session,
    *,
    machine_uid: str,
    alias: str,
    idle_timeout: int,
    last_request_age_seconds: float | None,
    connected: bool = True,
    backend_status: str = "running",
    vram_required: int = 50,
    backend_loaded_age_seconds: float | None = None,
) -> ProviderInstance:
    machine = session.query(Machine).filter(Machine.uid == machine_uid).first()
    if machine is None:
        machine = Machine(uid=machine_uid, name=machine_uid, host="127.0.0.1")
        machine.total_vram_bytes = 1000
        session.add(machine)
        session.commit()
    definition = ProviderDefinition(
        alias=alias,
        provider_type="mock",
        idle_timeout_seconds=idle_timeout,
        vram_required_bytes=vram_required,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    session.add(definition)
    session.commit()
    from datetime import UTC, datetime, timedelta

    last = (
        None
        if last_request_age_seconds is None
        else datetime.now(UTC) - timedelta(seconds=last_request_age_seconds)
    )
    loaded_at = (
        None
        if backend_loaded_age_seconds is None
        else datetime.now(UTC) - timedelta(seconds=backend_loaded_age_seconds)
    )
    ag = _agent(session, machine, connected=connected)
    instance = ProviderInstance(
        agent_id=ag.id,
        provider_definition_id=definition.id,
        backend_status=backend_status,
        last_request_at=last,
        backend_loaded_at=loaded_at,
    )
    session.add(instance)
    session.commit()
    session.refresh(instance)
    return instance


async def test_reaper_stops_idle_loaded_instance(
    session: Session, aredis, monkeypatch
) -> None:
    """Idle past timeout, 0 active slots -> backend.stop sent, _booted
    cleared, mirror updated."""
    instance = make_reap_target(
        session,
        machine_uid="reap-1",
        alias="rp-a",
        idle_timeout=10,
        last_request_age_seconds=30,
    )
    calls: list[tuple[str, str]] = []

    async def fake(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append((payload.get("instance_id", agent_id), type_))
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", fake)
    scheduler = InferenceScheduler(aredis)
    await scheduler._mark_booted(
        str(instance.id),
        "reap-1",
        session.get(ProviderDefinition, instance.provider_definition_id),
    )
    assert await scheduler._reap_idle_once() == 1
    assert (str(instance.id), "backend.stop") in calls
    assert str(instance.id) not in scheduler._booted
    assert await aredis.hgetall(redis_keys.vram_used_key("reap-1")) == {}
    with Session(engine) as s:
        assert s.get(ProviderInstance, instance.id).backend_status == "stopped"


async def test_reaper_does_not_stop_freshly_loaded_instance_with_stale_clock(
    session: Session, aredis, monkeypatch
) -> None:
    """The boot re-arms the idle clock: last_request_at predates the load,
    but backend_loaded_at is fresh -> nothing to reap yet."""
    instance = make_reap_target(
        session,
        machine_uid="reap-fresh",
        alias="rf-a",
        idle_timeout=10,
        last_request_age_seconds=30,
        backend_loaded_age_seconds=1,
    )
    calls: list[tuple[str, str]] = []

    async def fake(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append((payload.get("instance_id", agent_id), type_))
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", fake)
    scheduler = InferenceScheduler(aredis)
    await scheduler._mark_booted(
        str(instance.id),
        "reap-fresh",
        session.get(ProviderDefinition, instance.provider_definition_id),
    )
    assert await scheduler._reap_idle_once() == 0
    assert calls == []


async def test_reaper_stops_never_served_instance_from_load_clock(
    session: Session, aredis, monkeypatch
) -> None:
    """Never served a request (last_request_at None): the load clock alone
    decides — loaded past the timeout -> reap."""
    instance = make_reap_target(
        session,
        machine_uid="reap-loadonly",
        alias="rl-a",
        idle_timeout=10,
        last_request_age_seconds=None,
        backend_loaded_age_seconds=30,
    )
    calls: list[tuple[str, str]] = []

    async def fake(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append((payload.get("instance_id", agent_id), type_))
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", fake)
    scheduler = InferenceScheduler(aredis)
    await scheduler._mark_booted(
        str(instance.id),
        "reap-loadonly",
        session.get(ProviderDefinition, instance.provider_definition_id),
    )
    assert await scheduler._reap_idle_once() == 1
    assert (str(instance.id), "backend.stop") in calls


async def test_reaper_never_stops_zero_timeout(
    session: Session, aredis, monkeypatch
) -> None:
    """idle_timeout_seconds == 0 means never idle-stop."""
    instance = make_reap_target(
        session,
        machine_uid="reap-0",
        alias="rz-a",
        idle_timeout=0,
        last_request_age_seconds=100000,
    )
    calls: list[tuple[str, str]] = []

    async def fake(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append((payload.get("instance_id", agent_id), type_))
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", fake)
    scheduler = InferenceScheduler(aredis)
    await scheduler._mark_booted(
        str(instance.id),
        "reap-0",
        session.get(ProviderDefinition, instance.provider_definition_id),
    )
    assert await scheduler._reap_idle_once() == 0
    assert calls == []
    assert str(instance.id) in scheduler._booted


async def test_reaper_skips_instance_with_active_slots(
    session: Session, aredis, monkeypatch
) -> None:
    """An idle-past-timeout instance with an active in-process slot is not
    stopped."""
    instance = make_reap_target(
        session,
        machine_uid="reap-busy",
        alias="rb-a",
        idle_timeout=10,
        last_request_age_seconds=30,
    )
    calls: list[tuple[str, str]] = []

    async def fake(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append((payload.get("instance_id", agent_id), type_))
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", fake)
    scheduler = InferenceScheduler(aredis)
    # Occupy a slot, then backdate last_request_at so the instance looks
    # idle by TIME but is active by SLOT COUNT — the slot guard must save it.
    await scheduler.acquire("rb-a", "r1")
    from datetime import UTC, datetime, timedelta

    with Session(engine) as s:
        inst = s.get(ProviderInstance, instance.id)
        inst.last_request_at = datetime.now(UTC) - timedelta(seconds=30)
        s.add(inst)
        s.commit()
    assert scheduler._active_count_on_instance(str(instance.id)) == 1
    assert await scheduler._reap_idle_once() == 0
    assert (str(instance.id), "backend.stop") not in calls
    await scheduler.release("rb-a", "r1")


async def test_reaper_skips_disconnected_instance(
    session: Session, aredis, monkeypatch
) -> None:
    """A disconnected instance is never reaped."""
    instance = make_reap_target(
        session,
        machine_uid="reap-disc",
        alias="rd-a",
        idle_timeout=10,
        last_request_age_seconds=30,
        connected=False,
    )
    calls: list[tuple[str, str]] = []

    async def fake(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append((payload.get("instance_id", agent_id), type_))
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", fake)
    scheduler = InferenceScheduler(aredis)
    await scheduler._mark_booted(
        str(instance.id),
        "reap-disc",
        session.get(ProviderDefinition, instance.provider_definition_id),
    )
    assert await scheduler._reap_idle_once() == 0
    assert calls == []


async def test_reaper_survives_send_command_exception(
    session: Session, aredis, monkeypatch
) -> None:
    """A raising send_command is caught; the loop keeps running and stops
    the instance on a later tick."""
    instance = make_reap_target(
        session,
        machine_uid="reap-throw",
        alias="rt-a",
        idle_timeout=10,
        last_request_age_seconds=30,
    )
    state = {"raised": False}

    async def flaky(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        if not state["raised"] and type_ == "backend.stop":
            state["raised"] = True
            raise ConnectionError("socket gone")
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", flaky)
    scheduler = InferenceScheduler(aredis, idle_reaper_interval=0.05)
    await scheduler._mark_booted(
        str(instance.id),
        "reap-throw",
        session.get(ProviderDefinition, instance.provider_definition_id),
    )
    # First pass: the stop raises -> nothing stopped, still booted, task alive.
    assert await scheduler._reap_idle_once() == 0
    assert str(instance.id) in scheduler._booted
    # Drive the actual loop with the short interval so we exercise the
    # try/except-around-tick resilience end to end.
    await scheduler.start_background()
    try:
        await wait_until(lambda: str(instance.id) not in scheduler._booted, timeout=3.0)
    finally:
        await scheduler.stop()
    assert state["raised"]  # the exception path was hit
    assert str(instance.id) not in scheduler._booted


# ---------------------------------------------------------------------------
# B1: admission must never land on an instance under eviction/stop
# ---------------------------------------------------------------------------
async def test_try_admit_skips_instance_in_evicting(
    session: Session,
    aredis,
    boot_calls,  # noqa: ARG001
) -> None:
    """Focused invariant: an instance id present in ``_evicting`` is
    skipped by ``_try_admit`` on BOTH the already-booted and the
    needs-boot paths (the reaper's in-flight stops are covered too)."""
    _, _, instance = make_stack(
        session,
        machine_uid="sch-b1-unit",
        alias="b1u-a",
        total_vram=100,
        vram_required=50,
        backend_status="running",
    )
    scheduler = InferenceScheduler(aredis)
    definition = scheduler._definition("b1u-a")
    await scheduler._mark_booted(str(instance.id), "sch-b1-unit", definition)

    # Without an in-flight stop the already-booted instance admits.
    admission = await scheduler._try_admit("b1u-a", "r-ok")
    assert admission is not None and admission.instance_id == str(instance.id)
    await scheduler.release("b1u-a", "r-ok")

    # Put the instance under eviction: the same admit must skip it.
    scheduler._evicting.add(str(instance.id))
    admission = await scheduler._try_admit("b1u-a", "r-skip")
    assert admission is None
    assert scheduler.active_count("b1u-a") == 0
    scheduler._evicting.discard(str(instance.id))

    # Same guard on the needs-boot path (DB stopped, not in the ledger).
    await scheduler._mark_stopped(str(instance.id), "sch-b1-unit")
    with Session(engine) as s:
        inst = s.get(ProviderInstance, instance.id)
        inst.backend_status = "stopped"
        s.add(inst)
        s.commit()
    scheduler._evicting.add(str(instance.id))
    admission = await scheduler._try_admit("b1u-a", "r-boot-skip")
    assert admission is None
    assert (str(instance.id), "backend.start") not in boot_calls


async def test_concurrent_acquire_not_admitted_onto_eviction_victim(
    session: Session, aredis, monkeypatch
) -> None:
    """Race window: alias 'vb1' evicts idle instance V (alias 'va1') to
    make VRAM room; while V's ``backend.stop`` is in flight, a concurrent
    acquire for V's OWN alias must NOT be admitted onto V (the evictor
    holds the requester's alias lock, not V's). V gets stopped; the
    victim-alias request stays queued."""
    machine = Machine(uid="sch-b1-race", name="sch-b1-race", host="127.0.0.1")
    machine.total_vram_bytes = 100
    session.add(machine)
    def_a = ProviderDefinition(
        alias="va1",
        provider_type="mock",
        vram_required_bytes=80,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    def_b = ProviderDefinition(
        alias="vb1",
        provider_type="mock",
        vram_required_bytes=80,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    session.add_all([def_a, def_b])
    session.commit()
    inst_v = ProviderInstance(  # the victim (alias va1, running, idle)
        agent_id=_agent(session, machine).id,
        provider_definition_id=def_a.id,
        backend_status="running",
    )
    inst_b = ProviderInstance(
        agent_id=_agent(session, machine).id,
        provider_definition_id=def_b.id,
        backend_status="stopped",
    )
    session.add_all([inst_v, inst_b])
    session.commit()

    stop_gate = asyncio.Event()
    calls: list[tuple[str, str]] = []

    async def gated_send_command(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append((payload.get("instance_id", agent_id), type_))
        if type_ == "backend.stop" and payload.get("instance_id") == str(inst_v.id):
            await stop_gate.wait()  # hold the eviction window open
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", gated_send_command)
    scheduler = InferenceScheduler(aredis, queue_timeout=10.0)

    # A serves and releases: V keeps its 80-byte booted hold, idle.
    await scheduler.acquire("va1", "ra")
    await scheduler.release("va1", "ra")
    assert str(inst_v.id) in scheduler._booted

    # B's acquire must evict V; the stop blocks inside the window.
    tb = asyncio.create_task(scheduler.acquire("vb1", "rb"))
    await wait_until(lambda: str(inst_v.id) in scheduler._evicting)

    # Concurrent acquire for V's own alias while it is being evicted.
    ta = asyncio.create_task(scheduler.acquire("va1", "ra2"))
    await asyncio.sleep(0.2)  # give the admit path a chance to (wrongly) land
    assert scheduler.active_count("va1") == 0, (
        "B1: request admitted onto an instance mid-eviction"
    )
    assert scheduler._active_count_on_instance(str(inst_v.id)) == 0

    # Let the eviction complete: V stops, B boots into the freed space.
    stop_gate.set()
    admission_b = await asyncio.wait_for(tb, timeout=3)
    assert admission_b.instance_id == str(inst_b.id)
    assert (str(inst_v.id), "backend.stop") in calls
    assert str(inst_v.id) not in scheduler._booted

    # The queued victim-alias request was never admitted; cancel cleanly.
    ta.cancel()
    with pytest.raises(asyncio.CancelledError):
        await ta
    assert scheduler.active_count("va1") == 0
    await scheduler.release("vb1", "rb")


# ---------------------------------------------------------------------------
# N2: eviction victims include _booted instances with a stale DB status
# ---------------------------------------------------------------------------
async def test_eviction_can_reclaim_stopped_status_booted_instance(
    session: Session, aredis, monkeypatch
) -> None:
    """A ``_booted`` instance whose DB ``backend_status`` is stale
    "stopped" still holds VRAM in the merged ledger and must be evictable
    (different alias, idle, positive hold) instead of making the
    requester queue forever."""
    machine = Machine(uid="sch-n2", name="sch-n2", host="127.0.0.1")
    machine.total_vram_bytes = 100
    session.add(machine)
    def_a = ProviderDefinition(
        alias="n2a",
        provider_type="mock",
        vram_required_bytes=70,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    def_b = ProviderDefinition(
        alias="n2b",
        provider_type="mock",
        vram_required_bytes=70,
        capacity=1,
        backend_config={"model": {"file": "m.gguf"}},
    )
    session.add_all([def_a, def_b])
    session.commit()
    inst_stale = ProviderInstance(  # booted in-process, DB says stopped
        agent_id=_agent(session, machine).id,
        provider_definition_id=def_a.id,
        backend_status="stopped",
    )
    inst_target = ProviderInstance(
        agent_id=_agent(session, machine).id,
        provider_definition_id=def_b.id,
        backend_status="stopped",
    )
    session.add_all([inst_stale, inst_target])
    session.commit()

    calls: list[tuple[str, str]] = []

    async def fake(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append((payload.get("instance_id", agent_id), type_))
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", fake)
    scheduler = InferenceScheduler(aredis, queue_timeout=1.0)
    await scheduler._mark_booted(str(inst_stale.id), "sch-n2", def_a)
    # Sanity: the stale-booted hold counts against the machine.
    assert await scheduler._free_vram(machine, str(inst_target.id)) == 30

    admission = await scheduler.acquire("n2b", "rb")
    assert admission.instance_id == str(inst_target.id)
    assert (str(inst_stale.id), "backend.stop") in calls
    assert str(inst_stale.id) not in scheduler._booted
    assert str(inst_target.id) in scheduler._booted
    await scheduler.release("n2b", "rb")


# ---------------------------------------------------------------------------
# S2: out-of-band backend.status (stopped/error) prunes the VRAM hold
# ---------------------------------------------------------------------------
async def _deliver_backend_status(
    scheduler: InferenceScheduler, aredis, instance_id: str, backend_status: str
) -> None:
    """Push a backend.status frame through the real connection_manager
    handler with a minimal fake app/state (addressed to the owning agent)."""
    from types import SimpleNamespace

    from app.models import ProviderInstance as _PI
    from app.services.connection_manager import ConnectionState

    with Session(engine) as s:
        agent_id = str(s.get(_PI, uuid.UUID(instance_id)).agent_id)

    class _FakeWS:
        def __init__(self, app):
            self.app = app

        async def send_text(self, text):  # noqa: ARG001
            return None

    app = SimpleNamespace(state=SimpleNamespace(redis=aredis, scheduler=scheduler))
    state = ConnectionState(
        agent_id=agent_id,
        websocket=_FakeWS(app),
        epoch=1,
        connection_token="tok",
    )
    frame = Frame(
        type=FrameKind.BACKEND_STATUS,
        id="f-1",
        epoch=1,
        payload={"instance_id": instance_id, "backend_status": backend_status},
    )
    await manager.handle_inbound(app, state, frame)


async def test_external_stop_prunes_booted_hold_and_mirror(
    session: Session, aredis
) -> None:
    """A backend stopped out-of-band (admin UI / provider side / crash)
    emits backend.status=stopped; the scheduler's ``_booted`` hold and
    the Redis ``im:vram:used`` field must be dropped immediately so the
    machine isn't permanently over-counted."""
    _, _, instance = make_stack(
        session,
        machine_uid="sch-s2",
        alias="s2a",
        total_vram=100,
        vram_required=70,
        backend_status="running",
    )
    scheduler = InferenceScheduler(aredis)
    definition = scheduler._definition("s2a")
    await scheduler._mark_booted(str(instance.id), "sch-s2", definition)
    machine = session.get(Machine, instance.agent.machine_id)
    assert (await scheduler._machine_vram_held(machine))[str(instance.id)] == 70

    await _deliver_backend_status(
        scheduler, aredis, str(instance.id), BackendStatusValue.STOPPED
    )
    assert str(instance.id) not in scheduler._booted
    assert await aredis.hgetall(redis_keys.vram_used_key("sch-s2")) == {}
    # The freed space is visible to the next admission math.
    assert await scheduler._free_vram(machine, str(instance.id)) == 100
    # Idempotent: pruning again (not held) is a safe no-op.
    await scheduler.note_backend_stopped(str(instance.id), "sch-s2")
    assert str(instance.id) not in scheduler._booted


async def test_running_status_does_not_prune_booted_hold(
    session: Session, aredis
) -> None:
    """Only non-loaded states prune: a running backend keeps its hold."""
    _, _, instance = make_stack(
        session,
        machine_uid="sch-s2r",
        alias="s2ra",
        total_vram=100,
        vram_required=70,
        backend_status="running",
    )
    scheduler = InferenceScheduler(aredis)
    await scheduler._mark_booted(
        str(instance.id), "sch-s2r", scheduler._definition("s2ra")
    )
    await _deliver_backend_status(
        scheduler, aredis, str(instance.id), BackendStatusValue.RUNNING
    )
    assert str(instance.id) in scheduler._booted
    assert await aredis.hgetall(redis_keys.vram_used_key("sch-s2r")) == {
        str(instance.id): "70"
    }
    # Error state prunes as well.
    await _deliver_backend_status(
        scheduler, aredis, str(instance.id), BackendStatusValue.ERROR
    )
    assert str(instance.id) not in scheduler._booted


# ---------------------------------------------------------------------------
# Idle-timeout reaper
# ---------------------------------------------------------------------------
async def test_reaper_refreshes_mirror_ttl(
    session: Session, aredis, monkeypatch
) -> None:
    """Each tick refreshes im:vram:used TTL for this process's booted holds."""
    instance = make_reap_target(
        session,
        machine_uid="reap-ttl",
        alias="rtl-a",
        idle_timeout=0,  # never stopped, so it stays booted
        last_request_age_seconds=5,
    )

    async def fake(instance_id, type_, payload, timeout=30.0):  # noqa: ARG001
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", fake)
    scheduler = InferenceScheduler(aredis)
    defn = session.get(ProviderDefinition, instance.provider_definition_id)
    await scheduler._mark_booted(str(instance.id), "reap-ttl", defn)
    # Drop the TTL so we can see it get refreshed.
    await aredis.persist(redis_keys.vram_used_key("reap-ttl"))
    assert await aredis.ttl(redis_keys.vram_used_key("reap-ttl")) == -1
    await scheduler._reap_idle_once()
    ttl = await aredis.ttl(redis_keys.vram_used_key("reap-ttl"))
    assert 0 < ttl <= redis_keys.VRAM_USED_TTL_SECONDS


# ===========================================================================
# Phase 16 slice 4: per-agent max_running_backends cap + hot-swap eviction
#
# An agent is single-type; the scheduler counts its loaded backends and, at
# ProviderType.max_running_backends (0 = unlimited), hot-swaps an idle
# same-agent backend LRU-first before admitting a new one. VRAM is kept
# generous in these tests so the cap — not the machine budget — is the
# binding constraint.
# ===========================================================================
def capped_agent_stack(
    session: Session,
    *,
    machine_uid: str,
    provider_type: str,
    max_running: int,
    aliases: list[str],
    total_vram: int = 1000,
    vram_required: int = 10,
    first_status: str = "running",
) -> tuple[Machine, ProviderAgent, list[ProviderInstance]]:
    """One machine + one connected agent of ``provider_type`` (cap
    ``max_running``) hosting one backend per alias. The first backend is
    ``first_status`` (loaded by default), the rest ``stopped``."""
    make_provider_type(session, provider_type, max_running=max_running)
    machine = get_or_create_machine(session, uid=machine_uid, total_vram=total_vram)
    agent = make_agent(session, machine, provider_type=provider_type, connected=True)
    instances: list[ProviderInstance] = []
    for idx, alias in enumerate(aliases):
        definition = make_definition(
            session,
            alias=alias,
            provider_type=provider_type,
            vram_required=vram_required,
            capacity=1,
            backend_config={"model": {"file": "m.gguf"}},
        )
        status = first_status if idx == 0 else "stopped"
        instances.append(
            make_instance(session, agent, definition, backend_status=status)
        )
    return machine, agent, instances


async def test_max_running_one_hot_swaps_idle_backend(
    session: Session, aredis, boot_calls
) -> None:
    """cap=1: booting a 2nd backend evicts the idle 1st (LRU) then boots."""
    _, _, (inst_a, inst_b) = capped_agent_stack(
        session,
        machine_uid="mr-hot",
        provider_type="mrt1",
        max_running=1,
        aliases=["mra", "mrb"],
    )
    scheduler = InferenceScheduler(aredis, queue_timeout=1.0)
    admission = await scheduler.acquire("mrb", "rb")
    assert admission.instance_id == str(inst_b.id)
    # Idle A hot-swapped out, B booted in.
    assert (str(inst_a.id), "backend.stop") in boot_calls
    assert (str(inst_b.id), "backend.start") in boot_calls
    assert str(inst_a.id) not in scheduler._booted
    assert str(inst_b.id) in scheduler._booted


async def test_max_running_one_never_evicts_busy_backend(
    session: Session, aredis, boot_calls
) -> None:
    """cap=1: the 1st backend holds an ACTIVE slot -> not evictable; the 2nd
    boot falls through and the request stays queued (QueueTimeout)."""
    _, _, (inst_a, inst_b) = capped_agent_stack(
        session,
        machine_uid="mr-busy",
        provider_type="mrb1",
        max_running=1,
        aliases=["mrba", "mrbb"],
    )
    scheduler = InferenceScheduler(aredis, queue_timeout=0.3)
    # Occupy A with an active slot (A is already running; no boot). Admitting
    # onto the DB-running A adopts its hold into this process's ledger.
    await scheduler.acquire("mrba", "ra")
    assert scheduler._active_count_on_instance(str(inst_a.id)) == 1
    assert str(inst_a.id) in scheduler._booted
    with pytest.raises(QueueTimeout):
        await scheduler.acquire("mrbb", "rb")
    # Busy A was never stopped; B never booted.
    assert (str(inst_a.id), "backend.stop") not in boot_calls
    assert (str(inst_b.id), "backend.start") not in boot_calls
    assert str(inst_b.id) not in scheduler._booted
    await scheduler.release("mrba", "ra")


async def test_max_running_zero_is_unlimited(
    session: Session, aredis, boot_calls
) -> None:
    """cap=0 (unlimited): a 2nd backend boots alongside the idle 1st; no
    hot-swap, both resident."""
    _, _, (inst_a, inst_b) = capped_agent_stack(
        session,
        machine_uid="mr-zero",
        provider_type="mrz0",
        max_running=0,
        aliases=["mrza", "mrzb"],
    )
    scheduler = InferenceScheduler(aredis, queue_timeout=1.0)
    admission = await scheduler.acquire("mrzb", "rb")
    assert admission.instance_id == str(inst_b.id)
    # Unlimited: booting B alongside the idle A must NOT hot-swap A out.
    assert (str(inst_a.id), "backend.stop") not in boot_calls
    assert (str(inst_b.id), "backend.start") in boot_calls
    assert str(inst_b.id) in scheduler._booted


async def test_max_running_hot_swap_is_same_agent_only(
    session: Session, aredis, boot_calls
) -> None:
    """The hot-swap victim is scoped to the requesting agent: an idle backend
    on a DIFFERENT agent of the same type is never touched."""
    make_provider_type(session, "mrx", max_running=1)
    machine = get_or_create_machine(session, uid="mr-xagent", total_vram=1000)
    agent_a = make_agent(session, machine, provider_type="mrx", connected=True)
    agent_b = make_agent(session, machine, provider_type="mrx", connected=True)
    # Agent A: idle running backend (victim) + a stopped target.
    def_a1 = make_definition(
        session, alias="xa1", provider_type="mrx", vram_required=10,
        backend_config={"model": {"file": "m.gguf"}},
    )
    def_a2 = make_definition(
        session, alias="xa2", provider_type="mrx", vram_required=10,
        backend_config={"model": {"file": "m.gguf"}},
    )
    # Agent B: an idle running backend that must be left alone.
    def_b1 = make_definition(
        session, alias="xb1", provider_type="mrx", vram_required=10,
        backend_config={"model": {"file": "m.gguf"}},
    )
    inst_a1 = make_instance(session, agent_a, def_a1, backend_status="running")
    inst_a2 = make_instance(session, agent_a, def_a2, backend_status="stopped")
    inst_b1 = make_instance(session, agent_b, def_b1, backend_status="running")

    scheduler = InferenceScheduler(aredis, queue_timeout=1.0)
    admission = await scheduler.acquire("xa2", "ra2")
    assert admission.instance_id == str(inst_a2.id)
    # Same-agent idle victim A1 hot-swapped out; cross-agent B1 untouched.
    assert (str(inst_a1.id), "backend.stop") in boot_calls
    assert (str(inst_b1.id), "backend.stop") not in boot_calls
    assert (str(inst_a2.id), "backend.start") in boot_calls


async def test_max_running_hot_swap_lru_order_same_agent(
    session: Session, aredis, boot_calls
) -> None:
    """cap=2 with two idle same-agent backends and a 3rd boot: only the LRU
    (oldest last_request_at) is hot-swapped out to get under the cap."""
    from datetime import UTC, datetime, timedelta

    make_provider_type(session, "mrl", max_running=2)
    machine = get_or_create_machine(session, uid="mr-lru", total_vram=1000)
    agent = make_agent(session, machine, provider_type="mrl", connected=True)
    now = datetime.now(UTC)
    defs = {}
    insts = {}
    for alias, age in (("lra", 200), ("lrb", 5)):
        defs[alias] = make_definition(
            session, alias=alias, provider_type="mrl", vram_required=10,
            backend_config={"model": {"file": "m.gguf"}},
        )
        insts[alias] = make_instance(
            session,
            agent,
            defs[alias],
            backend_status="running",
            last_request_at=now - timedelta(seconds=age),
        )
    def_c = make_definition(
        session, alias="lrc", provider_type="mrl", vram_required=10,
        backend_config={"model": {"file": "m.gguf"}},
    )
    inst_c = make_instance(session, agent, def_c, backend_status="stopped")

    scheduler = InferenceScheduler(aredis, queue_timeout=1.0)
    admission = await scheduler.acquire("lrc", "rc")
    assert admission.instance_id == str(inst_c.id)
    # Two running (lra, lrb) at cap=2; booting a 3rd needs running<2, so the
    # single LRU (lra, oldest) is evicted and lrb is left resident.
    assert (str(insts["lra"].id), "backend.stop") in boot_calls
    assert (str(insts["lrb"].id), "backend.stop") not in boot_calls
    assert (str(inst_c.id), "backend.start") in boot_calls


# ===========================================================================
# Phase 16 slice 4: proactive init warm-up on agent connect
# ===========================================================================
def warmup_agent_stack(
    session: Session,
    *,
    machine_uid: str,
    provider_type: str,
    max_running: int,
    aliases: list[str],
    total_vram: int = 1000,
    vram_required: int = 10,
    last_request_ages: dict[str, float] | None = None,
) -> tuple[ProviderAgent, dict[str, ProviderInstance]]:
    """A connected agent with N stopped backends (one per alias) for warm-up."""
    make_provider_type(session, provider_type, max_running=max_running)
    machine = get_or_create_machine(session, uid=machine_uid, total_vram=total_vram)
    agent = make_agent(session, machine, provider_type=provider_type, connected=True)
    from datetime import UTC, datetime, timedelta

    insts: dict[str, ProviderInstance] = {}
    for alias in aliases:
        definition = make_definition(
            session,
            alias=alias,
            provider_type=provider_type,
            vram_required=vram_required,
            capacity=1,
            backend_config={"model": {"file": "m.gguf"}},
        )
        age = (last_request_ages or {}).get(alias)
        last = None if age is None else datetime.now(UTC) - timedelta(seconds=age)
        insts[alias] = make_instance(
            session, agent, definition, backend_status="stopped", last_request_at=last
        )
    return agent, insts


async def test_warm_up_boots_up_to_cap_serialized(
    session: Session, aredis, monkeypatch
) -> None:
    """cap=2 of 4 assigned backends: exactly 2 boot, one at a time (never
    concurrently), the rest stay stopped for on-demand boot."""
    agent, insts = warmup_agent_stack(
        session,
        machine_uid="wu-cap",
        provider_type="wucap",
        max_running=2,
        aliases=["wua", "wub", "wuc", "wud"],
    )
    calls: list[tuple[str, str]] = []
    state = {"in_flight": 0, "max_in_flight": 0}

    async def fake(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        if type_ == "backend.start":
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            await asyncio.sleep(0.02)  # widen any concurrency window
            state["in_flight"] -= 1
        calls.append((payload.get("instance_id", agent_id), type_))
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", fake)
    scheduler = InferenceScheduler(aredis)
    await scheduler.warm_up_agent(str(agent.id))

    starts = [c for c in calls if c[1] == "backend.start"]
    assert len(starts) == 2, f"expected exactly cap=2 boots, got {starts}"
    assert state["max_in_flight"] == 1, "warm-up boots must be serialized"
    # Exactly two of the four are now held; two remain stopped.
    assert len([i for i in insts.values() if str(i.id) in scheduler._booted]) == 2


async def test_warm_up_bounded_by_vram(
    session: Session, aredis, monkeypatch
) -> None:
    """Unlimited cap but tight VRAM: warm-up stops at the first backend that
    does not fit (no eviction during warm-up)."""
    agent, _ = warmup_agent_stack(
        session,
        machine_uid="wu-vram",
        provider_type="wuvram",
        max_running=0,  # unlimited cap -> VRAM is the only bound
        aliases=["wva", "wvb", "wvc"],
        total_vram=100,
        vram_required=60,
    )
    calls: list[tuple[str, str]] = []

    async def fake(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append((payload.get("instance_id", agent_id), type_))
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", fake)
    scheduler = InferenceScheduler(aredis)
    await scheduler.warm_up_agent(str(agent.id))
    # 60 fits (free 100); the next needs 60 but only 40 remains -> stop.
    assert [c for c in calls if c[1] == "backend.start"] and len(
        [c for c in calls if c[1] == "backend.start"]
    ) == 1
    # Warm-up never evicts.
    assert not [c for c in calls if c[1] == "backend.stop"]


async def test_warm_up_order_is_most_recently_used(
    session: Session, aredis, monkeypatch
) -> None:
    """Warm-up boots MRU-first (last_request_at desc), never-requested last."""
    agent, insts = warmup_agent_stack(
        session,
        machine_uid="wu-order",
        provider_type="wuord",
        max_running=0,  # unlimited so all three boot
        aliases=["wo-old", "wo-new", "wo-none"],
        last_request_ages={"wo-old": 500, "wo-new": 10},  # wo-none never requested
    )
    order: list[str] = []

    async def fake(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        if type_ == "backend.start":
            order.append(payload.get("instance_id"))
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", fake)
    scheduler = InferenceScheduler(aredis)
    await scheduler.warm_up_agent(str(agent.id))
    assert order == [
        str(insts["wo-new"].id),
        str(insts["wo-old"].id),
        str(insts["wo-none"].id),
    ]


async def test_warm_up_skips_already_running_and_is_guarded(
    session: Session, aredis, monkeypatch
) -> None:
    """An already-loaded backend is not re-booted; concurrent warm-up passes
    for the same agent collapse to one (per-agent ``_warming`` guard)."""
    make_provider_type(session, "wug", max_running=0)
    machine = get_or_create_machine(session, uid="wu-guard", total_vram=1000)
    agent = make_agent(session, machine, provider_type="wug", connected=True)
    def_run = make_definition(
        session, alias="grun", provider_type="wug", vram_required=10,
        backend_config={"model": {"file": "m.gguf"}},
    )
    def_stop = make_definition(
        session, alias="gstop", provider_type="wug", vram_required=10,
        backend_config={"model": {"file": "m.gguf"}},
    )
    inst_run = make_instance(session, agent, def_run, backend_status="running")
    inst_stop = make_instance(session, agent, def_stop, backend_status="stopped")

    calls: list[tuple[str, str]] = []

    async def fake(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append((payload.get("instance_id", agent_id), type_))
        await asyncio.sleep(0.02)
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", fake)
    scheduler = InferenceScheduler(aredis)
    # Two passes concurrently: the second must no-op on the guard.
    await asyncio.gather(
        scheduler.warm_up_agent(str(agent.id)),
        scheduler.warm_up_agent(str(agent.id)),
    )
    starts = [c for c in calls if c[1] == "backend.start"]
    # Only the stopped backend boots; the running one is skipped; the guard
    # prevents a double pass.
    assert starts == [(str(inst_stop.id), "backend.start")]
    assert (str(inst_run.id), "backend.start") not in calls


async def test_warm_up_survives_boot_failure(
    session: Session, aredis, monkeypatch
) -> None:
    """A refused backend.start does not abort the whole warm-up pass; the
    next assigned backend still boots."""
    agent, insts = warmup_agent_stack(
        session,
        machine_uid="wu-fail",
        provider_type="wufail",
        max_running=0,
        aliases=["wfa", "wfb"],
    )
    calls: list[tuple[str, str]] = []

    async def fake(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        iid = payload.get("instance_id", agent_id)
        calls.append((iid, type_))
        if type_ == "backend.start" and iid == str(insts["wfa"].id):
            return Frame(type="ack", payload={"ok": False, "error": "boom"})
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", fake)
    scheduler = InferenceScheduler(aredis)
    await scheduler.warm_up_agent(str(agent.id))
    # wfa failed (not held), wfb still warmed.
    assert (str(insts["wfa"].id), "backend.start") in calls
    assert (str(insts["wfb"].id), "backend.start") in calls
    assert str(insts["wfa"].id) not in scheduler._booted
    assert str(insts["wfb"].id) in scheduler._booted


async def test_warm_up_skips_backend_booted_by_concurrent_request(
    session: Session, aredis, monkeypatch
) -> None:
    """TOCTOU (Low): a backend a concurrent request booted after the warm-up
    target list was built must NOT get a redundant backend.start — the
    in-lock re-check skips it (mirrors _try_admit's needs_boot guard)."""
    # Two stopped backends, unlimited cap; MRU tiebreak orders toca before
    # tocb (both never-requested -> alias asc).
    agent, insts = warmup_agent_stack(
        session,
        machine_uid="wu-toctou",
        provider_type="wutoct",
        max_running=0,
        aliases=["toca", "tocb"],
    )
    gate = asyncio.Event()
    starts: list[str] = []

    async def fake(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        iid = payload.get("instance_id", agent_id)
        if type_ == "backend.start":
            starts.append(iid)
            if iid == str(insts["toca"].id):
                await gate.wait()  # park warm-up inside toca's boot
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", fake)
    scheduler = InferenceScheduler(aredis)
    warm = asyncio.create_task(scheduler.warm_up_agent(str(agent.id)))
    # Warm-up has begun booting toca and is parked; tocb is still a target.
    await wait_until(lambda: str(insts["toca"].id) in starts)
    # A concurrent request boots tocb (different alias -> no lock contention).
    await scheduler.acquire("tocb", "rb")
    assert str(insts["tocb"].id) in scheduler._booted
    # Release warm-up: it finishes toca, then reaches tocb (now loaded) and
    # must skip it rather than re-boot.
    gate.set()
    await asyncio.wait_for(warm, timeout=3)
    # tocb booted exactly once (by the request), never by warm-up.
    assert starts.count(str(insts["tocb"].id)) == 1
    assert starts.count(str(insts["toca"].id)) == 1
    await scheduler.release("tocb", "rb")


async def test_warm_up_stops_when_agent_disconnects_mid_pass(
    session: Session, aredis, monkeypatch
) -> None:
    """Nit: when the agent's socket drops mid-warm-up the pass stops cleanly
    instead of failing every remaining backend.start with a ConnectionError."""
    agent, insts = warmup_agent_stack(
        session,
        machine_uid="wu-disc",
        provider_type="wudisc",
        max_running=0,
        aliases=["dca", "dcb", "dcc"],
    )
    starts: list[str] = []

    async def fake(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        iid = payload.get("instance_id", agent_id)
        if type_ == "backend.start":
            starts.append(iid)
            if iid == str(insts["dca"].id):
                # Simulate the socket dropping right after the first boot:
                # flip the mirrored liveness flag the warm-up loop re-reads.
                with Session(engine) as s:
                    row = s.get(ProviderAgent, agent.id)
                    row.websocket_connected = False
                    s.add(row)
                    s.commit()
        return Frame(type="ack", payload={"ok": True})

    monkeypatch.setattr(manager, "send_command", fake)
    scheduler = InferenceScheduler(aredis)
    await scheduler.warm_up_agent(str(agent.id))
    # Only the first backend booted; the disconnect broke the loop before dcb.
    assert starts == [str(insts["dca"].id)]
    assert str(insts["dcb"].id) not in scheduler._booted
    assert str(insts["dcc"].id) not in scheduler._booted


# ---------------------------------------------------------------------------
# Queue observability & operator clear (Phase 19 S1)
#
# ``queue_snapshot`` reads the authoritative in-process state; ``clear_queue``
# drains *waiters only* (admitted slots, boots, and VRAM holds untouched) and
# wakes each blocked ``acquire`` to raise ``QueueCleared``.
# ---------------------------------------------------------------------------
def _snapshot_for(snapshot: list[dict], alias: str) -> dict:
    return next(row for row in snapshot if row["alias"] == alias)


async def test_queue_snapshot_counts_queued_and_active(
    session: Session, aredis, boot_calls  # noqa: ARG001
) -> None:
    make_stack(session, machine_uid="sch-snap", alias="snap-a", capacity=1)
    scheduler = InferenceScheduler(aredis)
    # An alias with no state is absent from the snapshot.
    assert await scheduler.queue_snapshot() == []

    await scheduler.acquire("snap-a", "r1")  # active
    r2_task = asyncio.create_task(scheduler.acquire("snap-a", "r2"))
    await wait_until(lambda: "r2" in scheduler._states["snap-a"].waiters)

    snapshot = await scheduler.queue_snapshot()
    row = _snapshot_for(snapshot, "snap-a")
    assert row == {"alias": "snap-a", "queued": 1, "active": 1}

    r2_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await r2_task

async def test_clear_queue_wakes_waiter_with_queue_cleared(
    session: Session, aredis, boot_calls  # noqa: ARG001
) -> None:
    make_stack(session, machine_uid="sch-clr", alias="clr-a", capacity=1)
    scheduler = InferenceScheduler(aredis)
    await scheduler.acquire("clr-a", "r1")  # admitted (untouched by clear)
    r2_task = asyncio.create_task(scheduler.acquire("clr-a", "r2"))
    await wait_until(lambda: "r2" in scheduler._states["clr-a"].waiters)

    cleared = await scheduler.clear_queue("clr-a")
    assert cleared == 1

    with pytest.raises(QueueCleared):
        await asyncio.wait_for(r2_task, timeout=3)

    # Admitted slot survives; only the waiter was drained.
    assert scheduler.active_count("clr-a") == 1
    assert await aredis.smembers(redis_keys.sched_active_key("clr-a")) == {"r1"}
    # No leaked waiter, no leaked cleared id (single cleanup point in
    # _drop_waiter).
    state = scheduler._states["clr-a"]
    assert list(state.waiters) == []
    assert state.cleared == set()
    await scheduler.release("clr-a", "r1")


async def test_clear_queue_all_across_aliases(
    session: Session, aredis, boot_calls  # noqa: ARG001
) -> None:
    make_stack(session, machine_uid="sch-clrall-1", alias="clra-1", capacity=1)
    make_stack(session, machine_uid="sch-clrall-2", alias="clra-2", capacity=1)
    scheduler = InferenceScheduler(aredis)
    await scheduler.acquire("clra-1", "a1")
    await scheduler.acquire("clra-2", "b1")
    ta = asyncio.create_task(scheduler.acquire("clra-1", "a2"))
    tb = asyncio.create_task(scheduler.acquire("clra-2", "b2"))
    await wait_until(lambda: "a2" in scheduler._states["clra-1"].waiters)
    await wait_until(lambda: "b2" in scheduler._states["clra-2"].waiters)

    assert await scheduler.clear_queue() == 2  # alias=None -> clear-all

    with pytest.raises(QueueCleared):
        await asyncio.wait_for(ta, timeout=3)
    with pytest.raises(QueueCleared):
        await asyncio.wait_for(tb, timeout=3)
    assert scheduler.active_count("clra-1") == 1
    assert scheduler.active_count("clra-2") == 1
    await scheduler.release("clra-1", "a1")
    await scheduler.release("clra-2", "b1")


async def test_clear_empty_or_unknown_queue_is_noop(
    session: Session, aredis, boot_calls  # noqa: ARG001
) -> None:
    make_stack(session, machine_uid="sch-clrempty", alias="clre-a", capacity=1)
    scheduler = InferenceScheduler(aredis)
    # Unknown alias (no state) -> 0.
    assert await scheduler.clear_queue("ghost") == 0
    # Admitted but no waiters -> 0 (empty queue).
    await scheduler.acquire("clre-a", "r1")
    assert await scheduler.clear_queue("clre-a") == 0
    assert scheduler.active_count("clre-a") == 1
    # Idempotent: a second clear is still 0.
    assert await scheduler.clear_queue("clre-a") == 0
    await scheduler.release("clre-a", "r1")


async def test_clear_queue_removes_redis_mirror_entries(
    session: Session, aredis, boot_calls  # noqa: ARG001
) -> None:
    make_stack(session, machine_uid="sch-clrmir", alias="clrm-a", capacity=1)
    scheduler = InferenceScheduler(aredis)
    await scheduler.acquire("clrm-a", "r1")
    r2_task = asyncio.create_task(scheduler.acquire("clrm-a", "r2"))
    await wait_until(lambda: "r2" in scheduler._states["clrm-a"].waiters)

    # Mirror populated before the clear.
    async def queue_mirrored() -> bool:
        return await aredis.lrange(redis_keys.sched_queue_key("clrm-a"), 0, -1) == [
            "r2"
        ]

    await wait_until(queue_mirrored)
    assert await aredis.exists(redis_keys.sched_wait_key("r2"))

    assert await scheduler.clear_queue("clrm-a") == 1
    with pytest.raises(QueueCleared):
        await asyncio.wait_for(r2_task, timeout=3)

    # Queue mirror + per-request wait key removed.
    assert await aredis.lrange(redis_keys.sched_queue_key("clrm-a"), 0, -1) == []
    assert not await aredis.exists(redis_keys.sched_wait_key("r2"))
    # Admitted r1's active set untouched.
    assert await aredis.smembers(redis_keys.sched_active_key("clrm-a")) == {"r1"}
    await scheduler.release("clrm-a", "r1")


async def test_queue_still_functional_after_clear(
    session: Session, aredis, boot_calls  # noqa: ARG001
) -> None:
    _, _, instance = make_stack(
        session, machine_uid="sch-clrreuse", alias="clrr-a", capacity=1
    )
    scheduler = InferenceScheduler(aredis)
    await scheduler.acquire("clrr-a", "r1")
    r2_task = asyncio.create_task(scheduler.acquire("clrr-a", "r2"))
    await wait_until(lambda: "r2" in scheduler._states["clrr-a"].waiters)
    assert await scheduler.clear_queue("clrr-a") == 1
    with pytest.raises(QueueCleared):
        await asyncio.wait_for(r2_task, timeout=3)

    # A fresh request after the clear queues normally and is admitted when the
    # held slot frees.
    r3_task = asyncio.create_task(scheduler.acquire("clrr-a", "r3"))
    await wait_until(lambda: "r3" in scheduler._states["clrr-a"].waiters)
    await scheduler.release("clrr-a", "r1")
    admission = await asyncio.wait_for(r3_task, timeout=3)
    assert admission.instance_id == str(instance.id)
    assert scheduler.active_count("clrr-a") == 1
    await scheduler.release("clrr-a", "r3")

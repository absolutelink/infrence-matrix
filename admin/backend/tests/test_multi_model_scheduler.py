"""Phase 25 scheduler: multi-model name -> single owning instance.

A served name resolves to its definition's ONE backend process. acquire/release
stay keyed by the requested NAME (per-name FIFO fairness) while boot/VRAM
operate on the owning instance: two names on one definition share one boot and
one VRAM hold (no double-hold, no re-boot), share the capacity pool, and are
never eviction victims of each other.
"""

import asyncio
import uuid

import pytest
import redis.asyncio as aioredis
from sqlmodel import Session

from app.models import (
    Machine,
    ProviderAgent,
    ProviderDefinition,
    ProviderInstance,
    ProviderModel,
)
from app.services.connection_manager import manager
from app.services.scheduler import InferenceScheduler, QueueTimeout
from app.services.wire import BackendStatusValue, Frame

TEST_REDIS_URL = __import__("os").environ["TEST_REDIS_URL"]


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


def _mm_stack(
    session: Session,
    *,
    machine_uid: str,
    alias: str,
    names: list[str],
    total_vram: int,
    vram_required: int,
    capacity: int,
    backend_status: str = "stopped",
) -> tuple[Machine, ProviderAgent, ProviderDefinition, ProviderInstance]:
    """One machine + agent + definition (multi-model) + one backend, with a
    ProviderModel row per served name."""
    machine = Machine(
        uid=machine_uid, name=machine_uid, host="127.0.0.1", total_vram_bytes=total_vram
    )
    session.add(machine)
    session.commit()
    agent = ProviderAgent(
        machine_id=machine.id,
        provider_type="mock",
        agent_id=f"ag-{uuid.uuid4().hex[:8]}",
        websocket_connected=True,
        base_port=8081,
    )
    session.add(agent)
    session.commit()
    definition = ProviderDefinition(
        alias=alias,
        provider_type="mock",
        modality="llm",
        backend_config={"model": {"file": "m.gguf"}},
        vram_required_bytes=vram_required,
        capacity=capacity,
    )
    session.add(definition)
    session.commit()
    for name in names:
        session.add(
            ProviderModel(
                definition_id=definition.id,
                name=name,
                modality="llm",
                backend_config={"slug": name},
                enabled=True,
            )
        )
    instance = ProviderInstance(
        agent_id=agent.id,
        provider_definition_id=definition.id,
        backend_status=backend_status,
    )
    session.add(instance)
    session.commit()
    return machine, agent, definition, instance


async def test_two_names_one_instance_single_boot_single_vram_hold(
    session: Session, aredis, boot_calls
) -> None:
    machine, _agent, definition, instance = _mm_stack(
        session,
        machine_uid="mm-boot",
        alias="mm-a",
        names=["mm-a", "mm-b"],
        total_vram=100,
        vram_required=60,
        capacity=2,
    )
    scheduler = InferenceScheduler(aredis)

    adm_a = await scheduler.acquire("mm-a", "req-a")
    assert adm_a.instance_id == str(instance.id)
    # One boot so far.
    starts = [c for c in boot_calls if c[1] == "backend.start"]
    assert len(starts) == 1
    assert str(instance.id) in scheduler._booted
    assert scheduler._booted[str(instance.id)] == ("mm-boot", 60)

    # Second served name on the SAME definition: no re-boot, no extra VRAM hold.
    adm_b = await scheduler.acquire("mm-b", "req-b")
    assert adm_b.instance_id == str(instance.id)
    starts = [c for c in boot_calls if c[1] == "backend.start"]
    assert len(starts) == 1, "second name must not re-boot the shared instance"
    assert len(scheduler._booted) == 1
    assert scheduler._booted[str(instance.id)] == ("mm-boot", 60)

    held = await scheduler._machine_vram_held(machine)
    assert held[str(instance.id)] == 60, "VRAM must be held once, not summed per name"

    await scheduler.release("mm-a", "req-a")
    await scheduler.release("mm-b", "req-b")
    assert definition.capacity == 2


async def test_shared_capacity_pool_across_names(
    session: Session,
    aredis,
    boot_calls,  # noqa: ARG001
) -> None:
    """capacity is a per-instance pool shared by every served name: with
    capacity=1 the second name cannot admit while the first holds the slot."""
    _mm_stack(
        session,
        machine_uid="mm-cap",
        alias="cap-a",
        names=["cap-a", "cap-b"],
        total_vram=1000,
        vram_required=0,
        capacity=1,
    )
    scheduler = InferenceScheduler(aredis, queue_timeout=0.2)
    await scheduler.acquire("cap-a", "req-a")
    # cap-b shares the same instance (at capacity) -> stays queued -> timeout.
    with pytest.raises(QueueTimeout):
        await scheduler.acquire("cap-b", "req-b")
    await scheduler.release("cap-a", "req-a")


async def test_per_name_fifo_state_is_separate(
    session: Session,
    aredis,
    boot_calls,  # noqa: ARG001
) -> None:
    _mm_stack(
        session,
        machine_uid="mm-fifo",
        alias="fifo-a",
        names=["fifo-a", "fifo-b"],
        total_vram=1000,
        vram_required=0,
        capacity=2,
    )
    scheduler = InferenceScheduler(aredis)
    await scheduler.acquire("fifo-a", "req-a")
    await scheduler.acquire("fifo-b", "req-b")
    # Each served name has its own queue state (per-name FIFO fairness).
    assert "fifo-a" in scheduler._states
    assert "fifo-b" in scheduler._states
    assert scheduler.active_count("fifo-a") == 1
    assert scheduler.active_count("fifo-b") == 1
    await scheduler.release("fifo-a", "req-a")
    await scheduler.release("fifo-b", "req-b")


async def test_served_name_resolves_to_owning_definition(
    session: Session, aredis
) -> None:
    _mm_stack(
        session,
        machine_uid="mm-res",
        alias="res-a",
        names=["res-a", "res-b"],
        total_vram=1000,
        vram_required=0,
        capacity=1,
    )
    scheduler = InferenceScheduler(aredis)
    # A non-alias served name still resolves to the definition + its instance.
    definition = scheduler._definition("res-b")
    assert definition is not None and definition.alias == "res-a"
    candidates = scheduler._candidates("res-b")
    assert len(candidates) == 1


async def test_eviction_excludes_same_definition_served_names(
    session: Session,
    aredis,
    boot_calls,  # noqa: ARG001
) -> None:
    """A served name that is NOT the definition alias must not make the
    definition's own instance an eviction victim of a sibling name."""
    machine, _agent, definition, instance = _mm_stack(
        session,
        machine_uid="mm-ev",
        alias="ev-a",
        names=["ev-a", "ev-b"],
        total_vram=100,
        vram_required=40,
        capacity=1,
        backend_status=BackendStatusValue.RUNNING,
    )
    scheduler = InferenceScheduler(aredis)
    await scheduler._mark_booted(str(instance.id), "mm-ev", definition)
    held = {str(instance.id): 40}
    # Requesting ev-b's space: the only instance on the machine belongs to the
    # SAME definition -> excluded (not a victim of its sibling ev-a).
    victims = scheduler._eviction_candidates(
        "mm-ev", definition.id, str(instance.id), held
    )
    assert victims == []
    # A different definition's instance IS a victim.
    other = ProviderDefinition(
        alias="ev-other",
        provider_type="mock",
        vram_required_bytes=40,
        capacity=1,
        backend_config={},
    )
    session.add(other)
    session.commit()
    other_inst = ProviderInstance(
        agent_id=_agent.id,
        provider_definition_id=other.id,
        backend_status=BackendStatusValue.RUNNING,
    )
    session.add(other_inst)
    session.commit()
    held[str(other_inst.id)] = 40
    victims = scheduler._eviction_candidates(
        "mm-ev", definition.id, str(instance.id), held
    )
    assert [v["id"] for v in victims] == [str(other_inst.id)]


async def test_acquire_unknown_served_name_raises(session: Session, aredis) -> None:
    from app.services.scheduler import NoProviderAvailable

    _mm_stack(
        session,
        machine_uid="mm-unk",
        alias="unk-a",
        names=["unk-a", "unk-b"],
        total_vram=1000,
        vram_required=0,
        capacity=1,
    )
    scheduler = InferenceScheduler(aredis)
    with pytest.raises(NoProviderAvailable):
        await scheduler.acquire("does-not-exist", "req")


async def test_concurrent_sibling_names_boot_once(
    session: Session, aredis, monkeypatch
) -> None:
    """MEDIUM-1 regression: two concurrent acquires on two served names of one
    COLD multi-model definition must dispatch backend.start EXACTLY once (the
    per-instance boot lock serializes the boot decision) and hold VRAM once,
    while both requests still admit. Without the per-instance lock both names
    (each under its own per-name lock) would race into backend.start."""
    machine, _agent, definition, instance = _mm_stack(
        session,
        machine_uid="mm-conc",
        alias="conc-a",
        names=["conc-a", "conc-b"],
        total_vram=1000,
        vram_required=60,
        capacity=2,
        backend_status="stopped",
    )

    starts: list[str] = []

    async def fake_send_command(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        if type_ == "backend.start":
            # Real yield so the sibling served name reliably reaches its own
            # boot decision before this one marks the instance booted (makes
            # the test discriminating against the pre-fix code).
            await asyncio.sleep(0.01)
            starts.append(payload.get("instance_id"))
        return Frame(type="ack", payload={"ok": True, "detail": {"capacity": 2}})

    monkeypatch.setattr(manager, "send_command", fake_send_command)

    scheduler = InferenceScheduler(aredis)
    adm_a, adm_b = await asyncio.gather(
        scheduler.acquire("conc-a", "r-a"),
        scheduler.acquire("conc-b", "r-b"),
    )

    assert len(starts) == 1, f"expected exactly one backend.start, got {starts}"
    assert starts[0] == str(instance.id)
    # Both names admitted onto the ONE shared instance.
    assert adm_a.instance_id == str(instance.id)
    assert adm_b.instance_id == str(instance.id)
    # One VRAM hold (not summed per name).
    assert len(scheduler._booted) == 1
    assert scheduler._booted[str(instance.id)] == ("mm-conc", 60)
    assert scheduler.active_count("conc-a") == 1
    assert scheduler.active_count("conc-b") == 1
    held = await scheduler._machine_vram_held(machine)
    assert held[str(instance.id)] == 60
    assert definition.capacity == 2

    await scheduler.release("conc-a", "r-a")
    await scheduler.release("conc-b", "r-b")


async def test_warmup_and_sibling_request_boot_once(
    session: Session, aredis, monkeypatch
) -> None:
    """Round-3 MEDIUM regression: a warm-up boot (keyed by definition.alias) and
    a concurrent request on a SIBLING served name hold DIFFERENT per-name
    state.locks, so only the shared per-instance boot lock serializes them. On a
    cold multi-model instance exactly ONE backend.start must be dispatched.
    Without the warm-up boot-lock wrap both paths race into backend.start."""
    machine, agent, definition, instance = _mm_stack(
        session,
        machine_uid="mm-wu",
        alias="wu-a",
        names=["wu-a", "wu-b"],
        total_vram=1000,
        vram_required=60,
        capacity=2,
        backend_status="stopped",
    )

    starts: list[str] = []

    async def fake_send_command(agent_id, type_, payload, timeout=30.0):  # noqa: ARG001
        if type_ == "backend.start":
            # Real yield so the other boot path reliably reaches its decision
            # before this one marks the instance booted (discriminating).
            await asyncio.sleep(0.01)
            starts.append(payload.get("instance_id"))
        return Frame(type="ack", payload={"ok": True, "detail": {"capacity": 2}})

    monkeypatch.setattr(manager, "send_command", fake_send_command)

    scheduler = InferenceScheduler(aredis)
    # Fire warm-up (boots via definition.alias 'wu-a') concurrently with a
    # request on the sibling name 'wu-b' — both target the same cold instance.
    _warm, adm_b = await asyncio.gather(
        scheduler.warm_up_agent(str(agent.id)),
        scheduler.acquire("wu-b", "r-b"),
    )

    assert len(starts) == 1, f"expected exactly one backend.start, got {starts}"
    assert starts[0] == str(instance.id)
    # The sibling request still admitted onto the one shared instance.
    assert adm_b.instance_id == str(instance.id)
    assert len(scheduler._booted) == 1
    assert scheduler._booted[str(instance.id)] == ("mm-wu", 60)
    assert machine.total_vram_bytes == 1000
    assert definition.capacity == 2

    await scheduler.release("wu-b", "r-b")


async def test_boot_lock_pruned_on_stop(aredis) -> None:
    """Round-3 LOW: the per-instance boot lock is reclaimed on stop so the dict
    does not grow unboundedly across boot/stop churn — but ONLY when not held
    (dropping a held lock would let a fresh _boot_lock_for mint a different lock
    and re-open the double-boot race)."""
    scheduler = InferenceScheduler(aredis)
    iid = "inst-prune"

    # Mint a lock via the boot-path helper (as a boot would).
    lock = scheduler._boot_lock_for(iid)
    assert iid in scheduler._boot_locks

    # While the lock is held, _mark_stopped must NOT prune it.
    await lock.acquire()
    await scheduler._mark_stopped(iid, "m-prune")
    assert iid in scheduler._boot_locks, "held lock must be kept"

    # Once released, the next stop reclaims it.
    lock.release()
    await scheduler._mark_stopped(iid, "m-prune")
    assert iid not in scheduler._boot_locks, "uncontended lock pruned on stop"

    # Idempotent: stopping an instance with no lock entry is a no-op.
    await scheduler._mark_stopped("never-seen", "m-prune")

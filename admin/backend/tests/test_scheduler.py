"""InferenceScheduler: FIFO admission, capacity, VRAM, boot, timeout,
idempotent + cancellation-safe release.

The provider WS command path is simulated by monkeypatching
``connection_manager.manager.send_command`` (same pattern as
``test_metrics_ownership.py``). Instances are seeded directly in the UTF8
test DB with ``websocket_connected=True``.
"""

import asyncio
import inspect
import time
import uuid
from collections import deque

import pytest
import redis.asyncio as aioredis
from sqlmodel import Session, select

from app.models import Machine, ProviderDefinition, ProviderInstance
from app.services import redis_keys
from app.services.connection_manager import manager
from app.services.scheduler import (
    InferenceScheduler,
    NoProviderAvailable,
    QueueTimeout,
)
from app.services.wire import Frame

TEST_REDIS_URL = __import__("os").environ["TEST_REDIS_URL"]


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
    existing = session.exec(select(Machine).where(Machine.uid == machine_uid)).first()
    if existing is None:
        existing = Machine(
            uid=machine_uid, name=f"name-{machine_uid}", host="127.0.0.1"
        )
        session.add(existing)
    existing.total_vram_bytes = total_vram
    session.add(existing)
    session.commit()
    definition = ProviderDefinition(
        alias=alias,
        provider_type="mock",
        registration_token=f"tok-{alias}",
        vram_required_bytes=vram_required,
        capacity=capacity,
    )
    session.add(definition)
    session.commit()
    instance = ProviderInstance(
        machine_id=existing.id,
        provider_definition_id=definition.id,
        port=port,
        backend_status=backend_status,
        websocket_connected=connected,
    )
    session.add(instance)
    session.commit()
    session.refresh(instance)
    return existing, definition, instance


@pytest.fixture
async def aredis():
    client = aioredis.from_url(TEST_REDIS_URL, decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def boot_calls(monkeypatch):
    """Record backend.start commands and ack ok."""
    calls: list[tuple[str, str]] = []

    async def fake_send_command(instance_id, type_, payload, timeout=30.0):  # noqa: ARG001
        calls.append((instance_id, type_))
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
# VRAM admission
# ---------------------------------------------------------------------------
async def test_vram_second_alias_cannot_fit_same_machine(
    session: Session, aredis, boot_calls
) -> None:
    machine = Machine(uid="sch-vram", name="sch-vram", host="127.0.0.1")
    machine.total_vram_bytes = 100
    session.add(machine)
    def_a = ProviderDefinition(
        alias="va",
        provider_type="mock",
        registration_token="tok-va",
        vram_required_bytes=80,
        capacity=1,
    )
    def_b = ProviderDefinition(
        alias="vb",
        provider_type="mock",
        registration_token="tok-vb",
        vram_required_bytes=80,
        capacity=1,
    )
    session.add_all([def_a, def_b])
    session.commit()
    inst_a = ProviderInstance(
        machine_id=machine.id,
        provider_definition_id=def_a.id,
        backend_status="running",
        websocket_connected=True,
    )
    inst_b = ProviderInstance(
        machine_id=machine.id,
        provider_definition_id=def_b.id,
        backend_status="stopped",
        websocket_connected=True,
    )
    session.add_all([inst_a, inst_b])
    session.commit()

    scheduler = InferenceScheduler(aredis, queue_timeout=0.3)

    # A is admitted (already running; no boot).
    admission = await scheduler.acquire("va", "ra")
    assert admission.instance_id == str(inst_a.id)
    assert boot_calls == []
    used = await aredis.hgetall(redis_keys.vram_used_key("sch-vram"))
    assert used == {str(inst_a.id): "80"}

    # B needs 80 but only 20 is free (A's running backend holds 80):
    # queues and times out; no VRAM double-booked.
    with pytest.raises(QueueTimeout):
        await scheduler.acquire("vb", "rb")
    used = await aredis.hgetall(redis_keys.vram_used_key("sch-vram"))
    assert used == {str(inst_a.id): "80"}
    assert scheduler.active_count("vb") == 0

    # Release A -> B now fits and is admitted.
    await scheduler.release("va", "ra")
    admission_b = await scheduler.acquire("vb", "rb2")
    assert admission_b.instance_id == str(inst_b.id)
    assert (str(inst_b.id), "backend.start") in boot_calls
    used = await aredis.hgetall(redis_keys.vram_used_key("sch-vram"))
    assert used == {str(inst_b.id): "80"}


# ---------------------------------------------------------------------------
# Boot failure
# ---------------------------------------------------------------------------
async def test_boot_failure_does_not_admit_or_leak(
    session: Session, aredis, monkeypatch
) -> None:
    _, _, instance = make_stack(
        session, machine_uid="sch-bootfail", alias="bf-a", capacity=1
    )

    async def refuse_send_command(instance_id, type_, payload, timeout=30.0):  # noqa: ARG001
        return Frame(type="ack", payload={"ok": False, "error": "boot exploded"})

    monkeypatch.setattr(manager, "send_command", refuse_send_command)
    scheduler = InferenceScheduler(aredis, queue_timeout=0.3)

    with pytest.raises(QueueTimeout):
        await scheduler.acquire("bf-a", "r1")
    assert scheduler.active_count("bf-a") == 0
    assert await aredis.smembers(redis_keys.sched_active_key("bf-a")) == set()
    assert await aredis.hgetall(redis_keys.vram_used_key("sch-bootfail")) == {}

    # A working boot afterwards still admits.
    async def ok_send_command(instance_id, type_, payload, timeout=30.0):  # noqa: ARG001
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
        machine_id=machine2.id,
        provider_definition_id=definition.id,
        port=9091,
        backend_status="stopped",
        websocket_connected=True,
    )
    session.add(inst2)
    session.commit()

    async def flaky_send_command(instance_id, type_, payload, timeout=30.0):  # noqa: ARG001
        if instance_id != str(inst2.id):
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
    make_stack(session, machine_uid="sch-idem", alias="id-a", capacity=1)
    scheduler = InferenceScheduler(aredis)
    await scheduler.acquire("id-a", "r1")
    await scheduler.release("id-a", "r1")
    # Second and third release: no-ops, no exceptions.
    await scheduler.release("id-a", "r1")
    await scheduler.release("id-a", "never-admitted")
    assert scheduler.active_count("id-a") == 0
    assert await aredis.smembers(redis_keys.sched_active_key("id-a")) == set()
    assert await aredis.hgetall(redis_keys.vram_used_key("sch-idem")) == {}


async def test_release_inside_cancelled_task_still_cleans_redis(
    session: Session,
    aredis,
    boot_calls,  # noqa: ARG001
) -> None:
    make_stack(session, machine_uid="sch-cancel", alias="cx-a", capacity=1)
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
    # the Redis cleanup land.
    async def released_everywhere() -> bool:
        return (
            scheduler.active_count("cx-a") == 0
            and await aredis.smembers(redis_keys.sched_active_key("cx-a")) == set()
            and await aredis.hgetall(redis_keys.vram_used_key("sch-cancel")) == {}
        )

    await wait_until(released_everywhere)


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
        alias="ad-a", provider_type="mock", registration_token="tok-ad-a"
    )
    session.add(definition)
    session.commit()
    instance = ProviderInstance(
        machine_id=machine.id,
        provider_definition_id=definition.id,
        port=8081,
        backend_status="running",
        websocket_connected=True,
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

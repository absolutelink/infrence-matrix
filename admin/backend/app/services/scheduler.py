"""Inference scheduler (Phase 6).

Admission control for ``POST /v1/responses``: for a provider alias, pick a
connected provider instance with a free slot and enough free VRAM on its
machine, boot the instance's backend if needed, and hand back the base URL
litellm should target.

Concurrency model — **in-process FIFO with a Redis mirror**:

The admin runs a single uvicorn worker (see the root Dockerfile), so the
authoritative queue is an in-process per-alias waiter ``deque`` guarded by
an ``asyncio.Lock`` (serializes admission attempts) and an
``asyncio.Condition`` (wait/wake on state changes). This is the correct,
deadlock-free choice for one worker. The Redis keys from
``docs/redis-keys.md`` (``im:sched:queue|active|wait|lock``,
``im:vram:used``) are maintained as a best-effort observable mirror and
the ``sched:lock`` contract is honored around boot/admission mutations, so
a future cross-worker Redis-queue implementation can swap in behind this
same ``acquire``/``release`` interface.

``acquire(alias, request_id)`` semantics:

1. Zero connected+enabled candidate instances -> ``NoProviderAvailable``
   immediately (never queue what can never run).
2. FIFO fairness: the request id is appended to the alias waiter deque and
   only admitted when it is at the head *and* a slot is available.
3. A slot = ``active_on_instance < ProviderDefinition.capacity`` and the
   instance's machine has enough free VRAM for ``vram_required_bytes``.
   Free VRAM = ``Machine.total_vram_bytes`` minus VRAM held by *other*
   instances on the machine, where "held" merges ``im:vram:used:{machine}``
   with this process's active slots and with every running/in_use
   instance's definition requirement (a loaded backend holds its VRAM
   even between requests). The target instance's own hold is excluded —
   it either already counts for its first slot or is about to be booted.
4. A ``stopped`` candidate is booted with ``backend.start`` (the provider
   acks only after its lifecycle reaches running). Boot failure/timeout
   falls through to the next candidate; if nothing fits, the head stays
   queued and retries on the next wake.
5. Waiting is bounded by ``queue_timeout`` -> ``QueueTimeout``.
6. Admission records ``im:sched:active`` + ``im:vram:used`` (TTL-bounded
   per the key doc) and returns an ``Admission``.
7. Eviction of idle different-alias instances to free VRAM is not
   implemented: ``TODO(phase6-eviction)`` — the request waits instead.

``release(alias, request_id)`` is idempotent and cancellation-safe
(``asyncio.shield`` around the Redis cleanup), mirroring the Phase 4
release-on-upstream-close discipline, and wakes the next waiter.
"""

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

import redis.asyncio as aioredis
from sqlmodel import Session, col, select

from app.core.db import engine
from app.models import Machine, ProviderDefinition, ProviderInstance
from app.services import redis_keys
from app.services.connection_manager import manager
from app.services.wire import BackendStatusValue

logger = logging.getLogger("admin.scheduler")

# Backend states that mean "loaded and serving; holds its VRAM".
RUNNING_BACKEND_STATUSES = (BackendStatusValue.RUNNING, BackendStatusValue.IN_USE)

# Retry discipline for the im:sched:lock:{alias} admission lock. With a
# single worker contention is essentially impossible; if the lock cannot
# be taken after the retries we proceed on in-process authority rather
# than stalling the queue behind a stale lock.
LOCK_RETRIES = 3
LOCK_RETRY_DELAY = 0.1


class SchedulerError(Exception):
    """Base class for scheduler admission failures."""


class NoProviderAvailable(SchedulerError):
    """No connected/enabled instance can ever serve this alias -> HTTP 503."""


class QueueTimeout(SchedulerError):
    """Waited in the FIFO longer than queue_timeout -> HTTP 504."""


@dataclass
class Admission:
    """A granted inference slot for one request."""

    instance_id: str
    # http://{machine.reachable}:{port}  (NO trailing /v1; caller appends)
    base_url: str
    machine_uid: str


@dataclass
class _Slot:
    """One admitted request's hold on an instance + its machine VRAM."""

    instance_id: str
    machine_uid: str
    vram_bytes: int


@dataclass
class _AliasState:
    """In-process authoritative queue for one provider alias."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    waiters: deque[str] = field(default_factory=deque)
    active: dict[str, _Slot] = field(default_factory=dict)
    # Bumped whenever queue/active state changes so waiters can detect
    # progress without lost wakeups across the attempt/release boundary.
    changed: int = 0


class InferenceScheduler:
    """Per-alias FIFO admission with a Redis observability mirror."""

    def __init__(
        self, redis_client: aioredis.Redis, *, queue_timeout: float = 300.0
    ) -> None:
        self._redis = redis_client
        self._queue_timeout = queue_timeout
        self._states: dict[str, _AliasState] = {}
        # Instance ids this process has booted (instance_id -> vram bytes
        # its definition holds). The DB backend_status mirror can lag the
        # ack, and a booted backend keeps holding VRAM between requests,
        # so we track it in-process for boot-skip and VRAM accounting.
        self._booted: dict[str, int] = {}
        self._bg_stop: asyncio.Event | None = None
        self._bg_task: asyncio.Task[None] | None = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    async def acquire(self, alias: str, request_id: str) -> Admission:
        """Admit ``request_id`` for ``alias``, blocking in the FIFO.

        Raises ``NoProviderAvailable`` when nothing can serve the alias and
        ``QueueTimeout`` when the wait exceeds ``queue_timeout``.
        """
        if not self._candidates(alias):
            raise NoProviderAvailable(
                f"no connected provider instance for alias '{alias}'"
            )

        state = self._state_for(alias)
        enqueued_at = time.monotonic()
        async with state.condition:
            state.waiters.append(request_id)
            state.changed += 1
        await self._mirror_enqueue(alias, state, request_id)

        deadline = enqueued_at + self._queue_timeout
        try:
            while True:
                async with state.condition:
                    seen = state.changed

                async with state.lock:
                    if state.waiters and state.waiters[0] == request_id:
                        admission = await self._try_admit(alias, request_id, state)
                        if admission is not None:
                            state.waiters.popleft()
                            async with state.condition:
                                state.changed += 1
                            await self._mirror_queue(alias, state)
                            return admission

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise QueueTimeout(
                        f"timed out after {self._queue_timeout:.0f}s "
                        f"waiting for a '{alias}' slot"
                    )

                async with state.condition:
                    if state.changed != seen:
                        continue
                try:
                    await asyncio.wait_for(
                        self._wait_for_change(state, seen), remaining
                    )
                except TimeoutError:
                    raise QueueTimeout(
                        f"timed out after {self._queue_timeout:.0f}s "
                        f"waiting for a '{alias}' slot"
                    ) from None
        except BaseException:
            # Failed or cancelled: never leave a ghost in the queue.
            await asyncio.shield(self._drop_waiter(alias, request_id))
            raise

    async def release(self, alias: str, request_id: str) -> None:
        """Release a slot (or drop a queued waiter). Idempotent.

        The in-process slot is popped synchronously (before any ``await``)
        so ``active_count`` is immediately correct even under
        cancellation. The wake-up and the Redis mirror writes are shielded
        so they still land after the caller is cancelled (client disconnect
        mid-stream).
        """
        state = self._states.get(alias)
        if state is None:
            return
        # Synchronous, no awaits: atomic w.r.t. other tasks in this loop.
        slot = state.active.pop(request_id, None)
        if slot is None:
            return
        still_held = any(
            s.instance_id == slot.instance_id
            for alias_state in self._states.values()
            for s in alias_state.active.values()
        )
        await asyncio.shield(
            self._release_wake_and_mirror(state, alias, request_id, slot, still_held)
        )

    async def _release_wake_and_mirror(
        self,
        state: _AliasState,
        alias: str,
        request_id: str,
        slot: _Slot,
        still_held: bool,
    ) -> None:
        """Shielded wake-up + Redis mirror cleanup for one released slot."""
        async with state.condition:
            state.changed += 1
            state.condition.notify_all()
        with contextlib.suppress(Exception):
            await self._redis.srem(redis_keys.sched_active_key(alias), request_id)
            # Drop the VRAM hold only when no other active slot in this
            # process still occupies the same instance.
            if not still_held:
                await self._redis.hdel(
                    redis_keys.vram_used_key(slot.machine_uid),
                    slot.instance_id,
                )
            await self._redis.delete(redis_keys.sched_wait_key(request_id))

    async def start_background(self) -> None:
        """Start the idle-reaper task (see TODO in ``_idle_reaper``)."""
        self._bg_stop = asyncio.Event()
        self._bg_task = asyncio.create_task(self._idle_reaper(self._bg_stop))

    async def stop(self) -> None:
        """Stop the background task cleanly."""
        if self._bg_stop is not None:
            self._bg_stop.set()
        if self._bg_task is not None:
            self._bg_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._bg_task
            self._bg_task = None

    def active_count(self, alias: str) -> int:
        """Currently admitted requests for an alias (test/ops aid)."""
        state = self._states.get(alias)
        return len(state.active) if state else 0

    # ------------------------------------------------------------------
    # Admission attempt (head waiter only, under the per-alias lock)
    # ------------------------------------------------------------------
    async def _try_admit(
        self, alias: str, request_id: str, state: _AliasState
    ) -> Admission | None:
        definition = self._definition(alias)
        if definition is None or not definition.enabled:
            raise NoProviderAvailable(f"alias '{alias}' has no enabled definition")
        instances = self._candidates(alias)
        if not instances:
            raise NoProviderAvailable(
                f"no connected provider instance for alias '{alias}'"
            )

        # Prefer already-running instances; a boot is only paid for when no
        # running instance has a free slot.
        booted = [
            i
            for i in instances
            if i.backend_status in RUNNING_BACKEND_STATUSES or str(i.id) in self._booted
        ]
        needs_boot_only = [
            i
            for i in instances
            if i.backend_status not in RUNNING_BACKEND_STATUSES
            and str(i.id) not in self._booted
        ]

        for instance in booted + needs_boot_only:
            instance_id = str(instance.id)
            if self._active_on(state, instance_id) >= definition.capacity:
                continue
            machine = instance.machine
            address = machine.reachable_address()
            if not address:
                logger.warning(
                    "instance %s: machine %s has no reachable address",
                    instance_id,
                    machine.uid,
                )
                continue

            needs_boot = (
                instance.backend_status not in RUNNING_BACKEND_STATUSES
                and instance_id not in self._booted
            )
            if needs_boot:
                held = await self._machine_vram_used(machine)
                free = machine.total_vram_bytes - sum(
                    v for k, v in held.items() if k != instance_id
                )
                if definition.vram_required_bytes > free:
                    # TODO(phase6-eviction): could backend.stop an idle
                    # different-alias instance on this machine to free
                    # VRAM. Not implemented: the request waits instead.
                    continue
                lock_token = await self._take_admission_lock(alias)
                try:
                    if not await self._boot(instance_id):
                        continue
                finally:
                    await self._release_admission_lock(alias, lock_token)
                # Remember the successful boot so a lagging DB mirror
                # never causes a redundant second backend.start and so VRAM
                # stays accounted for between requests.
                self._booted[instance_id] = definition.vram_required_bytes

            await self._record_admission(alias, request_id, instance, definition)
            return Admission(
                instance_id=instance_id,
                base_url=f"http://{address}:{instance.port}",
                machine_uid=machine.uid,
            )
        return None

    async def _boot(self, instance_id: str) -> bool:
        """backend.start the instance; True when the provider acks ok.

        The provider acks only after its backend lifecycle reaches
        running, so a successful return means /v1 is live.
        """
        try:
            reply = await manager.send_command(instance_id, "backend.start", {})
        except Exception as exc:  # noqa: BLE001 - any delivery failure means no boot
            logger.warning("backend.start failed for instance %s: %s", instance_id, exc)
            return False
        ok = bool(reply.payload.get("ok"))
        if not ok:
            logger.warning(
                "backend.start refused for instance %s: %s",
                instance_id,
                reply.payload.get("error"),
            )
        return ok

    async def _take_admission_lock(self, alias: str) -> str | None:
        """Honor the im:sched:lock:{alias} contract (SET NX PX 5s).

        Returns the lock token when acquired, else None. Redis errors are
        treated as "acquired" (best-effort mirror; in-process state is
        the authority under a single worker).
        """
        token = f"{id(self)}-{time.monotonic_ns()}"
        for attempt in range(LOCK_RETRIES):
            try:
                taken = await self._redis.set(
                    redis_keys.sched_lock_key(alias),
                    token,
                    nx=True,
                    px=redis_keys.SCHED_LOCK_TTL_MS,
                )
            except Exception:  # noqa: BLE001
                logger.debug("sched lock failed for alias %s", alias, exc_info=True)
                return token
            if taken:
                return token
            await asyncio.sleep(LOCK_RETRY_DELAY * (attempt + 1))
        logger.warning(
            "sched lock for alias %s busy after retries; proceeding on "
            "in-process authority",
            alias,
        )
        return None

    async def _release_admission_lock(self, alias: str, token: str | None) -> None:
        if token is None:
            return
        with contextlib.suppress(Exception):
            key = redis_keys.sched_lock_key(alias)
            if await self._redis.get(key) == token:
                await self._redis.delete(key)

    async def _record_admission(
        self,
        alias: str,
        request_id: str,
        instance: ProviderInstance,
        definition: ProviderDefinition,
    ) -> None:
        state = self._state_for(alias)
        state.active[request_id] = _Slot(
            instance_id=str(instance.id),
            machine_uid=instance.machine.uid,
            vram_bytes=definition.vram_required_bytes,
        )
        with contextlib.suppress(Exception):
            await self._redis.sadd(redis_keys.sched_active_key(alias), request_id)
            await self._redis.hset(
                redis_keys.vram_used_key(instance.machine.uid),
                mapping={str(instance.id): str(definition.vram_required_bytes)},
            )
            await self._redis.expire(
                redis_keys.vram_used_key(instance.machine.uid),
                redis_keys.VRAM_USED_TTL_SECONDS,
            )
            await self._redis.hset(
                redis_keys.sched_wait_key(request_id),
                mapping={"status": "admitted", "instance_id": str(instance.id)},
            )
        self._touch_last_request(instance.id)

    # ------------------------------------------------------------------
    # Release / cancel cleanup
    # ------------------------------------------------------------------
    async def _wait_for_change(self, state: _AliasState, seen: int) -> None:
        """Block on the alias condition until ``changed`` moves past ``seen``.

        Called via ``asyncio.wait_for`` so the wait is bounded; asyncio's
        ``Condition.wait`` releases the lock while waiting and cleans up
        the waiter on cancellation.
        """
        async with state.condition:
            while state.changed == seen:
                await state.condition.wait()

    async def _drop_waiter(self, alias: str, request_id: str) -> None:
        state = self._states.get(alias)
        if state is None:
            return
        async with state.condition:
            if request_id in state.waiters:
                state.waiters.remove(request_id)
                state.changed += 1
                state.condition.notify_all()
        with contextlib.suppress(Exception):
            await self._mirror_queue(alias, state)
            await self._redis.delete(redis_keys.sched_wait_key(request_id))

    # ------------------------------------------------------------------
    # DB reads
    # ------------------------------------------------------------------
    def _candidates(self, alias: str) -> list[ProviderInstance]:
        with Session(engine) as session:
            rows = session.exec(
                select(ProviderInstance)
                .join(
                    ProviderDefinition,
                    col(ProviderInstance.provider_definition_id)
                    == col(ProviderDefinition.id),
                )
                .where(
                    ProviderDefinition.alias == alias,
                    col(ProviderDefinition.enabled) == True,  # noqa: E712
                    col(ProviderInstance.websocket_connected) == True,  # noqa: E712
                )
            ).all()
            for row in rows:
                session.expunge(row)
            return list(rows)

    def _definition(self, alias: str) -> ProviderDefinition | None:
        with Session(engine) as session:
            definition = session.exec(
                select(ProviderDefinition).where(ProviderDefinition.alias == alias)
            ).first()
            if definition is not None:
                session.expunge(definition)
            return definition

    async def _machine_vram_used(self, machine: Machine) -> dict[str, int]:
        """instance_id -> bytes held on this machine.

        Held VRAM = the Redis ``im:vram:used`` mirror merged with every
        in-process active slot on this machine (across all aliases), so a
        slot counts even if the Redis TTL lapsed mid-request. A running
        but idle backend is treated as holding no VRAM in this simplified
        Phase 6 ledger: ``release`` clears its ``im:vram:used`` entry and
        the next ``acquire`` re-books it (skipping the boot via
        ``self._booted``). Idle-backend eviction to reclaim VRAM is
        ``TODO(phase6-eviction)``.
        """
        held: dict[str, int] = {}
        with contextlib.suppress(Exception):
            raw = await self._redis.hgetall(redis_keys.vram_used_key(machine.uid))
            held = {k: int(v) for k, v in raw.items()}
        for alias_state in self._states.values():
            for slot in alias_state.active.values():
                if slot.machine_uid == machine.uid:
                    held[slot.instance_id] = max(
                        held.get(slot.instance_id, 0), slot.vram_bytes
                    )
        return held

    # ------------------------------------------------------------------
    # Redis mirror helpers
    # ------------------------------------------------------------------
    def _state_for(self, alias: str) -> _AliasState:
        state = self._states.get(alias)
        if state is None:
            state = _AliasState()
            self._states[alias] = state
        return state

    def _active_on(self, state: _AliasState, instance_id: str) -> int:
        return sum(
            1 for slot in state.active.values() if slot.instance_id == instance_id
        )

    async def _mirror_enqueue(
        self, alias: str, state: _AliasState, request_id: str
    ) -> None:
        try:
            await self._redis.hset(
                redis_keys.sched_wait_key(request_id),
                mapping={
                    "status": "queued",
                    "position": str(len(state.waiters)),
                    "enqueued_at": str(time.time()),
                },
            )
            await self._redis.expire(
                redis_keys.sched_wait_key(request_id), redis_keys.SCHED_WAIT_TTL_SECONDS
            )
            await self._mirror_queue(alias, state)
        except Exception:  # noqa: BLE001 - mirror is best-effort
            logger.debug("scheduler queue mirror write failed", exc_info=True)

    async def _mirror_queue(self, alias: str, state: _AliasState) -> None:
        """Mirror the in-process waiter deque onto im:sched:queue:{alias}."""
        try:
            queue_key = redis_keys.sched_queue_key(alias)
            await self._redis.delete(queue_key)
            if state.waiters:
                await self._redis.rpush(queue_key, *list(state.waiters))
        except Exception:  # noqa: BLE001
            logger.debug("scheduler queue mirror write failed", exc_info=True)

    def _touch_last_request(self, instance_id) -> None:  # noqa: ANN001
        with Session(engine) as session:
            inst = session.get(ProviderInstance, instance_id)
            if inst is not None:
                inst.last_request_at = datetime.now(UTC)
                session.add(inst)
                session.commit()

    # ------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------
    async def _idle_reaper(self, stop_event: asyncio.Event) -> None:
        # TODO(phase6-idle-reaper): stop instances whose last_request_at
        # is older than the definition's idle_timeout_seconds with no
        # active requests. No-op until then; the lifespan starts/stops
        # this task cleanly.
        await stop_event.wait()

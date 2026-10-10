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
3. A slot = ``active_on_instance < ProviderDefinition.capacity``. VRAM
   is accounted **per booted instance**, not per request: a loaded
   backend keeps its weights resident between requests, so
   ``self._booted`` (instance_id -> (machine_uid, vram_bytes)) is the
   authoritative hold ledger, mirrored to ``im:vram:used:{machine}`` on
   boot/stop (TTL refreshed by the reaper while booted). An
   already-booted target needs no new VRAM and admits on capacity alone;
   a *boot* is gated on ``machine.total_vram_bytes - held_on(machine)``
   (the target's own hold excluded).
4. If a boot does not fit, idle **different-alias** backends on the same
   machine are evicted (``backend.stop``) LRU-first — oldest
   ``last_request_at`` (null = oldest) — until room is made. Eviction runs
   inside the same ``im:sched:lock`` acquisition as the boot, and an
   in-process ``_evicting`` set stops concurrent acquires (different
   aliases) from targeting the same victim. Instances with active slots
   are never evicted; a refused stop (NAK) moves on to the next candidate.
5. A ``stopped`` candidate is booted with ``backend.start`` (the provider
   acks only after its lifecycle reaches running). Boot failure/timeout
   falls through to the next candidate; if nothing fits, the head stays
   queued and retries on the next wake.
6. Waiting is bounded by ``queue_timeout`` -> ``QueueTimeout``.
7. Admission records ``im:sched:active`` and returns an ``Admission``.
   ``release`` frees the capacity slot only — the booted backend keeps
   holding its VRAM until it is stopped (idle reaper or eviction).

The idle-timeout reaper (``_idle_reaper``, started by ``start_background``)
runs every ``IDLE_REAPER_INTERVAL_SECONDS``: connected instances whose
backend is loaded, with zero active in-process slots, and whose idle
clock — ``max(last_request_at, backend_loaded_at)``, falling back to the
row's ``created_at`` — is older than the definition's
``idle_timeout_seconds`` get ``backend.stop`` (``idle_timeout_seconds ==
0`` means never idle-stop). A boot re-arms the clock: a just-started
backend that has never served a request is not reaped against its
stale pre-boot timestamp. Each tick also refreshes the ``im:vram:used``
mirror TTL for this process's booted holds.

Phase 16 slice 4 adds two agent-scoped constraints on top of the per-machine
VRAM gate (ARCHITECTURE.md §6):

* **`max_running_backends` (per agent).** Before booting a backend of type
  ``T`` on agent ``A``, the scheduler counts ``A``'s loaded backends (an
  agent is single-type). If the count is at ``ProviderType.max_running_backends``
  (0 = unlimited; e.g. ``1`` for halogen-flash), it **hot-swaps**: evicts the
  LRU idle same-agent backend (subject to the same busy/zero-active-slot guard
  as VRAM eviction, never the target) before admitting the new one. If no
  evictable same-agent victim exists the boot falls through to the next
  candidate (stays queued).
* **Serialized init warm-up.** On agent (re)connect the WS accept path fires
  ``warm_up_agent`` as a detached background task: it boots the agent's
  assigned backends **one at a time** (per-agent ``_warming`` guard, each
  under ``im:sched:lock``), each gated by VRAM + ``max_running``, stopping at
  the first that does not fit (warm-up never evicts). The rest boot on demand.
"""

import asyncio
import contextlib
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

import redis.asyncio as aioredis
from sqlmodel import Session, col, select

from app.core.config import settings
from app.core.db import engine
from app.models import (
    Machine,
    ProviderAgent,
    ProviderDefinition,
    ProviderInstance,
    ProviderModel,
    ProviderType,
)
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

# Idle-reaper tick interval (seconds). Injectable via the constructor for
# tests so they don't sleep 15s per tick.
IDLE_REAPER_INTERVAL_SECONDS = 15.0


class SchedulerError(Exception):
    """Base class for scheduler admission failures."""


class NoProviderAvailable(SchedulerError):
    """No connected/enabled instance can ever serve this alias -> HTTP 503."""


class QueueTimeout(SchedulerError):
    """Waited in the FIFO longer than queue_timeout -> HTTP 504."""


class QueueCleared(SchedulerError):
    """An operator cleared the queue while this request was waiting -> HTTP 503.

    Only *waiters* are cleared: already-admitted requests, boots, and VRAM
    holds are never touched (ARCHITECTURE.md §6 queue-clear).
    """


@dataclass
class Admission:
    """A granted inference slot for one request."""

    instance_id: str
    # http://{machine.reachable}:{agent.base_port}  (the agent's single env
    # /v1 port; NO trailing /v1; caller appends). litellm routes to the
    # specific backend by the request's model (= alias).
    base_url: str
    machine_uid: str


@dataclass
class _Slot:
    """One admitted request's hold on an instance.

    ``vram_bytes`` is kept only so the in-process active map can still
    show what each request is running against; it no longer participates
    in machine VRAM accounting (holds are per booted instance, see
    ``InferenceScheduler._booted``).
    """

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
    # Request ids an operator cleared out of the FIFO (Phase 19 queue-clear).
    # Consulted by each blocked ``acquire`` after every wake so it raises
    # ``QueueCleared``; entries are discarded again in ``_drop_waiter`` so the
    # set never leaks (covers the cancel-mid-clear race where the waiter's
    # top-of-loop check never runs).
    cleared: set[str] = field(default_factory=set)
    # Bumped whenever queue/active state changes so waiters can detect
    # progress without lost wakeups across the attempt/release boundary.
    changed: int = 0


class InferenceScheduler:
    """Per-alias FIFO admission with a Redis observability mirror."""

    def __init__(
        self,
        redis_client: aioredis.Redis,
        *,
        queue_timeout: float = 300.0,
        idle_reaper_interval: float = IDLE_REAPER_INTERVAL_SECONDS,
    ) -> None:
        self._redis = redis_client
        self._queue_timeout = queue_timeout
        self._idle_reaper_interval = idle_reaper_interval
        self._states: dict[str, _AliasState] = {}
        # Authoritative per-booted-instance VRAM hold ledger:
        #   instance_id -> (machine_uid, vram_bytes)
        # A booted backend keeps its weights resident between requests, so
        # the hold survives ``release`` and is only dropped when the backend
        # actually stops (idle reaper or eviction) or this process forgets
        # it (restart -> Redis mirror + DB backend_status reconcile).
        self._booted: dict[str, tuple[str, int]] = {}
        # Instance ids this process has an in-flight eviction/stop command
        # for, so concurrent acquires on different aliases can never both
        # target the same victim.
        self._evicting: set[str] = set()
        # Per-instance boot locks (Phase 25). Two served names of one
        # multi-model definition share ONE ProviderInstance but each holds its
        # OWN per-name admission lock, so the per-name lock does not serialize
        # their boots. This dict maps instance_id -> asyncio.Lock guarding the
        # boot decision for that instance so exactly one backend.start fires
        # per cold instance under concurrency (ARCHITECTURE.md §6 boot
        # serialization). The per-name FIFO queues are untouched.
        self._boot_locks: dict[str, asyncio.Lock] = {}
        # Agent ids with an in-flight proactive warm-up pass (see
        # ``warm_up_agent``). Serializes warm-up boots per agent so a
        # reconnect during an in-progress warm-up never double-boots.
        self._warming: set[str] = set()
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
                    if request_id in state.cleared:
                        # Operator cleared this waiter (Phase 19). The
                        # ``except BaseException`` unwind below runs the
                        # shielded ``_drop_waiter`` (mirror cleanup + discard
                        # from ``cleared``), same as the timeout/cancel paths.
                        raise QueueCleared(
                            f"the '{alias}' queue was cleared by an operator"
                        )
                    seen = state.changed

                async with state.lock:
                    if state.waiters and state.waiters[0] == request_id:
                        admission = await self._try_admit(alias, request_id)
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
        await asyncio.shield(
            self._release_wake_and_mirror(state, alias, request_id, slot)
        )

    async def _release_wake_and_mirror(
        self,
        state: _AliasState,
        alias: str,
        request_id: str,
        slot: _Slot,  # noqa: ARG002 - kept for signature clarity/logging
    ) -> None:
        """Shielded wake-up + Redis mirror cleanup for one released slot.

        Only the request's own bookkeeping is dropped: the machine VRAM
        hold belongs to the booted instance, not the request, so it stays
        until the backend is actually stopped (idle reaper / eviction).
        """
        async with state.condition:
            state.changed += 1
            state.condition.notify_all()
        with contextlib.suppress(Exception):
            await self._redis.srem(redis_keys.sched_active_key(alias), request_id)
            await self._redis.delete(redis_keys.sched_wait_key(request_id))

    async def start_background(self) -> None:
        """Start the idle-reaper task (stopped again by ``stop``)."""
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

    def has_candidates(self, alias: str) -> bool:
        """True when at least one connected+enabled instance can serve the
        alias. Lets the streaming route keep the cheap zero-candidates check
        as an HTTP 503 *before* the response starts (never open a stream
        that can never carry anything) while the slow boot/queue-wait phase
        happens inside the stream behind keepalives.
        """
        return bool(self._candidates(alias))

    # ------------------------------------------------------------------
    # Queue observability & operator clear (Phase 19)
    # ------------------------------------------------------------------
    async def queue_snapshot(self) -> list[dict[str, object]]:
        """Authoritative per-alias queue depth from in-process state.

        Returns ``{alias, queued, active}`` for every alias that currently
        has scheduler state (an alias that has never been acquired has no
        state and is omitted — the admin route joins with definitions).
        Reads each alias's waiter count and in-process active-slot count
        under its own condition lock; never touches the Redis mirror.
        """
        snapshot: list[dict[str, object]] = []
        for alias, state in list(self._states.items()):
            async with state.condition:
                snapshot.append(
                    {
                        "alias": alias,
                        "queued": len(state.waiters),
                        "active": len(state.active),
                    }
                )
        return snapshot

    async def clear_queue(self, alias: str | None = None) -> int:
        """Operator drain: cancel every *waiter* and return how many cleared.

        With ``alias`` set, clears that alias's queue; with ``None``, clears
        every alias's queue (clear-all). For each target, under the alias's
        admission lock: pop every waiter id, record it in ``state.cleared``
        so its blocked ``acquire`` raises ``QueueCleared`` on the next wake,
        then bump the state generation and ``notify_all`` on the condition so
        waiters wake promptly. Redis mirror entries (``im:sched:queue`` +
        each ``im:sched:wait``) are removed best-effort.

        Already-admitted requests, boots, and VRAM holds are NEVER touched —
        clearing only cancels waiters. Clearing an empty (or unknown) queue
        is a no-op returning 0; the operation is idempotent.
        """
        if alias is None:
            targets = list(self._states.items())
        else:
            state = self._states.get(alias)
            targets = [(alias, state)] if state is not None else []

        total = 0
        for target_alias, state in targets:
            # Under the admission lock so a concurrent head-admission (which
            # holds this same lock across its head-check + popleft) can never
            # race the pop. Lock order matches the acquire path (lock then
            # condition) — never the reverse — so no deadlock.
            async with state.lock:
                cleared_ids = list(state.waiters)
                if not cleared_ids:
                    continue
                for request_id in cleared_ids:
                    state.cleared.add(request_id)
                state.waiters.clear()
            async with state.condition:
                state.changed += 1
                state.condition.notify_all()
            total += len(cleared_ids)
            with contextlib.suppress(Exception):
                await self._redis.delete(redis_keys.sched_queue_key(target_alias))
                for request_id in cleared_ids:
                    await self._redis.delete(redis_keys.sched_wait_key(request_id))
        return total

    # ------------------------------------------------------------------
    # Admission attempt (head waiter only, under the per-alias lock)
    # ------------------------------------------------------------------
    def _boot_lock_for(self, instance_id: str) -> asyncio.Lock:
        """Get-or-create the per-instance boot lock (Phase 25).

        The get-or-create is synchronous (no ``await``), so it is atomic within
        a single event-loop step — two coroutines can never create two locks
        for the same instance.
        """
        lock = self._boot_locks.get(instance_id)
        if lock is None:
            lock = asyncio.Lock()
            self._boot_locks[instance_id] = lock
        return lock

    async def _try_admit(self, alias: str, request_id: str) -> Admission | None:
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
            # Admission invariant: never admit onto an instance with an
            # in-flight stop (eviction or idle reaper). The eviction path
            # holds the *requesting* alias's sched lock, not the victim's,
            # so a concurrent acquire for the victim's own alias would
            # otherwise slip a request onto a backend that is stopping.
            if instance_id in self._evicting:
                continue
            # Phase 25: capacity is a per-instance pool shared by every served
            # name on a multi-model definition, so count active slots across all
            # alias states (== the per-name count for a single-model def).
            if self._active_count_on_instance(instance_id) >= definition.capacity:
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
                # Phase 25: serialize the boot decision PER INSTANCE. Two
                # served names of one multi-model definition share this
                # instance but each holds its own per-name admission lock, so
                # without this a concurrent acquire(mm-a)/acquire(mm-b) on a
                # cold instance would BOTH dispatch backend.start. The first
                # name to take the per-instance lock boots; a sibling that
                # arrives meanwhile blocks here, then re-checks and finds the
                # instance already booted (no second start). The per-name FIFO
                # queues and their locks are untouched (ordering preserved).
                async with self._boot_lock_for(instance_id):
                    # TOCTOU re-check under the boot lock: a sibling served
                    # name may have completed the boot while we waited.
                    if instance_id in self._booted:
                        needs_boot = False
                    else:
                        # Single sched-lock acquisition covers both the
                        # eviction of VRAM-freeing victims and the boot itself,
                        # so two concurrent acquires (even for different
                        # aliases on the same machine) can never both evict the
                        # same victim or both boot into the same space.
                        lock_token = await self._take_admission_lock(alias)
                        try:
                            free = await self._free_vram(machine, instance_id)
                            if definition.vram_required_bytes > free:
                                free = await self._evict_for_space(
                                    machine,
                                    instance_id,
                                    alias,
                                    definition.id,
                                    definition.vram_required_bytes,
                                    free,
                                )
                            if definition.vram_required_bytes > free:
                                logger.debug(
                                    "alias %s: no VRAM free on machine %s after "
                                    "eviction attempts (%d needed, %d free); "
                                    "staying queued",
                                    alias,
                                    machine.uid,
                                    definition.vram_required_bytes,
                                    free,
                                )
                                continue
                            # Phase 16 slice 4: per-agent max_running_backends
                            # cap. An agent may run at most
                            # ProviderType.max_running_backends of its
                            # (single-type) backends at once (0 = unlimited). If
                            # the boot would exceed the cap, hot-swap evict an
                            # idle same-agent backend LRU-first; if none is
                            # evictable the boot falls through to the next
                            # candidate (stays queued).
                            if not await self._ensure_agent_capacity(
                                machine,
                                instance_id,
                                str(instance.agent_id),
                                alias,
                                definition,
                            ):
                                logger.debug(
                                    "alias %s: agent %s at max_running_backends "
                                    "with no idle same-agent victim; staying "
                                    "queued",
                                    alias,
                                    instance.agent_id,
                                )
                                continue
                            if not await self._boot(
                                str(instance.agent_id), instance_id
                            ):
                                continue
                        finally:
                            await self._release_admission_lock(alias, lock_token)
                        # Remember the successful boot so a lagging DB mirror
                        # never causes a redundant second backend.start and so
                        # VRAM stays accounted for between requests.
                        await self._mark_booted(instance_id, machine.uid, definition)

            if not needs_boot and instance_id not in self._booted:
                # Running per the DB but this process never booted it (e.g.
                # after an admin restart): adopt the hold so the ledger
                # reflects the resident weights.
                await self._mark_booted(instance_id, machine.uid, definition)

            await self._record_admission(alias, request_id, instance, definition)
            return Admission(
                instance_id=instance_id,
                base_url=f"http://{address}:{instance.agent.base_port}",
                machine_uid=machine.uid,
            )
        return None

    async def _boot(self, agent_id: str, instance_id: str) -> bool:
        """backend.start the instance over its agent's socket; True when the
        provider acks ok.

        The provider acks only after its lifecycle reaches running, so a
        successful return means /v1 is live. The ack window is the BOOT
        budget, not the 30s command default: a cold halogen-flash boot
        downloads its checkpoint and companions inside `backend.start`
        (tens of GB), and abandoning that ack would mean abandoning a boot
        the provider is still performing. A caller request still gives up
        at its own `queue_timeout` — the boot continues (the provider
        heartbeats `backend.status initializing` and the DB mirror follows),
        and a later request adopts the now-running instance.
        """
        try:
            reply = await manager.send_command(
                agent_id,
                "backend.start",
                {"instance_id": instance_id, "wait_for_running": True},
                timeout=settings.BACKEND_BOOT_TIMEOUT_SECONDS,
            )
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
                redis_keys.sched_wait_key(request_id),
                mapping={"status": "admitted", "instance_id": str(instance.id)},
            )
        self._touch_last_request(instance.id)

    # ------------------------------------------------------------------
    # VRAM hold ledger (per booted instance)
    # ------------------------------------------------------------------
    async def _mark_booted(
        self,
        instance_id: str,
        machine_uid: str,
        definition: ProviderDefinition,
    ) -> None:
        """Record (or adopt) a booted instance's VRAM hold + mirror it.

        Zero-byte holds are kept in-process (boot-skip bookkeeping) but not
        mirrored: the Redis ledger lists actual VRAM reservations only.
        """
        self._booted[instance_id] = (machine_uid, definition.vram_required_bytes)
        if definition.vram_required_bytes <= 0:
            return
        with contextlib.suppress(Exception):
            await self._redis.hset(
                redis_keys.vram_used_key(machine_uid),
                mapping={instance_id: str(definition.vram_required_bytes)},
            )
            await self._redis.expire(
                redis_keys.vram_used_key(machine_uid),
                redis_keys.VRAM_USED_TTL_SECONDS,
            )

    async def _mark_stopped(self, instance_id: str, machine_uid: str) -> None:
        """Drop a stopped instance's VRAM hold from ledger + mirror.

        Idempotent: a no-op when this process doesn't hold the instance.

        Phase 25: also reclaim the per-instance boot lock so ``_boot_locks``
        does not grow unboundedly across definition delete/recreate churn. The
        pop is done synchronously (before any ``await``) and ONLY when the lock
        is not currently held: dropping a lock another coroutine is mid-boot on
        would let a fresh ``_boot_lock_for`` mint a *different* lock and re-open
        the double-boot race. A still-locked entry is left in place and dropped
        on a later uncontended stop (the check-and-pop is atomic within one
        event-loop step, so no coroutine can grab the lock between them).
        """
        self._booted.pop(instance_id, None)
        lock = self._boot_locks.get(instance_id)
        if lock is not None and not lock.locked():
            del self._boot_locks[instance_id]
        with contextlib.suppress(Exception):
            await self._redis.hdel(redis_keys.vram_used_key(machine_uid), instance_id)

    async def note_backend_stopped(self, instance_id: str, machine_uid: str) -> None:
        """Prune the VRAM hold when an *external* actor reports the
        backend as stopped/error (admin UI stop, provider-side stop,
        crash-emitted ``backend.status``).

        Without this, only scheduler-initiated stops (``_stop_instance`` →
        ``_mark_stopped``) clear ``self._booted``: an out-of-band stop
        leaves a stale hold that the reaper's TTL refresh keeps alive
        forever and that permanently over-counts the machine. Called from
        ``connection_manager`` on non-loaded ``backend.status`` frames;
        safe (no-op) when the instance isn't held.
        """
        if instance_id in self._booted:
            await self._mark_stopped(instance_id, machine_uid)

    async def _stop_instance(
        self, agent_id: str, instance_id: str, machine_uid: str, reason: str
    ) -> bool:
        """backend.stop an instance (over its agent's socket) and clear its
        VRAM hold on success.

        Returns True when the provider acked ok. NAKs/exceptions are
        logged and surfaced as False — the caller decides what to do (try
        the next eviction candidate, retry the reaper next tick).
        """
        try:
            reply = await manager.send_command(
                agent_id, "backend.stop", {"instance_id": instance_id}
            )
        except Exception as exc:  # noqa: BLE001 - delivery failure = not stopped
            logger.warning(
                "backend.stop (%s) failed for instance %s: %s", reason, instance_id, exc
            )
            return False
        if not reply.payload.get("ok"):
            logger.warning(
                "backend.stop (%s) refused for instance %s: %s",
                reason,
                instance_id,
                reply.payload.get("error"),
            )
            return False
        await self._mark_stopped(instance_id, machine_uid)
        try:
            self._persist_stopped(instance_id)
        except Exception as exc:  # noqa: BLE001 - stop acked; DB mirror is best-effort
            logger.warning(
                "backend.stop (%s) acked for instance %s but the DB "
                "backend_status mirror write failed: %s",
                reason,
                instance_id,
                exc,
            )
        return True

    def _persist_stopped(self, instance_id: str) -> None:
        """Set the DB backend_status mirror to stopped after an acked stop.

        The provider emits ``backend.status`` per transition, so this is
        normally redundant — but doing it here keeps the per-booted-
        instance VRAM ledger (which merges the DB lifecycle state) honest
        even if that event is lost or lags, and makes stop-driven frees
        immediately visible to the next admission.
        """
        with Session(engine) as session:
            inst = session.get(ProviderInstance, uuid.UUID(instance_id))
            if inst is not None and inst.backend_status != BackendStatusValue.STOPPED:
                inst.backend_status = BackendStatusValue.STOPPED
                # Not loaded -> no load clock (the invariant that keeps
                # max(last_request_at, backend_loaded_at) honest).
                inst.backend_loaded_at = None
                inst.updated_at = datetime.now(UTC)
                session.add(inst)
                session.commit()

    async def _evict_for_space(
        self,
        machine: Machine,
        target_instance_id: str,
        requesting_alias: str,
        requesting_definition_id: uuid.UUID,
        required: int,
        free: int,
    ) -> int:
        """Stop idle different-definition backends on ``machine`` to free VRAM.

        Victims are LRU-ordered by ``last_request_at`` (null = oldest) and
        must have zero active in-process slots and a positive VRAM hold in
        the ledger (evicting something that holds nothing could never help
        and would only cascade). Each accepted stop frees the victim's
        held VRAM; returns the updated free bytes. A NAK or delivery
        failure skips that victim and moves to the next.

        Phase 25: the exclusion is by the requesting **definition id**, not
        the requesting alias string — every served name on a multi-model
        definition shares one instance and must never be an eviction victim
        of a sibling name on the same definition.

        The caller holds ``im:sched:lock:{alias}`` plus the in-process
        per-alias ``state.lock`` for the whole call. Those serialize
        same-alias acquires; ``self._evicting`` additionally guards the
        victims against concurrent acquires on *other* aliases sharing
        the machine (whose per-alias lock would not serialize here).
        """
        held = await self._machine_vram_held(machine)
        victims = self._eviction_candidates(
            machine.uid,
            requesting_definition_id,
            target_instance_id,
            held,
        )
        for victim in victims:
            if required <= free:
                break
            victim_id = victim["id"]
            self._evicting.add(victim_id)
            try:
                stopped = await self._stop_instance(
                    victim["agent_id"],
                    victim_id,
                    machine.uid,
                    f"eviction for alias '{requesting_alias}'",
                )
            finally:
                self._evicting.discard(victim_id)
            if not stopped:
                continue
            # Re-read the ledger instead of accumulating locally so a
            # concurrent eviction on another alias (which also freed VRAM
            # in the mirror) is visible and we never over-evict.
            free = await self._free_vram(machine, target_instance_id)
            logger.warning(
                "scheduler eviction: alias '%s' evicted idle instance %s "
                "(alias '%s') holding %d VRAM bytes on machine %s "
                "[target %s needs %d, free now %d]",
                requesting_alias,
                victim_id,
                victim["alias"],
                victim["vram_bytes"],
                machine.uid,
                target_instance_id,
                required,
                free,
            )
        return free

    def _eviction_candidates(
        self,
        machine_uid: str,
        requesting_definition_id: uuid.UUID,
        target_instance_id: str,
        held: dict[str, int],
        *,
        agent_id: str | None = None,
        require_hold: bool = True,
    ) -> list[dict]:
        """Idle, loaded instances on ``machine_uid`` eligible for eviction.

        Returns dicts with ``id``/``agent_id``/``alias``/``vram_bytes``/
        ``last_request_at``, LRU-ordered (oldest request first; ``None``
        sorts first). Instances with any active in-process slot (any
        alias), already being evicted, or of the requesting **definition**
        itself are excluded — we never evict something busy or the only home
        of the definition that is asking for the space. Phase 25: the
        exclusion is by definition id (not alias string) so sibling served
        names on one multi-model definition (one shared instance) are never
        victims of each other.

        Two eviction scopes share this builder:

        * **VRAM** (``agent_id=None``, ``require_hold=True``): any
          different-alias instance on the machine; a positive VRAM hold in
          the merged ledger is required (evicting something that holds
          nothing could never free space).
        * **max_running hot-swap** (``agent_id=<A>``, ``require_hold=False``):
          restricted to agent ``A`` and required to be *loaded* (DB
          running/in_use or in ``self._booted``) — a loaded backend counts
          against the per-agent cap even when it holds zero VRAM (e.g. a
          mock model), so the hold filter is replaced by a loaded filter.

        The DB ``backend_status`` filter is deliberately **absent** for the
        VRAM scope: any instance with a positive hold in the merged ledger
        is a victim, including ``self._booted`` instances whose DB mirror
        is stale (e.g. a status event lost after an out-of-band stop
        attempt). The hold requirement — not the DB state — is what makes
        VRAM eviction able to reclaim the space.
        """
        with Session(engine) as session:
            stmt = (
                select(ProviderInstance, ProviderDefinition)
                .join(
                    ProviderDefinition,
                    col(ProviderInstance.provider_definition_id)
                    == col(ProviderDefinition.id),
                )
                .join(
                    ProviderAgent,
                    col(ProviderInstance.agent_id) == col(ProviderAgent.id),
                )
                .join(Machine, col(ProviderAgent.machine_id) == col(Machine.id))
                .where(
                    Machine.uid == machine_uid,
                    ProviderInstance.provider_definition_id != requesting_definition_id,
                )
            )
            if agent_id is not None:
                stmt = stmt.where(ProviderAgent.id == uuid.UUID(agent_id))
            rows = session.exec(stmt).all()
            out: list[dict] = []
            for inst, definition in rows:
                instance_id = str(inst.id)
                if instance_id == target_instance_id:
                    continue
                if instance_id in self._evicting:
                    continue
                if self._active_count_on_instance(instance_id) > 0:
                    continue
                vram_bytes = held.get(instance_id, 0)
                loaded = (
                    inst.backend_status in RUNNING_BACKEND_STATUSES
                    or instance_id in self._booted
                )
                if require_hold:
                    if vram_bytes <= 0:
                        continue
                elif not loaded:
                    # max_running scope: only loaded same-agent backends
                    # count against the cap, so a stopped one is not a victim.
                    continue
                out.append(
                    {
                        "id": instance_id,
                        "agent_id": str(inst.agent_id),
                        "alias": definition.alias,
                        "vram_bytes": vram_bytes,
                        "last_request_at": inst.last_request_at,
                    }
                )
        # LRU: oldest last_request_at first; never-requested (None) is oldest.
        out.sort(
            key=lambda item: (
                item["last_request_at"] is not None,
                item["last_request_at"],
            )
        )
        return out

    # ------------------------------------------------------------------
    # Per-agent max_running_backends cap (Phase 16 slice 4)
    # ------------------------------------------------------------------
    def _max_running_for_type(self, provider_type: str) -> int:
        """The agent-scoped running cap for ``provider_type`` (0 = unlimited).

        Read from ``ProviderType.max_running_backends`` (declared in the
        type's shipped ``schema.json`` as ``x-max-running-backends`` and
        stored at registration). A type with no registry row (e.g. a bare
        test fixture) is treated as unlimited so the cap never fires
        accidentally.
        """
        with Session(engine) as session:
            ptype = session.exec(
                select(ProviderType).where(ProviderType.name == provider_type)
            ).first()
        return ptype.max_running_backends if ptype is not None else 0

    def _running_on_agent(self, agent_id: str, exclude_instance_id: str) -> int:
        """Count of loaded backends on ``agent_id`` (excluding the target).

        "Loaded" merges the DB lifecycle (``running``/``in_use``) with this
        process's ``self._booted`` ledger, so a backend this process just
        booted (whose DB status event may lag) still counts against the cap.

        NOTE: this counts ALL of the agent's backends without filtering
        ``provider_type``. That is correct ONLY because of the Phase 16
        invariant that a ``ProviderAgent`` is bound to exactly one
        ``provider_type`` (ARCHITECTURE.md §1 key invariant 1 / §4: every
        ``ProviderInstance`` on an agent shares the agent's type, and
        placement never mixes types onto one agent). Under that invariant the
        per-agent count IS the per-type count the cap is defined over; if an
        agent ever hosted multiple types this and ``_ensure_agent_capacity``
        would need a ``provider_type`` filter.
        """
        with Session(engine) as session:
            rows = session.exec(
                select(ProviderInstance.id, ProviderInstance.backend_status).where(
                    ProviderInstance.agent_id == uuid.UUID(agent_id)
                )
            ).all()
        count = 0
        for instance_id, backend_status in rows:
            sid = str(instance_id)
            if sid == exclude_instance_id:
                continue
            if backend_status in RUNNING_BACKEND_STATUSES or sid in self._booted:
                count += 1
        return count

    def _is_loaded(self, instance_id: str) -> bool:
        """True when ``instance_id`` is already resident (DB running/in_use or
        in this process's ``self._booted`` ledger)."""
        if instance_id in self._booted:
            return True
        with Session(engine) as session:
            status = session.exec(
                select(ProviderInstance.backend_status).where(
                    ProviderInstance.id == uuid.UUID(instance_id)
                )
            ).first()
        return status in RUNNING_BACKEND_STATUSES

    def _agent_connected(self, agent_id: str) -> bool:
        """True while the agent's mirrored liveness flag is set.

        ``websocket_connected`` is flipped False synchronously on socket
        teardown (``ws._mark_disconnected``) and by the presence sweep, so it
        is the testable, cross-process equivalent of the in-process
        ``manager.get(agent_id)`` live-socket check.
        """
        with Session(engine) as session:
            connected = session.exec(
                select(ProviderAgent.websocket_connected).where(
                    ProviderAgent.id == uuid.UUID(agent_id)
                )
            ).first()
        return bool(connected)

    async def _ensure_agent_capacity(
        self,
        machine: Machine,
        target_instance_id: str,
        agent_id: str,
        requesting_alias: str,
        definition: ProviderDefinition,
    ) -> bool:
        """Make room on ``agent_id`` for one more running backend of the
        target's type, hot-swapping an idle same-agent backend LRU-first if
        the agent is at ``max_running_backends``.

        Returns True when the boot may proceed (unlimited cap, already under
        the cap, or a victim was successfully stopped to get under it).
        Returns False when the cap is reached and no evictable same-agent
        victim exists (busy or none) — the caller then falls through to the
        next candidate (stays queued), never evicting a busy backend or the
        target itself.

        Reuses the shared ``_eviction_candidates`` builder (agent-scoped,
        loaded-not-hold filter) and ``_stop_instance`` + the ``_evicting``
        guard; the caller holds ``im:sched:lock:{alias}`` for the whole call.

        The agent-scoped eviction relies on the same single-type-per-agent
        invariant as ``_running_on_agent`` (see its note): every same-agent
        victim is of the target's type, so no ``provider_type`` filter is
        applied to the victim set.
        """
        cap = self._max_running_for_type(definition.provider_type)
        if cap <= 0:
            return True  # unlimited
        if self._running_on_agent(agent_id, target_instance_id) < cap:
            return True
        held = await self._machine_vram_held(machine)
        victims = self._eviction_candidates(
            machine.uid,
            definition.id,
            target_instance_id,
            held,
            agent_id=agent_id,
            require_hold=False,
        )
        for victim in victims:
            if self._running_on_agent(agent_id, target_instance_id) < cap:
                break
            victim_id = victim["id"]
            self._evicting.add(victim_id)
            try:
                stopped = await self._stop_instance(
                    victim["agent_id"],
                    victim_id,
                    machine.uid,
                    f"max_running hot-swap for alias '{requesting_alias}'",
                )
            finally:
                self._evicting.discard(victim_id)
            if not stopped:
                continue
            logger.warning(
                "scheduler hot-swap: alias '%s' evicted idle instance %s "
                "(alias '%s') on agent %s to respect max_running_backends=%d "
                "[target %s]",
                requesting_alias,
                victim_id,
                victim["alias"],
                agent_id,
                cap,
                target_instance_id,
            )
        return self._running_on_agent(agent_id, target_instance_id) < cap

    # ------------------------------------------------------------------
    # Proactive init warm-up (Phase 16 slice 4)
    # ------------------------------------------------------------------
    async def warm_up_agent(self, agent_id: str) -> None:
        """Boot an agent's assigned backends one at a time after it connects.

        Triggered as a detached background task from the WS accept path (and
        left as the seam for slice 5's ``agent.assignments.update`` re-warm)
        so it never blocks the socket handshake. Backends are warmed in a
        deterministic order (most-recently-used first, never-requested last,
        alias as tiebreak) and each is gated by machine VRAM admission and
        the type's ``max_running_backends`` — but warm-up NEVER evicts: it
        stops at the first backend that does not fit, leaving the rest
        ``stopped`` to boot on demand exactly as before.

        Serialization: an in-process per-agent ``_warming`` guard makes the
        pass one-at-a-time (a reconnect during an in-progress warm-up is a
        no-op), and each individual boot honors the ``im:sched:lock:{alias}``
        contract for the VRAM/cap decision + ``backend.start`` dispatch — the
        same discipline the request path uses — so a concurrent request on
        another alias sharing the machine can never double-book the space.
        Warmed backends idle out normally via the reaper.
        """
        if agent_id in self._warming:
            logger.debug("warm-up already in progress for agent %s; skipping", agent_id)
            return
        self._warming.add(agent_id)
        try:
            for target in self._warm_up_targets(agent_id):
                # Nit: if the agent's socket dropped mid-pass, stop cleanly
                # instead of letting every remaining target fail _boot with a
                # ConnectionError + warning (the socket is gone; nothing to
                # warm). Uses the mirrored websocket_connected flag (set False
                # synchronously on teardown) so it also holds in tests.
                if not self._agent_connected(agent_id):
                    logger.debug(
                        "warm-up agent %s: disconnected; stopping warm-up", agent_id
                    )
                    break
                instance_id = target["id"]
                definition = target["definition"]
                machine = target["machine"]
                if instance_id in self._evicting:
                    continue
                if self._active_count_on_instance(instance_id) > 0:
                    continue
                if definition.alias is None:  # pragma: no cover - defensive
                    continue
                # Serialize against request admissions for this alias the
                # same way ``acquire``/``_try_admit`` do: the per-alias
                # in-process ``state.lock`` is the real single-worker guard
                # (the Redis sched lock is best-effort). Without it a warm-up
                # boot and a concurrent request for the same alias could both
                # issue backend.start before either marks the instance booted.
                state = self._state_for(definition.alias)
                booted = False
                async with state.lock:
                    # TOCTOU re-check (mirrors _try_admit's needs_boot guard):
                    # the target list was built before this lock was taken, so
                    # a concurrent request may have booted this very backend in
                    # the meantime. Skip it rather than issue a redundant
                    # backend.start (idempotent on the provider, but the ledger
                    # and logs should not churn).
                    if self._is_loaded(instance_id):
                        continue
                    # Phase 25: serialize the boot decision PER INSTANCE, exactly
                    # as the request path does. A warm-up boot (keyed by
                    # definition.alias) and a concurrent request on a SIBLING
                    # served name hold DIFFERENT per-name state.locks, so the
                    # per-name lock does not serialize them; without this shared
                    # per-instance lock both could dispatch backend.start on a
                    # cold multi-model instance. Lock order stays acyclic
                    # (state.lock -> _boot_lock), identical to _try_admit.
                    async with self._boot_lock_for(instance_id):
                        # TOCTOU re-check under the boot lock: a sibling served
                        # name may have completed the boot while we waited.
                        if self._is_loaded(instance_id):
                            continue
                        lock_token = await self._take_admission_lock(definition.alias)
                        try:
                            free = await self._free_vram(machine, instance_id)
                            if definition.vram_required_bytes > free:
                                logger.debug(
                                    "warm-up agent %s: no VRAM for instance %s "
                                    "(%d needed, %d free); stopping warm-up",
                                    agent_id,
                                    instance_id,
                                    definition.vram_required_bytes,
                                    free,
                                )
                                break
                            cap = self._max_running_for_type(definition.provider_type)
                            if (
                                cap > 0
                                and self._running_on_agent(agent_id, instance_id) >= cap
                            ):
                                logger.debug(
                                    "warm-up agent %s: at max_running_backends=%d "
                                    "before instance %s; stopping warm-up",
                                    agent_id,
                                    cap,
                                    instance_id,
                                )
                                break
                            if await self._boot(agent_id, instance_id):
                                await self._mark_booted(
                                    instance_id, machine.uid, definition
                                )
                                booted = True
                            # else: a single backend failing to boot must not
                            # strand the rest of the pass; fall through to next.
                        finally:
                            await self._release_admission_lock(
                                definition.alias, lock_token
                            )
                if not booted:
                    continue
                logger.info(
                    "warm-up: booted instance %s (alias '%s') on agent %s",
                    instance_id,
                    definition.alias,
                    agent_id,
                )
        except Exception:  # noqa: BLE001 - warm-up is best-effort; never fatal
            logger.exception("warm-up failed for agent %s", agent_id)
        finally:
            self._warming.discard(agent_id)

    def _warm_up_targets(self, agent_id: str) -> list[dict]:
        """The agent's schedulable backends in warm-up order.

        Enabled definitions on a still-connected agent, ordered
        most-recently-used first (``last_request_at`` desc; never-requested
        last), with the definition alias as a deterministic tiebreak. Only
        ``stopped`` backends are returned — an already-loaded backend is warm
        by definition and must not be re-booted.
        """
        with Session(engine) as session:
            rows = session.exec(
                select(ProviderInstance, ProviderDefinition, Machine)
                .join(
                    ProviderDefinition,
                    col(ProviderInstance.provider_definition_id)
                    == col(ProviderDefinition.id),
                )
                .join(
                    ProviderAgent,
                    col(ProviderInstance.agent_id) == col(ProviderAgent.id),
                )
                .join(Machine, col(ProviderAgent.machine_id) == col(Machine.id))
                .where(
                    ProviderAgent.id == uuid.UUID(agent_id),
                    col(ProviderAgent.websocket_connected) == True,  # noqa: E712
                    col(ProviderDefinition.enabled) == True,  # noqa: E712
                )
            ).all()
            out: list[dict] = []
            expunged: set[int] = set()
            for inst, definition, machine in rows:
                if inst.backend_status in RUNNING_BACKEND_STATUSES:
                    continue  # already warm
                if str(inst.id) in self._booted:
                    continue
                # The Machine object is shared across every row of the same
                # agent (identity map); expunge each tracked instance only
                # once so a second expunge of the same object is a no-op.
                for obj in (inst, definition, machine):
                    if id(obj) not in expunged:
                        session.expunge(obj)
                        expunged.add(id(obj))
                out.append(
                    {
                        "id": str(inst.id),
                        "definition": definition,
                        "machine": machine,
                        "last_request_at": inst.last_request_at,
                    }
                )

        def _key(item: dict) -> tuple[int, float, str]:
            lra = item["last_request_at"]
            if lra is None:
                return (1, 0.0, item["definition"].alias)
            if lra.tzinfo is None:
                lra = lra.replace(tzinfo=UTC)
            # MRU first: negate the epoch so the most recent sorts smallest.
            return (0, -lra.timestamp(), item["definition"].alias)

        out.sort(key=_key)
        return out

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
            # A cleared id that reaches this path (normal clear unwind, or a
            # cancel that raced a clear before its top-of-loop check ran) must
            # not linger in ``cleared`` — this is the single cleanup point.
            state.cleared.discard(request_id)
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
    def _definition_for_name(
        self, session: Session, name: str
    ) -> ProviderDefinition | None:
        """Resolve a served name (or single-model alias) to its owning
        definition (Phase 25).

        ``ProviderModel.name`` first (a multi-model served name maps to its
        definition's single backend process), then ``ProviderDefinition.alias``
        (single-model). The caller owns the session.
        """
        row = session.exec(
            select(ProviderModel).where(ProviderModel.name == name)
        ).first()
        if row is not None:
            return session.get(ProviderDefinition, row.definition_id)
        return session.exec(
            select(ProviderDefinition).where(ProviderDefinition.alias == name)
        ).first()

    def _candidates(self, name: str) -> list[ProviderInstance]:
        with Session(engine) as session:
            # Phase 25: ``name`` may be a multi-model served name; resolve it to
            # the owning definition and return that definition's instances (one
            # backend process serves every served name).
            definition = self._definition_for_name(session, name)
            if definition is None or not definition.enabled:
                return []
            # Phase 16: an instance is schedulable when its owning agent's
            # socket is connected and its definition is enabled.
            rows = session.exec(
                select(ProviderInstance)
                .join(
                    ProviderAgent,
                    col(ProviderInstance.agent_id) == col(ProviderAgent.id),
                )
                .where(
                    ProviderInstance.provider_definition_id == definition.id,
                    col(ProviderAgent.websocket_connected) == True,  # noqa: E712
                )
            ).all()
            for row in rows:
                session.expunge(row)
            return list(rows)

    def _definition(self, name: str) -> ProviderDefinition | None:
        with Session(engine) as session:
            definition = self._definition_for_name(session, name)
            if definition is not None:
                session.expunge(definition)
            return definition

    async def _free_vram(self, machine: Machine, exclude_instance_id: str) -> int:
        """Bytes free on ``machine`` for a new boot.

        Total budget minus every hold on the machine **except** the target
        instance's own hold (the target either holds nothing yet or is
        about to be booted into its own space).
        """
        held = await self._machine_vram_held(machine)
        return machine.total_vram_bytes - sum(
            v for k, v in held.items() if k != exclude_instance_id
        )

    async def _machine_vram_held(self, machine: Machine) -> dict[str, int]:
        """instance_id -> bytes held on this machine (per booted instance).

        The authoritative ledger is ``self._booted`` (every backend this
        process knows is loaded, whether or not a request is using it).
        It is merged — taking the max per instance — with:

        * the Redis ``im:vram:used`` mirror (holds recorded before an
          admin restart or by another actor),
        * every in-process active slot on the machine (so a slot still
          counts even if its hold entry somehow vanished), and
        * connected instances whose DB lifecycle says loaded, at their
          definition's ``vram_required_bytes`` (a running backend holds
          its weights even after this process lost all trace of it).
        """
        held: dict[str, int] = {}
        with contextlib.suppress(Exception):
            raw = await self._redis.hgetall(redis_keys.vram_used_key(machine.uid))
            held = {k: int(v) for k, v in raw.items()}
        for instance_id, (machine_uid, vram_bytes) in self._booted.items():
            if machine_uid == machine.uid:
                held[instance_id] = max(held.get(instance_id, 0), vram_bytes)
        for alias_state in self._states.values():
            for slot in alias_state.active.values():
                if slot.machine_uid == machine.uid:
                    held[slot.instance_id] = max(
                        held.get(slot.instance_id, 0), slot.vram_bytes
                    )
        for instance_id, vram_bytes in self._loaded_instances_on(machine.uid):
            held[instance_id] = max(held.get(instance_id, 0), vram_bytes)
        return held

    def _loaded_instances_on(self, machine_uid: str) -> list[tuple[str, int]]:
        """(instance_id, vram_required_bytes) for connected instances on
        ``machine_uid`` whose backend lifecycle is loaded (running/in_use).

        VRAM is physically held by the booted backend, not by a request,
        so this is the DB-side source of truth for holds that outlive the
        in-process ledger and the TTL'd mirror (e.g. after an admin
        restart).
        """
        with Session(engine) as session:
            rows = session.exec(
                select(ProviderInstance, ProviderDefinition)
                .join(
                    ProviderDefinition,
                    col(ProviderInstance.provider_definition_id)
                    == col(ProviderDefinition.id),
                )
                .join(
                    ProviderAgent,
                    col(ProviderInstance.agent_id) == col(ProviderAgent.id),
                )
                .join(Machine, col(ProviderAgent.machine_id) == col(Machine.id))
                .where(
                    Machine.uid == machine_uid,
                    col(ProviderAgent.websocket_connected) == True,  # noqa: E712
                    col(ProviderInstance.backend_status).in_(RUNNING_BACKEND_STATUSES),
                )
            ).all()
            return [
                (str(inst.id), definition.vram_required_bytes)
                for inst, definition in rows
            ]

    def _active_count_on_instance(self, instance_id: str) -> int:
        """Active in-process slots on an instance, across all aliases."""
        return sum(
            1
            for alias_state in self._states.values()
            for slot in alias_state.active.values()
            if slot.instance_id == instance_id
        )

    # ------------------------------------------------------------------
    # Redis mirror helpers
    # ------------------------------------------------------------------
    def _state_for(self, alias: str) -> _AliasState:
        state = self._states.get(alias)
        if state is None:
            state = _AliasState()
            self._states[alias] = state
        return state

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
    # Background: idle-timeout reaper
    # ------------------------------------------------------------------
    async def _idle_reaper(self, stop_event: asyncio.Event) -> None:
        """Periodically stop backends idle past their definition's timeout.

        Each tick:
        * Refresh the ``im:vram:used`` TTL for every hold this process
          owns so the observability mirror survives a boot that outlives
          the 60s key TTL (the ledger itself is in-process and authoritative).
        * Stop every connected instance whose backend is loaded, which has
          **zero active in-process slots**, and whose idle clock —
          ``max(last_request_at, backend_loaded_at)``, falling back to the
          row's ``created_at`` — is at least ``idle_timeout_seconds`` old.
          ``idle_timeout_seconds == 0`` means "never idle-stop".

        The loop is resilient: an exception in a tick is logged and the
        next tick still runs. Only stopped-on-success instances lose their
        VRAM hold; a NAK (e.g. a race where the backend became busy) is
        simply retried on the following tick.
        """
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), self._idle_reaper_interval)
                continue  # stopping
            except TimeoutError:
                pass
            try:
                await self._reap_idle_once()
            except Exception:  # noqa: BLE001 - one bad tick must not kill the task
                logger.exception("idle reaper tick failed; continuing")

    async def _reap_idle_once(self) -> None:
        """One reaper pass. Returns the number of backends stopped."""
        await self._refresh_vram_mirror_ttl()
        now = datetime.now(UTC)
        stopped = 0
        for candidate in self._idle_stop_candidates(now):
            instance_id = candidate["id"]
            machine_uid = candidate["machine_uid"]
            if instance_id in self._evicting:
                continue
            if self._active_count_on_instance(instance_id) > 0:
                continue
            self._evicting.add(instance_id)
            try:
                stopped_ok = await self._stop_instance(
                    candidate["agent_id"], instance_id, machine_uid, "idle timeout"
                )
            finally:
                self._evicting.discard(instance_id)
            if not stopped_ok:
                continue
            stopped += 1
            logger.warning(
                "idle reaper: stopped instance %s (alias '%s') on machine %s "
                "after %ds idle (timeout %ds)",
                instance_id,
                candidate["alias"],
                machine_uid,
                candidate["idle_seconds"],
                candidate["idle_timeout_seconds"],
            )
        return stopped

    async def _refresh_vram_mirror_ttl(self) -> None:
        """Keep im:vram:used alive while this process holds booted backends."""
        if not self._booted:
            return
        machines = {machine_uid for machine_uid, _ in self._booted.values()}
        for machine_uid in machines:
            with contextlib.suppress(Exception):
                await self._redis.expire(
                    redis_keys.vram_used_key(machine_uid),
                    redis_keys.VRAM_USED_TTL_SECONDS,
                )

    def _idle_stop_candidates(self, now: datetime) -> list[dict]:
        """Loaded, connected, idle-past-timeout instances (zero slots checked
        by the caller against in-process state)."""
        with Session(engine) as session:
            rows = session.exec(
                select(ProviderInstance, ProviderDefinition, Machine)
                .join(
                    ProviderDefinition,
                    col(ProviderInstance.provider_definition_id)
                    == col(ProviderDefinition.id),
                )
                .join(
                    ProviderAgent,
                    col(ProviderInstance.agent_id) == col(ProviderAgent.id),
                )
                .join(Machine, col(ProviderAgent.machine_id) == col(Machine.id))
                .where(
                    col(ProviderAgent.websocket_connected) == True,  # noqa: E712
                    col(ProviderInstance.backend_status).in_(RUNNING_BACKEND_STATUSES),
                )
            ).all()
            out: list[dict] = []
            for inst, definition, machine in rows:
                timeout = definition.idle_timeout_seconds
                if timeout is None or timeout <= 0:
                    continue  # 0 = never idle-stop
                last = inst.last_request_at
                loaded_at = inst.backend_loaded_at
                # Normalize BEFORE comparing: a naive timestamp (legacy
                # row or driver) would raise TypeError inside the tick,
                # and the reaper swallows tick exceptions — the reap
                # would silently never happen again.
                if last is not None and last.tzinfo is None:
                    last = last.replace(tzinfo=UTC)
                if loaded_at is not None and loaded_at.tzinfo is None:
                    loaded_at = loaded_at.replace(tzinfo=UTC)
                # A just-loaded backend starts its idle window at boot:
                # never-served instances (last_request_at from a previous
                # boot, or None) must not be reaped against the old clock.
                if loaded_at is not None and (last is None or loaded_at > last):
                    last = loaded_at
                if last is None:
                    # Never served a request and load time unknown (e.g.
                    # rows predating the load clock): only reap if the row
                    # itself has been around at least that long.
                    last = inst.created_at
                if last.tzinfo is None:
                    last = last.replace(tzinfo=UTC)
                idle_seconds = int((now - last).total_seconds())
                if idle_seconds < timeout:
                    continue
                out.append(
                    {
                        "id": str(inst.id),
                        "agent_id": str(inst.agent_id),
                        "alias": definition.alias,
                        "machine_uid": machine.uid,
                        "idle_seconds": idle_seconds,
                        "idle_timeout_seconds": timeout,
                    }
                )
            return out

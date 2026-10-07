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
backend is loaded, with zero active in-process slots, and whose
``last_request_at`` is older than the definition's
``idle_timeout_seconds`` get ``backend.stop`` (``idle_timeout_seconds ==
0`` means never idle-stop). Each tick also refreshes the ``im:vram:used``
mirror TTL for this process's booted holds.
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

from app.core.db import engine
from app.models import (
    Machine,
    ProviderDefinition,
    ProviderInstance,
    backend_config_is_authored,
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


@dataclass
class Admission:
    """A granted inference slot for one request."""

    instance_id: str
    # http://{machine.reachable}:{port}  (NO trailing /v1; caller appends)
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
    # Admission attempt (head waiter only, under the per-alias lock)
    # ------------------------------------------------------------------
    async def _try_admit(
        self, alias: str, request_id: str, state: _AliasState
    ) -> Admission | None:
        definition = self._definition(alias)
        if definition is None or not definition.enabled:
            raise NoProviderAvailable(f"alias '{alias}' has no enabled definition")
        if not backend_config_is_authored(definition):
            # Phase 14: a shell definition can register an instance but
            # never serves — nothing has been configured to boot.
            raise NoProviderAvailable(
                f"alias '{alias}' is not configured yet (awaiting backend_config)"
            )
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
                # Single lock acquisition covers both the eviction of
                # VRAM-freeing victims and the boot itself, so two
                # concurrent acquires (even for different aliases on the
                # same machine) can never both evict the same victim or
                # both boot into the same space.
                lock_token = await self._take_admission_lock(alias)
                try:
                    free = await self._free_vram(machine, instance_id)
                    if definition.vram_required_bytes > free:
                        free = await self._evict_for_space(
                            machine,
                            instance_id,
                            alias,
                            definition.vram_required_bytes,
                            free,
                        )
                    if definition.vram_required_bytes > free:
                        logger.debug(
                            "alias %s: no VRAM free on machine %s after eviction "
                            "attempts (%d needed, %d free); staying queued",
                            alias,
                            machine.uid,
                            definition.vram_required_bytes,
                            free,
                        )
                        continue
                    if not await self._boot(instance_id):
                        continue
                finally:
                    await self._release_admission_lock(alias, lock_token)
                # Remember the successful boot so a lagging DB mirror
                # never causes a redundant second backend.start and so VRAM
                # stays accounted for between requests.
                await self._mark_booted(instance_id, machine.uid, definition)

            elif instance_id not in self._booted:
                # Running per the DB but this process never booted it (e.g.
                # after an admin restart): adopt the hold so the ledger
                # reflects the resident weights.
                await self._mark_booted(instance_id, machine.uid, definition)

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
        """
        self._booted.pop(instance_id, None)
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
        self, instance_id: str, machine_uid: str, reason: str
    ) -> bool:
        """backend.stop an instance and clear its VRAM hold on success.

        Returns True when the provider acked ok. NAKs/exceptions are
        logged and surfaced as False — the caller decides what to do (try
        the next eviction candidate, retry the reaper next tick).
        """
        try:
            reply = await manager.send_command(instance_id, "backend.stop", {})
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
                inst.updated_at = datetime.now(UTC)
                session.add(inst)
                session.commit()

    async def _evict_for_space(
        self,
        machine: Machine,
        target_instance_id: str,
        requesting_alias: str,
        required: int,
        free: int,
    ) -> int:
        """Stop idle different-alias backends on ``machine`` to free VRAM.

        Victims are LRU-ordered by ``last_request_at`` (null = oldest) and
        must have zero active in-process slots and a positive VRAM hold in
        the ledger (evicting something that holds nothing could never help
        and would only cascade). Each accepted stop frees the victim's
        held VRAM; returns the updated free bytes. A NAK or delivery
        failure skips that victim and moves to the next.

        The caller holds ``im:sched:lock:{alias}`` plus the in-process
        per-alias ``state.lock`` for the whole call. Those serialize
        same-alias acquires; ``self._evicting`` additionally guards the
        victims against concurrent acquires on *other* aliases sharing
        the machine (whose per-alias lock would not serialize here).
        """
        held = await self._machine_vram_held(machine)
        victims = self._eviction_candidates(
            machine.uid, requesting_alias, target_instance_id, held
        )
        for victim in victims:
            if required <= free:
                break
            victim_id = victim["id"]
            self._evicting.add(victim_id)
            try:
                stopped = await self._stop_instance(
                    victim_id, machine.uid, f"eviction for alias '{requesting_alias}'"
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
        requesting_alias: str,
        target_instance_id: str,
        held: dict[str, int],
    ) -> list[dict]:
        """Idle, loaded, different-alias instances on ``machine_uid``.

        Returns dicts with ``id``/``alias``/``vram_bytes``/
        ``last_request_at``, LRU-ordered (oldest request first; ``None``
        sorts first). Instances with any active in-process slot (any
        alias), with no VRAM hold in the ledger, already being evicted,
        or of the requesting alias itself are excluded — we never evict
        something busy, something that frees nothing, or the only home of
        the alias that is asking for the space.

        The DB ``backend_status`` filter is deliberately **absent**: any
        instance with a positive hold in the merged ledger is a victim,
        including ``self._booted`` instances whose DB mirror is stale
        (e.g. a status event lost after an out-of-band stop attempt).
        The hold requirement — not the DB state — is what makes eviction
        able to reclaim the space.
        """
        with Session(engine) as session:
            rows = session.exec(
                select(ProviderInstance, ProviderDefinition)
                .join(
                    ProviderDefinition,
                    col(ProviderInstance.provider_definition_id)
                    == col(ProviderDefinition.id),
                )
                .join(Machine, col(ProviderInstance.machine_id) == col(Machine.id))
                .where(
                    Machine.uid == machine_uid,
                    ProviderDefinition.alias != requesting_alias,
                )
            ).all()
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
                if vram_bytes <= 0:
                    continue
                out.append(
                    {
                        "id": instance_id,
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
            # Phase 14: shell definitions (backend_config NULL) are never
            # schedulable — the definition must have an authored config.
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
                    col(ProviderDefinition.backend_config).is_not(None),
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
                .join(Machine, col(ProviderInstance.machine_id) == col(Machine.id))
                .where(
                    Machine.uid == machine_uid,
                    col(ProviderInstance.websocket_connected) == True,  # noqa: E712
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
    # Background: idle-timeout reaper
    # ------------------------------------------------------------------
    async def _idle_reaper(self, stop_event: asyncio.Event) -> None:
        """Periodically stop backends idle past their definition's timeout.

        Each tick:
        * Refresh the ``im:vram:used`` TTL for every hold this process
          owns so the observability mirror survives a boot that outlives
          the 60s key TTL (the ledger itself is in-process and authoritative).
        * Stop every connected instance whose backend is loaded, which has
          **zero active in-process slots**, and whose ``last_request_at``
          is at least ``idle_timeout_seconds`` old. ``idle_timeout_seconds
          == 0`` means "never idle-stop".

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
                    instance_id, machine_uid, "idle timeout"
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
                .join(Machine, col(ProviderInstance.machine_id) == col(Machine.id))
                .where(
                    col(ProviderInstance.websocket_connected) == True,  # noqa: E712
                    col(ProviderInstance.backend_status).in_(RUNNING_BACKEND_STATUSES),
                    # Phase 14: a shell never boots, so never idle-stops.
                    col(ProviderDefinition.backend_config).is_not(None),
                )
            ).all()
            out: list[dict] = []
            for inst, definition, machine in rows:
                timeout = definition.idle_timeout_seconds
                if timeout is None or timeout <= 0:
                    continue  # 0 = never idle-stop
                last = inst.last_request_at
                if last is None:
                    # Never served a request: only reap if the row itself
                    # has been around at least that long.
                    last = inst.created_at
                if last.tzinfo is None:
                    last = last.replace(tzinfo=UTC)
                idle_seconds = int((now - last).total_seconds())
                if idle_seconds < timeout:
                    continue
                out.append(
                    {
                        "id": str(inst.id),
                        "alias": definition.alias,
                        "machine_uid": machine.uid,
                        "idle_seconds": idle_seconds,
                        "idle_timeout_seconds": timeout,
                    }
                )
            return out

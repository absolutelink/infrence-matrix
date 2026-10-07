"""Operator-initiated backend operations (admin -> provider commands).

`install_backend_ops` wires the lifecycle commands that let the admin UI
drive a backend by hand instead of waiting for the scheduler to boot it:

  ``backend.start``
      Boot the instance's one backend. ``wait_for_running`` (default
      ``true``) picks the contract. The scheduler needs an ack that means
      "/v1 is live" (docs/ws-protocol.md §4), so it keeps the blocking
      form. A manual click cannot: a cold halogen-flash boot first
      DOWNLOADS the checkpoint and its companions from HuggingFace — tens
      of GB, easily tens of minutes — so the UI sends
      ``wait_for_running: false``, the ack means "accepted", and the
      outcome arrives as ``backend.status`` / ``provider.status`` events.
  ``backend.stop``
      Stop it. Drain semantics live in ``BackendLifecycle``; a stop is
      quick, so this one always awaits.
  ``backend.restart``
      Drain-checked stop + start, honoring the same wait flag.
  ``provider.initialize``
      The full re-provision: re-register over HTTP (fresh instance secret +
      definition config), adopt the response, then drain-stop, boot
      (waiting through any engine download) and scrape ``list_models()``
      into a ``backend.metadata`` event. The slow part always runs in the
      background. Re-registering is also how a Phase 14 shell definition
      (``awaiting_config``) picks up an authored config when no
      ``provider.config.update`` push ever landed.

Concurrency: one operator-driven transition at a time per instance. A
second start/restart/initialize while one is in flight either JOINS it (a
waiting scheduler boot must not fail because an operator clicked Start
first) or NAKs ``boot_in_progress``, so a click is never silently
duplicated into a redundant spawn.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from provider_lib.admin_client import AdminClient
from provider_lib.backend import BackendLifecycle
from provider_lib.config_update import RETRY_AFTER_SECONDS
from provider_lib.wire import BackendStatusValue, Frame, InstanceStatusValue

logger = logging.getLogger("provider.ops")

# Driver attributes worth echoing in a start/restart ack. Providers name
# their ports differently (llama.cpp `backend_port`, halogen
# `api_port`/`engine_port`); getattr keeps one shared builder and simply
# omits whatever a driver does not expose.
_DETAIL_ATTRS = (
    "effective_capacity",
    "api_port",
    "engine_port",
    "backend_port",
)

# `re_register` returns the registration response's `provider_definition`
# (already adopted into the lifecycle/driver by the provider package), or
# None when it could not read one.
ReRegister = Callable[[], Awaitable["dict[str, Any] | None"]]


class BackendOps:
    """`backend.start` / `stop` / `restart` + `provider.initialize`."""

    def __init__(
        self,
        client: AdminClient,
        lifecycle: BackendLifecycle,
        *,
        backend_name: str,
        re_register: ReRegister | None = None,
    ) -> None:
        self._client = client
        self._lifecycle = lifecycle
        self._backend_name = backend_name
        self._re_register = re_register
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Introspection (tests, and the admin's status echo)
    # ------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    async def join(self) -> None:
        """Await the in-flight background transition (no-op when idle)."""
        task = self._task
        if task is not None and not task.done():
            await asyncio.shield(task)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    async def on_start(self, frame: Frame) -> dict[str, Any]:
        """`backend.start` — blocking for the scheduler, accept-style for
        an operator (``wait_for_running: false``)."""
        nak = await self._config_fence(frame)
        if nak is not None:
            return nak
        wait = bool((frame.payload or {}).get("wait_for_running", True))
        if self.busy:
            # Someone (operator or another boot) is already bringing the
            # backend up. A waiter joins that boot instead of failing or
            # spawning a second one; an accept-style caller just hears
            # "already accepted".
            if not wait:
                return self._nak("boot_in_progress", "start")
            await self.join()
            return self._settled_ack("start", waited=True)
        if wait:
            try:
                await self._lifecycle.start()
                await self._publish_metadata()
            except Exception as exc:  # noqa: BLE001 - reported as a NAK
                logger.exception("backend.start (wait) failed")
                return self._nak(f"start failed: {exc}", "start")
            return self._settled_ack("start", waited=True)
        self._spawn("backend.start", self._boot)
        return self._accepted_ack("start")

    async def on_stop(self, _frame: Frame) -> dict[str, Any]:
        """`backend.stop` — always awaited (a stop is quick)."""
        if self.busy:
            return self._nak("boot_in_progress", "stop")
        try:
            await self._lifecycle.stop()
        except Exception as exc:  # noqa: BLE001
            logger.exception("backend.stop failed")
            return self._nak(f"stop failed: {exc}", "stop")
        return {"ok": True, "detail": self._detail()}

    async def on_restart(self, frame: Frame) -> dict[str, Any]:
        """`backend.restart` — drain-checked stop + start."""
        nak = await self._config_fence(frame)
        if nak is not None:
            return nak
        if self.busy:
            return self._nak("boot_in_progress", "restart")
        busy_nak = self._in_use_nak("restart")
        if busy_nak is not None:
            return busy_nak
        wait = bool((frame.payload or {}).get("wait_for_running", True))
        if wait:
            try:
                await self._lifecycle.stop()
                await self._lifecycle.start()
                await self._publish_metadata()
            except Exception as exc:  # noqa: BLE001
                logger.exception("backend.restart (wait) failed")
                await self._emit_provider_status(InstanceStatusValue.ERROR, str(exc))
                return self._nak(f"restart failed: {exc}", "restart")
            return self._settled_ack("restart", waited=True)
        self._spawn("backend.restart", self._stop_then_boot)
        return self._accepted_ack("restart")

    async def on_initialize(self, frame: Frame) -> dict[str, Any]:  # noqa: ARG002
        """`provider.initialize` — re-register, adopt, reboot, re-scrape.

        Only the re-registration is synchronous (it is one HTTP round
        trip, and its failure must reach the caller as a NAK). The boot
        always runs in the background: it may download for an hour, and no
        admin HTTP request should sit open that long — progress and the
        terminal state arrive as events.
        """
        if self.busy:
            return self._nak("boot_in_progress", "initialize")
        if self._re_register is None:
            return self._nak("re_register_not_available", "initialize")
        try:
            await self._re_register()
        except Exception as exc:  # noqa: BLE001
            logger.exception("provider.initialize: re-registration failed")
            await self._emit_provider_status(
                InstanceStatusValue.ERROR, f"re-register failed: {exc}"
            )
            return self._nak(f"re-register failed: {exc}", "initialize")
        busy_nak = self._in_use_nak("initialize")
        if busy_nak is not None:
            return busy_nak
        # A definition with no authored config stays put: the provider must
        # never boot on defaults (the same Phase 14 fence as backend.start,
        # reported as a successful no-op because refreshing the
        # registration was still the right thing to do).
        if await self._no_applied_config():
            detail = self._detail()
            detail.update({"accepted": True, "no_config": True})
            return {"ok": True, "detail": detail}
        self._spawn("provider.initialize", self._stop_then_boot)
        if bool((frame.payload or {}).get("wait_for_running", False)):
            # The operator chose to wait for the whole re-provision
            # (download included) instead of polling.
            await self.join()
            return self._settled_ack("initialize", waited=True)
        ack = self._accepted_ack("initialize")
        ack["detail"]["steps"] = ["stop", "start", "metadata"]
        return ack

    # ------------------------------------------------------------------
    # Background transitions
    # ------------------------------------------------------------------
    def _spawn(self, name: str, step: Callable[[], Awaitable[None]]) -> None:
        self._task = asyncio.create_task(self._run(name, step), name=f"ops:{name}")

    async def _run(self, name: str, step: Callable[[], Awaitable[None]]) -> None:
        try:
            await step()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the ack already went out
            logger.exception("%s failed after being accepted", name)
            await self._emit_provider_status(
                InstanceStatusValue.ERROR, f"{name} failed: {exc}"
            )

    async def _boot(self) -> None:
        await self._lifecycle.start()
        await self._publish_metadata()
        await self._emit_provider_status(InstanceStatusValue.RUNNING)

    async def _stop_then_boot(self) -> None:
        await self._lifecycle.stop()
        await self._boot()

    # ------------------------------------------------------------------
    # Ack shapes
    # ------------------------------------------------------------------
    def _detail(self) -> dict[str, Any]:
        driver = self._lifecycle.driver
        detail: dict[str, Any] = {
            "backend": self._backend_name,
            "capacity": self._lifecycle.capacity,
            "backend_status": self._lifecycle.backend_status,
        }
        for attr in _DETAIL_ATTRS:
            value = getattr(driver, attr, None)
            if value is not None:
                detail[attr] = value
        return detail

    def _accepted_ack(self, step: str) -> dict[str, Any]:
        detail = self._detail()
        detail.update({"accepted": True, "step": step})
        return {"ok": True, "detail": detail}

    def _settled_ack(self, step: str, *, waited: bool) -> dict[str, Any]:
        """Ack for a boot that ran to completion (or was joined)."""
        status = self._lifecycle.backend_status
        if status not in (BackendStatusValue.RUNNING, BackendStatusValue.IN_USE):
            return self._nak(f"backend did not come up (status: {status})", step)
        detail = self._detail()
        detail["waited_for_running"] = waited
        return {"ok": True, "detail": detail}

    def _nak(self, error: str, step: str, **extra: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "error": error,
            "detail": {
                "step": step,
                "retry_after": RETRY_AFTER_SECONDS,
                "backend_status": self._lifecycle.backend_status,
                **extra,
            },
        }

    # ------------------------------------------------------------------
    # Fences (all read live state — never cached)
    # ------------------------------------------------------------------
    def _in_use_nak(self, step: str) -> dict[str, Any] | None:
        """Refuse a stop-triggering op while live requests hold slots.

        Same best-effort gate as `cache.clear` (reads the counters without
        the lifecycle lock): the stop itself is drain-checked for real by
        `provider.config.update`, which is the path that must not cut
        streams.
        """
        in_flight = self._lifecycle.in_flight
        if in_flight > 0 or self._lifecycle.backend_status == BackendStatusValue.IN_USE:
            return self._nak("backend_in_use", step, in_flight=in_flight)
        return None

    async def _no_applied_config(self) -> bool:
        """True when the Phase 14 fence would NAK a boot (no config applied)."""
        fence = getattr(self._client, "no_config_nak", None)
        if fence is None:
            return False
        probe = Frame(type="backend.start", id="ops-fence-probe")
        return (await fence(probe)) is not None

    async def _config_fence(self, frame: Frame) -> dict[str, Any] | None:
        """Phase 14: NAK a boot with no applied config (delegated)."""
        fence = getattr(self._client, "no_config_nak", None)
        if fence is None:
            return None
        return await fence(frame)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    async def _publish_metadata(self) -> None:
        """Scrape the running backend and emit `backend.metadata`.

        Best-effort: discovery must never fail a boot that already
        succeeded (mirrors `provider.config.update` step 9). The payload is
        the DB shape (`{"models": [...]}`) so the admin can store it
        verbatim.
        """
        list_models = getattr(self._lifecycle.driver, "list_models", None)
        if list_models is None:
            return
        try:
            models = await list_models()
        except Exception:  # noqa: BLE001
            logger.warning("backend.metadata scrape failed", exc_info=True)
            return
        if not models:
            return
        await self._client.send_event("backend.metadata", {"models": models})

    async def _emit_provider_status(
        self, instance_status: str, reason: str | None = None
    ) -> None:
        payload: dict[str, Any] = {
            "instance_status": instance_status,
            "backend_status": self._lifecycle.backend_status,
        }
        if reason:
            payload["error_message"] = reason
        await self._client.send_event("provider.status", payload)


def install_backend_ops(
    client: AdminClient,
    lifecycle: BackendLifecycle,
    *,
    backend_name: str,
    re_register: ReRegister | None = None,
) -> BackendOps:
    """Register the operator-driven lifecycle commands on `client`.

    `re_register` is provider-supplied — only the provider package knows
    its type, version, port, hardware report and schema. It must POST
    `/admin/api/providers/register` (which mints a fresh instance secret
    and rewrites `CACHE_DIR/provider_config.json`) and adopt the response
    into the lifecycle/driver (capacity + `backend_config` + applied
    fingerprint), returning the response's `provider_definition` or None.
    The live WebSocket is deliberately NOT recycled: the new secret is for
    future reconnects, while this socket — already authenticated, still at
    the current epoch — keeps carrying the status events the operator is
    watching.

    The Phase 14 shell fence is read off the client per command (installed
    by `provider_lib.config_update.install_config_handlers`), so install
    order does not matter.
    """
    ops = BackendOps(
        client, lifecycle, backend_name=backend_name, re_register=re_register
    )
    client.on_command("backend.start", ops.on_start)
    client.on_command("backend.stop", ops.on_stop)
    client.on_command("backend.restart", ops.on_restart)
    client.on_command("provider.initialize", ops.on_initialize)
    return ops


__all__ = ["BackendOps", "install_backend_ops"]

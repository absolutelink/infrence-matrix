"""Provider-side client for talking to the admin.

Responsibilities:
  - register() the provider instance with the admin (POST /admin/api/providers/register)
  - persist the returned provider config to CACHE_DIR/provider_config.json
  - maintain a bidirectional WebSocket to the admin (dial-out), with
    reconnect/backoff and a connect epoch supplied by the admin
  - expose send_command / event subscription primitives
"""

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import websockets

from provider_lib.config import ProviderSettings
from provider_lib.wire import Frame, now_iso

logger = logging.getLogger("provider.admin_client")


class RegistrationError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class SchemaPendingError(RegistrationError):
    """Registration refused by the Phase 12 schema consensus gate
    (409 `schema_pending` / `schema_conflict`, docs/ws-protocol.md §2).

    Subclasses RegistrationError so callers keep retrying with the
    normal backoff while the fleet converges on the new schema.
    """

    def __init__(self, status_code: int, message: str, detail: dict[str, Any]) -> None:
        super().__init__(status_code, message)
        self.error: str = detail.get("error", "")
        self.provider_type: str | None = detail.get("provider_type")
        self.committed_fingerprint: str | None = detail.get("committed_fingerprint")
        self.pending_fingerprint: str | None = detail.get("pending_fingerprint")
        self.voted: list[str] = list(detail.get("voted") or [])
        self.waiting_on: list[str] = list(detail.get("waiting_on") or [])


def _schema_refusal_message(detail: dict[str, Any]) -> str:
    """Readable operator log line for a Phase 12 schema-gate 409."""
    error = detail.get("error", "")
    voted = detail.get("voted") or []
    waiting_on = detail.get("waiting_on") or []
    if error == "schema_pending":
        kind = "waiting for all agents to update schema"
    elif error == "schema_conflict":
        kind = "schema conflict: another schema is pending consensus"
    else:
        return f"registration refused: {error or 'unknown schema error'}"
    return (
        f"{kind} ({len(voted)} voted, waiting on {len(waiting_on)}: "
        f"{waiting_on}; detail={detail})"
    )


class RegistrationResult:
    """Parsed ``POST /admin/api/providers/register`` response (Phase 16).

    The admin returns the agent identity plus the list of backends placed on
    this agent (one per assigned ``ProviderDefinition``). For the common
    single-backend case the first backend is exposed directly via
    :attr:`instance_id` / :attr:`provider_definition`.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.raw = data
        self.agent_id: str = data["agent_id"]
        self.agent_secret: str = data["agent_secret"]
        self.machine: dict[str, Any] = data.get("machine", {})
        self.backends: list[dict[str, Any]] = list(data.get("backends") or [])

    @property
    def instance_id(self) -> str | None:
        """The first backend's instance id (single-backend convenience)."""
        if self.backends:
            return self.backends[0].get("instance_id")
        return None

    @property
    def provider_definition(self) -> dict[str, Any]:
        """The first backend's definition dict (single-backend convenience)."""
        if self.backends:
            return self.backends[0].get("definition") or {}
        return {}

    @property
    def ws_url(self) -> str:
        base = self.machine.get("admin_ws_url")
        if not base:
            # Derive from ADMIN_BASE_URL by swapping scheme and path.
            base = (
                ProviderSettings()
                .ADMIN_BASE_URL.replace("https://", "wss://")
                .replace("http://", "ws://")
                .rstrip("/")
                + "/provider/ws"
            )
        # The admin identifies the connection by agent_id (query param) and
        # authenticates the Bearer secret against it.
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}agent_id={self.agent_id}"


class AdminClient:
    """Owns registration and the persistent WebSocket to the admin."""

    # Seconds between keepalive `ping` frames. Must stay below the admin's
    # presence-key TTL (60s) so an idle connection is not swept as
    # disconnected (docs/ws-protocol.md §3).
    PING_INTERVAL_SECONDS: float = 20.0

    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings
        self._registration: RegistrationResult | None = None
        self._ws: Any | None = None
        self._epoch: int = 0
        self._outgoing: asyncio.Queue[Frame] = asyncio.Queue()
        self._command_handlers: dict[
            str, Callable[[Frame], Awaitable[dict[str, Any]]]
        ] = {}
        self._pending: dict[str, asyncio.Future[Frame]] = {}
        self._recv_task: asyncio.Task[None] | None = None
        self._send_task: asyncio.Task[None] | None = None
        self._ping_task: asyncio.Task[None] | None = None
        self._connected = asyncio.Event()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    async def register(
        self,
        *,
        provider_type: str,
        version: str,
        base_port: int,
        hardware: dict[str, Any],
        schema: dict[str, Any] | None = None,
    ) -> RegistrationResult:
        body = {
            "machine_uid": self.settings.MACHINE_UID,
            "machine_secret": self.settings.MACHINE_SECRET,
            "agent_id": self.settings.AGENT_ID,
            "provider_type": provider_type,
            "version": version,
            "base_port": base_port,
            "hardware": hardware,
            "metrics_categories": sorted(self.settings.metrics_categories),
            "registered_at": now_iso(),
        }
        # Phase 12: the provider package's committed schema.json rides
        # along with the registration; the admin derives the fingerprint
        # and runs the consensus gate itself (docs/ws-protocol.md §2).
        # `None` keeps the TRANSITION (schema-omitted) path alive for
        # packages that haven't shipped a schema yet.
        if schema is not None:
            body["schema"] = schema
        url = self.settings.ADMIN_BASE_URL.rstrip("/") + "/admin/api/providers/register"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=body)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:  # noqa: BLE001
                detail = resp.text
            if resp.status_code == 409 and isinstance(detail, dict):
                # Structured Phase 12 schema-gate refusal (docs/ws-protocol.md
                # §2). Unknown/empty `error` kinds are still surfaced as
                # SchemaPendingError (retryable) with a generic message.
                raise SchemaPendingError(
                    resp.status_code, _schema_refusal_message(detail), detail
                )
            raise RegistrationError(resp.status_code, f"registration failed: {detail}")
        result = RegistrationResult(resp.json())
        self._registration = result
        self._persist_config(result)
        return result

    def _persist_config(self, result: RegistrationResult) -> None:
        self.settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent_id": result.agent_id,
            "machine": result.machine,
            "backends": result.backends,
            "saved_at": now_iso(),
        }
        self.settings.provider_config_path.write_text(json.dumps(payload, indent=2))

    def load_config(self) -> dict[str, Any] | None:
        if not self.settings.provider_config_path.exists():
            return None
        return json.loads(self.settings.provider_config_path.read_text())

    # ------------------------------------------------------------------
    # WebSocket lifecycle
    # ------------------------------------------------------------------
    @property
    def registration(self) -> RegistrationResult:
        if self._registration is None:
            raise RuntimeError("not registered; call register() first")
        return self._registration

    @property
    def is_registered(self) -> bool:
        return self._registration is not None

    @property
    def epoch(self) -> int:
        return self._epoch

    def on_command(
        self, name: str, handler: Callable[[Frame], Awaitable[dict[str, Any]]]
    ) -> None:
        self._command_handlers[name] = handler

    async def connect(self) -> None:
        """Dial the admin WS and start send/receive loops.

        Any loops left over from a previous connection are cancelled first so
        repeated connect() calls (e.g. from run_forever) never leave a
        zombie send task draining the outgoing queue onto a dead socket.
        """
        if self._recv_task is not None or self._send_task is not None:
            await self.disconnect()
        reg = self.registration
        url = reg.ws_url
        headers = {"Authorization": f"Bearer {reg.agent_secret}"}
        self._ws = await websockets.connect(url, additional_headers=headers)
        hello = json.loads(await self._ws.recv())
        self._epoch = int(hello.get("payload", {}).get("epoch", 0))
        self._connected.set()
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._send_task = asyncio.create_task(self._send_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())

    async def disconnect(self) -> None:
        self._connected.clear()
        for task in (self._recv_task, self._send_task, self._ping_task):
            if task:
                task.cancel()
        await asyncio.gather(
            self._recv_task,
            self._send_task,
            self._ping_task,
            return_exceptions=True,  # type: ignore[arg-type]
        )
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def send_event(self, type_: str, payload: dict[str, Any]) -> None:
        await self._outgoing.put(
            Frame(type=type_, id=str(uuid.uuid4()), epoch=self._epoch, payload=payload)
        )

    async def request(
        self, type_: str, payload: dict[str, Any], timeout: float = 30.0
    ) -> Frame:
        """Send a frame and await a reply frame (reply_to == sent id)."""
        if not self._connected.is_set():
            raise RuntimeError("websocket not connected")
        frame = Frame(
            type=type_, id=str(uuid.uuid4()), epoch=self._epoch, payload=payload
        )
        fut: asyncio.Future[Frame] = asyncio.get_running_loop().create_future()
        self._pending[frame.id] = fut
        await self._outgoing.put(frame)
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(frame.id, None)

    async def _send_loop(self) -> None:
        while self._ws is not None:
            frame = await self._outgoing.get()
            await self._ws.send(frame.to_json())

    async def _ping_loop(self) -> None:
        """Send a keepalive `ping` frame periodically so the admin's
        presence key never expires on an idle connection."""
        try:
            while True:
                await asyncio.sleep(self.PING_INTERVAL_SECONDS)
                await self._outgoing.put(
                    Frame(type="ping", id=str(uuid.uuid4()), epoch=self._epoch)
                )
        except asyncio.CancelledError:
            raise

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    frame = Frame.model_validate_json(raw)
                except Exception:  # noqa: BLE001
                    logger.warning("dropping malformed frame: %r", raw)
                    continue
                await self._dispatch(frame)
        except websockets.ConnectionClosed:
            logger.info("admin websocket closed")
        finally:
            self._connected.clear()

    async def _dispatch(self, frame: Frame) -> None:
        # A reply to an outstanding request?
        if frame.reply_to and frame.reply_to in self._pending:
            self._pending[frame.reply_to].set_result(frame)
            return
        # A command we should handle.
        handler = self._command_handlers.get(frame.type)
        if handler is None:
            logger.debug("no handler for command %s", frame.type)
            return
        try:
            ack_payload = await handler(frame)
        except Exception as exc:  # noqa: BLE001
            ack_payload = {"ok": False, "error": str(exc), "detail": {}}
        await self._outgoing.put(
            Frame(
                type="ack",
                id=str(uuid.uuid4()),
                reply_to=frame.id,
                epoch=self._epoch,
                payload=ack_payload,
            )
        )

    async def run_forever(
        self,
        on_connected: Callable[[], Awaitable[None]] | None = None,
        on_disconnected: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Reconnect loop with exponential backoff (1s -> 30s max).

        ``on_connected`` runs after every accepted connection (re-emit
        status so the admin's mirror is fresh); ``on_disconnected`` runs
        when an established connection drops (stop dependent tasks such
        as the metrics emitter). Neither runs when this task itself is
        cancelled — callers clean up in their own ``finally``.
        """
        backoff = 1.0
        while True:
            connected = False
            try:
                await self.connect()
                connected = True
                backoff = 1.0
                if on_connected:
                    await on_connected()
                if self._recv_task:
                    await self._recv_task
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("ws error: %s; reconnecting in %.1fs", exc, backoff)
            if connected and on_disconnected is not None:
                try:
                    await on_disconnected()
                except Exception:  # noqa: BLE001
                    logger.exception("on_disconnected callback failed")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

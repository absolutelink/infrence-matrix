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


class RegistrationResult:
    def __init__(self, data: dict[str, Any]) -> None:
        self.raw = data
        self.instance_id: str = data["instance_id"]
        self.instance_secret: str = data["instance_secret"]
        self.machine: dict[str, Any] = data.get("machine", {})
        self.provider_definition: dict[str, Any] = data.get("provider_definition", {})

    @property
    def ws_url(self) -> str:
        base = self.provider_definition.get("admin_ws_url")
        if base:
            return base
        # Derive from ADMIN_BASE_URL by swapping scheme and path.
        url = ProviderSettings().ADMIN_BASE_URL
        return (
            url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
            + "/provider/ws"
        )


class AdminClient:
    """Owns registration and the persistent WebSocket to the admin."""

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
        self._connected = asyncio.Event()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    async def register(
        self,
        *,
        provider_type: str,
        version: str,
        port: int,
        hardware: dict[str, Any],
    ) -> RegistrationResult:
        body = {
            "machine_uid": self.settings.MACHINE_UID,
            "registration_token": self.settings.PROVIDER_REGISTRATION_TOKEN,
            "provider_type": provider_type,
            "version": version,
            "port": port,
            "hardware": hardware,
            "metrics_categories": sorted(self.settings.metrics_categories),
            "registered_at": now_iso(),
        }
        url = self.settings.ADMIN_BASE_URL.rstrip("/") + "/admin/api/providers/register"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=body)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:  # noqa: BLE001
                detail = resp.text
            raise RegistrationError(resp.status_code, f"registration failed: {detail}")
        result = RegistrationResult(resp.json())
        self._registration = result
        self._persist_config(result)
        return result

    def _persist_config(self, result: RegistrationResult) -> None:
        self.settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "instance_id": result.instance_id,
            "machine": result.machine,
            "provider_definition": result.provider_definition,
            "config_fingerprint": result.provider_definition.get("config_fingerprint"),
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
    def epoch(self) -> int:
        return self._epoch

    def on_command(
        self, name: str, handler: Callable[[Frame], Awaitable[dict[str, Any]]]
    ) -> None:
        self._command_handlers[name] = handler

    async def connect(self) -> None:
        """Dial the admin WS and start send/receive loops."""
        reg = self.registration
        url = reg.ws_url
        headers = {"Authorization": f"Bearer {reg.instance_secret}"}
        self._ws = await websockets.connect(url, additional_headers=headers)
        hello = json.loads(await self._ws.recv())
        self._epoch = int(hello.get("payload", {}).get("epoch", 0))
        self._connected.set()
        self._recv_task = asyncio.create_task(self._recv_loop())
        self._send_task = asyncio.create_task(self._send_loop())

    async def disconnect(self) -> None:
        self._connected.clear()
        for task in (self._recv_task, self._send_task):
            if task:
                task.cancel()
        await asyncio.gather(
            self._recv_task,
            self._send_task,
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
        self, on_connected: Callable[[], Awaitable[None]] | None = None
    ) -> None:
        """Reconnect loop with exponential backoff."""
        backoff = 1.0
        while True:
            try:
                await self.connect()
                backoff = 1.0
                if on_connected:
                    await on_connected()
                if self._recv_task:
                    await self._recv_task
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("ws error: %s; reconnecting in %.1fs", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

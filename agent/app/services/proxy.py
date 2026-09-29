"""Proxies requests to llama.cpp servers."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.services.inference_operations import (
    activate_operation,
    bind_operation_task,
    finish_operation,
    get_operation,
    operation_exists,
    start_operation,
)

CONNECT_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0, 8.0, 8.0)
UPSTREAM_CONNECT_TIMEOUT_SECONDS = 10.0
MAX_PRE_BYTE_IDLE_RETRIES = 1


class ProxyOperationExists(Exception):
    """Raised when an operation ID has already been dispatched."""


@dataclass
class _SlotReservation:
    acquired: bool = False
    released: bool = False
    generation: int | None = None


class ServerProxy:
    """Proxies requests to llama.cpp servers.

    All mutable state lives on the injected ``server_manager``, so a single
    shared instance can serve every request. The underlying ``httpx`` client
    owns a bounded connection pool and must be closed at shutdown via
    :meth:`aclose`; constructing one per request leaks a socket (and its
    file descriptor) for every keep-alive upstream connection.
    """

    def __init__(self, server_manager) -> None:
        self.server_manager = server_manager
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=UPSTREAM_CONNECT_TIMEOUT_SECONDS),
            limits=httpx.Limits(
                max_connections=settings.PROXY_MAX_CONNECTIONS,
                max_keepalive_connections=settings.PROXY_MAX_KEEPALIVE_CONNECTIONS,
                keepalive_expiry=settings.PROXY_KEEPALIVE_EXPIRY_SECONDS,
            ),
        )

    async def aclose(self) -> None:
        """Close the shared upstream client and release pooled sockets."""
        await self.client.aclose()

    def _get_server_port(self, server_id: str) -> int:
        """Get server port from server manager."""
        config = self.server_manager.configs.get(server_id)
        if not config:
            raise ValueError(f"Server {server_id} not found")
        return config.port

    async def proxy_stream_background(
        self, *args, **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """Drain the LLM independently of downstream SSE backpressure.

        The producer owns the slot and closes it at upstream EOF, even when a
        client has stopped reading a frame already queued for delivery.
        """
        frames: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()

        async def drain() -> None:
            try:
                async for frame in self.proxy_stream(*args, **kwargs):
                    frames.put_nowait(frame)
            except BaseException as exc:
                frames.put_nowait(exc)
            finally:
                frames.put_nowait(None)

        producer = asyncio.create_task(drain())
        try:
            while True:
                item = await frames.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            if not producer.done():
                producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    async def proxy_request(
        self,
        server_id: str,
        method: str,
        path: str,
        headers: dict,
        json: dict | None = None,
        expected_slot_generation: int | None = None,
        enforce_capacity: bool = False,
        capacity_reserved: bool = False,
        operation_id: str | None = None,
    ) -> httpx.Response:
        """Proxy HTTP request to llama.cpp."""
        reservation = _SlotReservation(
            acquired=capacity_reserved, generation=expected_slot_generation
        )
        outcome = "failed"
        close_reason = "proxy_error"
        started_at = asyncio.get_running_loop().time()
        upstream_status: int | None = None
        response_bytes = 0
        upstream_opened = False
        idle_retries = 0
        inference = enforce_capacity or operation_id is not None
        try:
            self._check_generation(server_id, expected_slot_generation)
            url = self._get_server_url(server_id, path)
            if not reservation.acquired:
                await self._connection_started(
                    server_id,
                    enforce_capacity,
                    operation_id,
                    expected_slot_generation,
                    reservation,
                )
            connect_delays = iter(CONNECT_RETRY_DELAYS)
            while True:
                try:
                    self._check_generation(server_id, expected_slot_generation)
                    if inference:
                        # request() buffers the whole response, hiding whether a
                        # timed-out call had already returned body bytes. Read
                        # explicitly so only pre-byte stalls can be replayed.
                        async with self.client.stream(
                            method=method,
                            url=url,
                            headers=headers,
                            json=json,
                            timeout=self._inference_timeout(),
                        ) as upstream:
                            self._upstream_opened(server_id)
                            upstream_opened = True
                            upstream_status = upstream.status_code
                            content = bytearray()
                            async for chunk in upstream.aiter_bytes():
                                content.extend(chunk)
                                response_bytes += len(chunk)
                            response = httpx.Response(
                                upstream.status_code,
                                headers=upstream.headers,
                                content=bytes(content),
                                request=upstream.request,
                            )
                    else:
                        response = await self.client.request(
                            method=method, url=url, headers=headers, json=json
                        )
                    status_code = getattr(response, "status_code", None)
                    upstream_status = (
                        status_code if isinstance(status_code, int) else None
                    )
                    if not inference:
                        content = getattr(response, "content", b"")
                        response_bytes = (
                            len(content) if isinstance(content, bytes) else 0
                        )
                    if upstream_status == 400:
                        logger.error(
                            "inference_upstream_bad_request server_id=%s generation=%s method=%s path=%s request_body=%s response_body=%s",
                            server_id,
                            expected_slot_generation,
                            method,
                            path,
                            json,
                            response.text,
                        )
                    is_error = upstream_status is not None and upstream_status >= 400
                    outcome = "failed" if is_error else "completed"
                    close_reason = (
                        "llm_http_error" if is_error else "llm_response_complete"
                    )
                    return response
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    delay = next(connect_delays, None)
                    if upstream_opened or delay is None:
                        raise
                    await asyncio.sleep(delay)
                except httpx.ReadTimeout:
                    # Closing the first upstream context precedes this retry;
                    # keep the same operation ID and slot throughout.
                    if (
                        not inference
                        or response_bytes
                        or idle_retries >= MAX_PRE_BYTE_IDLE_RETRIES
                        or upstream_status is not None
                        and upstream_status >= 400
                    ):
                        raise
                    idle_retries += 1
                    logger.warning(
                        "inference_proxy_http_idle_retry operation_id=%s server_id=%s generation=%s attempt=%d idle_seconds=%.1f",
                        operation_id or "-",
                        server_id,
                        expected_slot_generation,
                        idle_retries,
                        settings.UPSTREAM_IDLE_TIMEOUT_SECONDS,
                    )
                    if upstream_opened:
                        self._upstream_closed(server_id)
                        upstream_opened = False
                    self._check_generation(server_id, expected_slot_generation)
                    continue
        except asyncio.CancelledError:
            outcome = "cancelled"
            close_reason = "request_cancelled_before_http_response"
            raise
        except Exception as exc:
            close_reason = (
                f"llm_http_error:{type(exc).__name__}"
                if upstream_status is not None
                else f"agent_connect_error:{type(exc).__name__}"
            )
            logger.warning(
                "inference_proxy_http_error operation_id=%s server_id=%s generation=%s source=%s upstream_status=%s error_type=%s error=%s",
                operation_id or "-",
                server_id,
                expected_slot_generation,
                "llm" if upstream_status is not None else "agent_or_connect",
                upstream_status,
                type(exc).__name__,
                str(exc),
            )
            raise
        finally:
            if upstream_opened:
                self._upstream_closed(server_id)
            if enforce_capacity or operation_id is not None:
                logger.info(
                    "inference_proxy_http_close operation_id=%s server_id=%s generation=%s outcome=%s close_reason=%s upstream_status=%s response_bytes=%d duration_ms=%.1f",
                    operation_id or "-",
                    server_id,
                    expected_slot_generation,
                    outcome,
                    close_reason,
                    upstream_status,
                    response_bytes,
                    (asyncio.get_running_loop().time() - started_at) * 1000,
                )
            await self._release_reservation(
                server_id, enforce_capacity, operation_id, outcome, reservation
            )

    async def proxy_stream(
        self,
        server_id: str,
        method: str,
        path: str,
        json: dict | None = None,
        expected_slot_generation: int | None = None,
        enforce_capacity: bool = False,
        capacity_reserved: bool = False,
        operation_id: str | None = None,
    ) -> AsyncGenerator[bytes, None]:
        """Proxy streaming request (SSE) to llama.cpp."""
        reservation = _SlotReservation(
            acquired=capacity_reserved, generation=expected_slot_generation
        )
        outcome = "failed"
        close_reason = "proxy_error"
        started_at = asyncio.get_running_loop().time()
        upstream_status: int | None = None
        upstream_opened = False
        connection_open = False
        chunks = 0
        bytes_sent = 0
        done_marker_seen = False
        done_marker_tail = b""
        reservation_task: asyncio.Task[None] | None = None
        idle_retries = 0
        try:
            self._check_generation(server_id, expected_slot_generation)
            url = self._get_server_url(server_id, path)
            if not reservation.acquired:
                reservation_task = asyncio.create_task(
                    self._connection_started(
                        server_id,
                        enforce_capacity,
                        operation_id,
                        expected_slot_generation,
                        reservation,
                    )
                )
                while not reservation_task.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(reservation_task), timeout=10.0
                        )
                    except TimeoutError:
                        yield b": inference slot queued\n\n"
                await reservation_task
                if operation_id is not None:
                    operation = get_operation(
                        self.server_manager, operation_id, server_id
                    )
                    if operation is None or operation["status"] != "active":
                        raise asyncio.CancelledError
                bind_operation_task(self.server_manager, operation_id)
            connect_delays = iter(CONNECT_RETRY_DELAYS)
            while True:
                connected = False
                try:
                    self._check_generation(server_id, expected_slot_generation)
                    async with self.client.stream(
                        method=method,
                        url=url,
                        json=json,
                        headers={"Accept": "text/event-stream"},
                        timeout=self._inference_timeout()
                        if enforce_capacity
                        else self.client.timeout,
                    ) as response:
                        connected = True
                        upstream_opened = True
                        self._upstream_opened(server_id)
                        connection_open = True
                        upstream_status = response.status_code
                        logger.info(
                            "inference_stream_open operation_id=%s server_id=%s generation=%s upstream_status=%s",
                            operation_id or "-",
                            server_id,
                            expected_slot_generation,
                            upstream_status,
                        )
                        if upstream_status == 400:
                            error_body = bytearray()
                            async for chunk in response.aiter_bytes():
                                chunks += 1
                                bytes_sent += len(chunk)
                                error_body.extend(chunk)
                            logger.error(
                                "inference_upstream_bad_request server_id=%s generation=%s method=%s path=%s request_body=%s response_body=%s",
                                server_id,
                                expected_slot_generation,
                                method,
                                path,
                                json,
                                bytes(error_body).decode(
                                    response.encoding or "utf-8",
                                    errors="replace",
                                ),
                            )
                            if error_body:
                                yield bytes(error_body)
                            outcome = "failed"
                            close_reason = "llm_http_error"
                            return
                        async for chunk in response.aiter_bytes():
                            chunks += 1
                            bytes_sent += len(chunk)
                            if not done_marker_seen:
                                scan = done_marker_tail + chunk
                                done_marker_seen = b"[DONE]" in scan
                                done_marker_tail = scan[-16:]
                            yield chunk
                    if connection_open:
                        self._upstream_closed(server_id)
                        connection_open = False
                    if upstream_status is not None and upstream_status >= 400:
                        outcome = "failed"
                        close_reason = "llm_http_error"
                    else:
                        outcome = "completed"
                        close_reason = (
                            "llm_eof_after_done_marker"
                            if done_marker_seen
                            else "llm_eof_without_done_marker"
                        )
                    return
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    delay = next(connect_delays, None)
                    if connected or delay is None:
                        raise
                    await asyncio.sleep(delay)
                except httpx.ReadTimeout:
                    # Never restart generation after anything was sent toward
                    # the backend, even if the upstream later goes idle.
                    if (
                        not enforce_capacity
                        or bytes_sent
                        or idle_retries >= MAX_PRE_BYTE_IDLE_RETRIES
                        or upstream_status is not None
                        and upstream_status >= 400
                    ):
                        raise
                    idle_retries += 1
                    logger.warning(
                        "inference_stream_idle_retry operation_id=%s server_id=%s generation=%s attempt=%d idle_seconds=%.1f",
                        operation_id or "-",
                        server_id,
                        expected_slot_generation,
                        idle_retries,
                        settings.UPSTREAM_IDLE_TIMEOUT_SECONDS,
                    )
                    if connection_open:
                        self._upstream_closed(server_id)
                        connection_open = False
                    self._check_generation(server_id, expected_slot_generation)
                    continue
        except asyncio.CancelledError:
            outcome = "cancelled"
            close_reason = (
                "downstream_cancelled_after_llm_connect"
                if upstream_opened
                else "downstream_cancelled_before_llm_connect"
            )
            if reservation_task is not None and not reservation_task.done():
                reservation_task.cancel()
                await asyncio.gather(reservation_task, return_exceptions=True)
            raise
        except GeneratorExit:
            outcome = "cancelled"
            close_reason = (
                "downstream_closed_after_llm_connect"
                if upstream_opened
                else "downstream_closed_before_llm_connect"
            )
            raise
        except Exception as exc:
            close_reason = (
                f"llm_stream_error:{type(exc).__name__}"
                if upstream_opened
                else f"agent_proxy_error:{type(exc).__name__}"
            )
            logger.warning(
                "inference_stream_error operation_id=%s server_id=%s generation=%s source=%s upstream_status=%s error_type=%s error=%s",
                operation_id or "-",
                server_id,
                expected_slot_generation,
                "llm" if upstream_opened else "agent_proxy_or_slot_wait",
                upstream_status,
                type(exc).__name__,
                str(exc),
            )
            raise
        finally:
            if connection_open:
                self._upstream_closed(server_id)
            if reservation_task is not None and not reservation_task.done():
                reservation_task.cancel()
                await asyncio.gather(reservation_task, return_exceptions=True)
            duration_ms = (asyncio.get_running_loop().time() - started_at) * 1000
            logger.info(
                "inference_stream_close operation_id=%s server_id=%s generation=%s outcome=%s close_reason=%s upstream_status=%s upstream_opened=%s done_marker=%s chunks=%d bytes=%d duration_ms=%.1f",
                operation_id or "-",
                server_id,
                expected_slot_generation,
                outcome,
                close_reason,
                upstream_status,
                upstream_opened,
                done_marker_seen,
                chunks,
                bytes_sent,
                duration_ms,
            )
            await self._release_reservation(
                server_id, enforce_capacity, operation_id, outcome, reservation
            )

    async def _connection_started(
        self,
        server_id: str,
        enforce_capacity: bool,
        operation_id: str | None = None,
        expected_generation: int | None = None,
        reservation: _SlotReservation | None = None,
    ) -> None:
        if not enforce_capacity:
            if reservation is not None:
                reservation.acquired = True
            return

        conditions = getattr(self.server_manager, "_inference_slot_conditions", None)
        if conditions is None:
            conditions = self.server_manager._inference_slot_conditions = {}
        condition = conditions.setdefault(server_id, asyncio.Condition())
        registered_operation = False
        try:
            async with condition:
                active = getattr(
                    self.server_manager, "_active_inference_requests", None
                )
                if active is None:
                    active = self.server_manager._active_inference_requests = {}
                if operation_id is not None:
                    if operation_exists(self.server_manager, operation_id):
                        raise ProxyOperationExists(
                            f"Inference operation {operation_id} already exists"
                        )
                    config = self.server_manager.configs.get(server_id)
                    if config is None:
                        raise ValueError(f"Server {server_id} not found")
                    if (
                        expected_generation is not None
                        and config.slot_generation != expected_generation
                    ):
                        raise httpx.HTTPStatusError(
                            "Inference slot generation changed while request waited",
                            request=httpx.Request("POST", "http://127.0.0.1/"),
                            response=httpx.Response(409),
                        )
                    start_operation(
                        self.server_manager,
                        operation_id,
                        server_id,
                        config.slot_generation,
                    )
                    registered_operation = True
                while True:
                    config = self.server_manager.configs.get(server_id)
                    if config is None:
                        raise ValueError(
                            f"Server {server_id} stopped while request waited"
                        )
                    if (
                        expected_generation is not None
                        and config.slot_generation != expected_generation
                    ):
                        raise httpx.HTTPStatusError(
                            "Inference slot generation changed while request waited",
                            request=httpx.Request("POST", "http://127.0.0.1/"),
                            response=httpx.Response(409),
                        )
                    capacity = max(
                        int(self.server_manager.get_effective_capacity(server_id)), 1
                    )
                    if active.get(server_id, 0) < capacity:
                        activate_operation(self.server_manager, operation_id)
                        active[server_id] = active.get(server_id, 0) + 1
                        if reservation is not None:
                            reservation.acquired = True
                            reservation.generation = config.slot_generation
                        return
                    await condition.wait()
        except asyncio.CancelledError:
            if registered_operation:
                finish_operation(self.server_manager, operation_id, "cancelled")
            raise
        except Exception:
            if registered_operation:
                finish_operation(self.server_manager, operation_id, "failed")
            raise

    async def reserve_inference_slot(
        self, server_id: str, operation_id: str | None = None
    ) -> None:
        """Atomically reserve capacity before a streaming response is returned."""
        await self._connection_started(
            server_id, enforce_capacity=True, operation_id=operation_id
        )

    async def _connection_finished(
        self,
        server_id: str,
        enforce_capacity: bool,
        operation_id: str | None = None,
        outcome: str = "completed",
        generation: int | None = None,
    ) -> None:
        if not enforce_capacity:
            return
        active = getattr(self.server_manager, "_active_inference_requests", None)
        conditions = getattr(self.server_manager, "_inference_slot_conditions", None)
        condition = conditions.get(server_id) if conditions else None
        if condition is None:
            condition = asyncio.Condition()
        async with condition:
            config = self.server_manager.configs.get(server_id)
            if (
                active is not None
                and config is not None
                and (generation is None or config.slot_generation == generation)
            ):
                remaining = active.get(server_id, 0) - 1
                if remaining > 0:
                    active[server_id] = remaining
                else:
                    active.pop(server_id, None)
            finish_operation(self.server_manager, operation_id, outcome)
            condition.notify_all()

    def _upstream_opened(self, server_id: str) -> None:
        connections = getattr(self.server_manager, "_active_connections", None)
        if connections is None:
            connections = self.server_manager._active_connections = {}
        connections[server_id] = connections.get(server_id, 0) + 1

    def _upstream_closed(self, server_id: str) -> None:
        connections = getattr(self.server_manager, "_active_connections", None)
        if connections and server_id in connections:
            remaining = connections[server_id] - 1
            if remaining > 0:
                connections[server_id] = remaining
            else:
                del connections[server_id]

    async def _release_reservation(
        self,
        server_id: str,
        enforce_capacity: bool,
        operation_id: str | None,
        outcome: str,
        reservation: _SlotReservation,
    ) -> None:
        if not reservation.acquired or reservation.released:
            return
        reservation.released = True
        release_task = asyncio.create_task(
            self._connection_finished(
                server_id,
                enforce_capacity,
                operation_id,
                outcome,
                reservation.generation,
            )
        )
        try:
            await asyncio.shield(release_task)
        except asyncio.CancelledError:
            await asyncio.shield(release_task)
            raise

    def _get_server_url(self, server_id: str, path: str) -> str:
        config = self.server_manager.configs.get(server_id)
        if not config:
            raise ValueError(f"Server {server_id} not found")
        port = getattr(config, "api_port", config.port)
        return f"http://127.0.0.1:{port}{path}"

    @staticmethod
    def _inference_timeout() -> httpx.Timeout:
        return httpx.Timeout(
            300.0,
            connect=UPSTREAM_CONNECT_TIMEOUT_SECONDS,
            read=settings.UPSTREAM_IDLE_TIMEOUT_SECONDS,
        )

    def _check_generation(self, server_id: str, expected: int | None) -> None:
        if expected is None:
            return
        config = self.server_manager.configs.get(server_id)
        if config is None or config.slot_generation != expected:
            raise httpx.HTTPStatusError(
                "Inference slot generation changed",
                request=httpx.Request("GET", "http://127.0.0.1/"),
                response=httpx.Response(409),
            )

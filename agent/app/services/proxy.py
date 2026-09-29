"""Proxies requests to llama.cpp servers."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import httpx

from app.services.inference_operations import (
    activate_operation,
    bind_operation_task,
    finish_operation,
    operation_exists,
    start_operation,
)

CONNECT_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0, 8.0, 8.0)
UPSTREAM_CONNECT_TIMEOUT_SECONDS = 10.0


class ProxyOperationExists(Exception):
    """Raised when an operation ID has already been dispatched."""


@dataclass
class _SlotReservation:
    acquired: bool = False
    released: bool = False


class ServerProxy:
    """Proxies requests to llama.cpp servers."""

    def __init__(self, server_manager) -> None:
        self.server_manager = server_manager
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=UPSTREAM_CONNECT_TIMEOUT_SECONDS)
        )

    def _get_server_port(self, server_id: str) -> int:
        """Get server port from server manager."""
        config = self.server_manager.configs.get(server_id)
        if not config:
            raise ValueError(f"Server {server_id} not found")
        return config.port

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
        reservation = _SlotReservation(acquired=capacity_reserved)
        outcome = "failed"
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
            for delay in (*CONNECT_RETRY_DELAYS, None):
                try:
                    self._check_generation(server_id, expected_slot_generation)
                    response = await self.client.request(
                        method=method, url=url, headers=headers, json=json
                    )
                    outcome = "completed"
                    return response
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    if delay is None:
                        raise
                    await asyncio.sleep(delay)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
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
        reservation = _SlotReservation(acquired=capacity_reserved)
        outcome = "failed"
        reservation_task: asyncio.Task[None] | None = None
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
                bind_operation_task(self.server_manager, operation_id)
            for delay in (*CONNECT_RETRY_DELAYS, None):
                connected = False
                try:
                    self._check_generation(server_id, expected_slot_generation)
                    async with self.client.stream(
                        method=method,
                        url=url,
                        json=json,
                        headers={"Accept": "text/event-stream"},
                    ) as response:
                        connected = True
                        async for chunk in response.aiter_bytes():
                            yield chunk
                    outcome = "completed"
                    return
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    if connected or delay is None:
                        raise
                    await asyncio.sleep(delay)
        except asyncio.CancelledError:
            outcome = "cancelled"
            if reservation_task is not None and not reservation_task.done():
                reservation_task.cancel()
                await asyncio.gather(reservation_task, return_exceptions=True)
            raise
        finally:
            if reservation_task is not None and not reservation_task.done():
                reservation_task.cancel()
                await asyncio.gather(reservation_task, return_exceptions=True)
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
        connections = getattr(self.server_manager, "_active_connections", None)
        if connections is None:
            connections = self.server_manager._active_connections = {}
        if not enforce_capacity:
            connections[server_id] = connections.get(server_id, 0) + 1
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
                        connections[server_id] = connections.get(server_id, 0) + 1
                        if reservation is not None:
                            reservation.acquired = True
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
    ) -> None:
        connections = getattr(self.server_manager, "_active_connections", None)
        if not enforce_capacity:
            if connections and server_id in connections:
                connections[server_id] -= 1
                if connections[server_id] <= 0:
                    del connections[server_id]
            return
        active = getattr(self.server_manager, "_active_inference_requests", None)
        conditions = getattr(self.server_manager, "_inference_slot_conditions", None)
        if active is None or conditions is None:
            if connections and server_id in connections:
                connections[server_id] -= 1
                if connections[server_id] <= 0:
                    del connections[server_id]
            if active is not None:
                remaining = active.get(server_id, 0) - 1
                if remaining > 0:
                    active[server_id] = remaining
                else:
                    active.pop(server_id, None)
            finish_operation(self.server_manager, operation_id, outcome)
            return
        condition = conditions.get(server_id)
        if condition is None:
            if connections and server_id in connections:
                connections[server_id] -= 1
                if connections[server_id] <= 0:
                    del connections[server_id]
            remaining = active.get(server_id, 0) - 1
            if remaining > 0:
                active[server_id] = remaining
            else:
                active.pop(server_id, None)
            finish_operation(self.server_manager, operation_id, outcome)
            return
        async with condition:
            if connections and server_id in connections:
                connections[server_id] -= 1
                if connections[server_id] <= 0:
                    del connections[server_id]
            remaining = active.get(server_id, 0) - 1
            if remaining > 0:
                active[server_id] = remaining
            else:
                active.pop(server_id, None)
            finish_operation(self.server_manager, operation_id, outcome)
            condition.notify_all()

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
                server_id, enforce_capacity, operation_id, outcome
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

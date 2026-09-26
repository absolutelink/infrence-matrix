"""Proxies requests to llama.cpp servers."""

import asyncio
from collections.abc import AsyncGenerator

import httpx

CONNECT_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0, 8.0, 8.0)


class ServerProxy:
    """Proxies requests to llama.cpp servers."""

    def __init__(self, server_manager) -> None:
        self.server_manager = server_manager
        self.client = httpx.AsyncClient(timeout=300.0)

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
    ) -> httpx.Response:
        """Proxy HTTP request to llama.cpp."""
        url = self._get_server_url(server_id, path)
        self._connection_started(server_id)

        try:
            for delay in (*CONNECT_RETRY_DELAYS, None):
                try:
                    return await self.client.request(
                        method=method, url=url, headers=headers, json=json
                    )
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    if delay is None:
                        raise
                    await asyncio.sleep(delay)
        finally:
            self._connection_finished(server_id)

    async def proxy_stream(
        self, server_id: str, method: str, path: str, json: dict | None = None
    ) -> AsyncGenerator[bytes, None]:
        """Proxy streaming request (SSE) to llama.cpp."""
        url = self._get_server_url(server_id, path)
        self._connection_started(server_id)

        try:
            for delay in (*CONNECT_RETRY_DELAYS, None):
                connected = False
                try:
                    async with self.client.stream(
                        method=method, url=url, json=json
                    ) as response:
                        connected = True
                        async for chunk in response.aiter_bytes():
                            yield chunk
                    return
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    if connected or delay is None:
                        raise
                    await asyncio.sleep(delay)
        finally:
            self._connection_finished(server_id)

    def _connection_started(self, server_id: str) -> None:
        connections = getattr(self.server_manager, "_active_connections", None)
        if connections is None:
            connections = self.server_manager._active_connections = {}
        connections[server_id] = connections.get(server_id, 0) + 1

    def _connection_finished(self, server_id: str) -> None:
        connections = getattr(self.server_manager, "_active_connections", None)
        if not connections or server_id not in connections:
            return
        connections[server_id] -= 1
        if connections[server_id] <= 0:
            del connections[server_id]

    def _get_server_url(self, server_id: str, path: str) -> str:
        config = self.server_manager.configs.get(server_id)
        if not config:
            raise ValueError(f"Server {server_id} not found")
        port = getattr(config, "api_port", config.port)
        return f"http://127.0.0.1:{port}{path}"

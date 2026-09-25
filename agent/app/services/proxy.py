"""Proxies requests to llama.cpp servers."""

from collections.abc import AsyncGenerator

import httpx


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

        response = await self.client.request(
            method=method, url=url, headers=headers, json=json
        )

        return response

    async def proxy_stream(
        self, server_id: str, method: str, path: str, json: dict | None = None
    ) -> AsyncGenerator[bytes, None]:
        """Proxy streaming request (SSE) to llama.cpp."""
        url = self._get_server_url(server_id, path)

        async with self.client.stream(method=method, url=url, json=json) as response:
            async for chunk in response.aiter_bytes():
                yield chunk

    def _get_server_url(self, server_id: str, path: str) -> str:
        config = self.server_manager.configs.get(server_id)
        if not config:
            raise ValueError(f"Server {server_id} not found")
        port = getattr(config, "api_port", config.port)
        return f"http://127.0.0.1:{port}{path}"

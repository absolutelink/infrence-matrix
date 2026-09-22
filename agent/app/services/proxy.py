"""Proxies requests to llama.cpp servers."""

from typing import AsyncGenerator
import httpx

from app.services.llama_server import LlamaServerManager


class LlamaCppProxy:
    """Proxies requests to llama.cpp servers."""

    def __init__(self, server_manager: LlamaServerManager) -> None:
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
        port = self._get_server_port(server_id)
        url = f"http://localhost:{port}{path}"

        response = await self.client.request(
            method=method, url=url, headers=headers, json=json
        )

        return response

    async def proxy_stream(
        self, server_id: str, method: str, path: str, json: dict | None = None
    ) -> AsyncGenerator[bytes, None]:
        """Proxy streaming request (SSE) to llama.cpp."""
        port = self._get_server_port(server_id)
        url = f"http://localhost:{port}{path}"

        async with self.client.stream(method=method, url=url, json=json) as response:
            async for chunk in response.aiter_bytes():
                yield chunk

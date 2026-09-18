"""Proxies llama.cpp HTTP API requests."""

from typing import AsyncGenerator, Optional
import httpx

from app.core.logging import logger


class LlamaCppProxy:
    """Proxies requests to llama.cpp servers."""
    
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(timeout=300.0)
        self.server_ports: dict[str, int] = {}
    
    def register_server(self, server_id: str, port: int) -> None:
        """Register a server's port."""
        self.server_ports[server_id] = port
        logger.info(f"Registered server {server_id} on port {port}")
    
    def unregister_server(self, server_id: str) -> None:
        """Unregister a server."""
        self.server_ports.pop(server_id, None)
    
    def _get_port(self, server_id: str) -> int:
        """Get server port."""
        if server_id not in self.server_ports:
            raise ValueError(f"Server {server_id} not registered")
        return self.server_ports[server_id]
    
    async def proxy_request(
        self,
        server_id: str,
        method: str,
        path: str,
        headers: Optional[dict] = None,
        json: Optional[dict] = None,
    ) -> httpx.Response:
        """Proxy HTTP request to llama.cpp."""
        port = self._get_port(server_id)
        url = f"http://localhost:{port}{path}"
        
        logger.debug(f"Proxying {method} {url}")
        
        response = await self.client.request(
            method=method,
            url=url,
            headers=headers or {},
            json=json,
        )
        
        return response
    
    async def proxy_stream(
        self,
        server_id: str,
        method: str,
        path: str,
        json: Optional[dict] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Proxy streaming request (SSE) to llama.cpp."""
        port = self._get_port(server_id)
        url = f"http://localhost:{port}{path}"
        
        logger.debug(f"Proxying stream {method} {url}")
        
        async with self.client.stream(
            method=method,
            url=url,
            json=json,
        ) as response:
            async for chunk in response.aiter_bytes():
                yield chunk

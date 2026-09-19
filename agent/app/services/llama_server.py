"""Manages llama.cpp subprocess lifecycle."""

import asyncio
import subprocess
import signal
import time
from typing import Dict, Optional
from dataclasses import dataclass

import httpx

from app.core.config import settings


@dataclass
class ServerConfig:
    model_path: str
    port: int
    gpu_layers: int = 35
    context_size: int = 4096
    batch_size: int = 512
    cache_prompt: bool = True
    flash_attn: bool = True


class LlamaServerManager:
    """Manages llama.cpp subprocesses."""

    def __init__(self) -> None:
        self.servers: Dict[str, subprocess.Popen] = {}
        self.configs: Dict[str, ServerConfig] = {}
        self.start_times: Dict[str, float] = {}

    async def start_server(
        self,
        server_id: str,
        config: ServerConfig
    ) -> bool:
        """Start a llama.cpp server subprocess."""
        if server_id in self.servers:
            return False

        cmd = [
            settings.LLAMA_SERVER_PATH,
            "--model", config.model_path,
            "--port", str(config.port),
            "--n-gpu-layers", str(config.gpu_layers),
            "--ctx-size", str(config.context_size),
            "--batch-size", str(config.batch_size),
        ]

        if config.cache_prompt:
            cmd.extend(["--prompt-cache", f"{settings.CACHE_PATH}/{server_id}.cache"])

        if config.flash_attn:
            cmd.append("--flash-attn")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        self.servers[server_id] = proc
        self.configs[server_id] = config
        self.start_times[server_id] = time.time()

        await self._wait_for_server(server_id, config.port)

        return True

    async def stop_server(self, server_id: str, force: bool = False) -> bool:
        """Stop a llama.cpp server."""
        if server_id not in self.servers:
            return False

        proc = self.servers[server_id]

        if force:
            proc.kill()
        else:
            proc.send_signal(signal.SIGTERM)

        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        self.servers.pop(server_id, None)
        self.configs.pop(server_id, None)
        self.start_times.pop(server_id, None)

        return True

    async def _wait_for_server(
        self,
        server_id: str,
        port: int,
        timeout: float = 30.0
    ) -> None:
        """Wait for server to be healthy."""
        start = time.time()

        while time.time() - start < timeout:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(
                        f"http://localhost:{port}/health"
                    )
                    if response.status_code == 200:
                        return
                except Exception:
                    pass

            await asyncio.sleep(1)

        raise TimeoutError(f"Server {server_id} failed to start")

    def get_server_uptime(self, server_id: str) -> float:
        """Get server uptime in seconds."""
        start_time = self.start_times.get(server_id)
        if not start_time:
            return 0.0
        return time.time() - start_time

    def list_servers(self) -> list[dict]:
        """List all running servers."""
        servers = []

        for server_id in self.servers.keys():
            config = self.configs[server_id]
            servers.append({
                "server_id": server_id,
                "model_path": config.model_path,
                "port": config.port,
                "status": "running",
                "uptime_seconds": self.get_server_uptime(server_id),
            })

        return servers

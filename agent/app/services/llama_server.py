"""Manages llama.cpp subprocess lifecycle."""

import asyncio
import subprocess
import signal
import time
from typing import Dict, Optional
from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import logger


@dataclass
class ServerConfig:
    model_path: str
    port: int
    gpu_layers: int = settings.DEFAULT_GPU_LAYERS
    context_size: int = settings.DEFAULT_CONTEXT_SIZE
    batch_size: int = settings.DEFAULT_BATCH_SIZE
    cache_prompt: bool = True


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
            logger.warning(f"Server {server_id} already running")
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
        
        logger.info(f"Starting server {server_id} on port {config.port}")
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        self.servers[server_id] = proc
        self.configs[server_id] = config
        self.start_times[server_id] = time.time()
        
        # Wait for server to start
        await self._wait_for_server(server_id, config.port)
        
        logger.info(f"Server {server_id} started successfully")
        return True
    
    async def stop_server(self, server_id: str, force: bool = False) -> bool:
        """Stop a llama.cpp server."""
        if server_id not in self.servers:
            return False
        
        proc = self.servers[server_id]
        logger.info(f"Stopping server {server_id}")
        
        if force:
            proc.kill()
        else:
            proc.send_signal(signal.SIGTERM)
        
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        
        self.servers.pop(server_id)
        self.configs.pop(server_id)
        self.start_times.pop(server_id)
        
        logger.info(f"Server {server_id} stopped")
        return True
    
    async def _wait_for_server(
        self, 
        server_id: str, 
        port: int, 
        timeout: float = 30.0
    ) -> bool:
        """Wait for server to be healthy."""
        import httpx
        
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"http://localhost:{port}/health",
                        timeout=2.0
                    )
                    if response.status_code == 200:
                        return True
            except Exception:
                pass
            
            await asyncio.sleep(1)
        
        logger.error(f"Server {server_id} failed to start within {timeout}s")
        return False
    
    def get_server_info(self, server_id: str) -> Optional[dict]:
        """Get server information."""
        if server_id not in self.servers:
            return None
        
        proc = self.servers[server_id]
        uptime = time.time() - self.start_times[server_id]
        
        return {
            "server_id": server_id,
            "port": self.configs[server_id].port,
            "uptime_seconds": int(uptime),
            "status": "running" if proc.poll() is None else "dead",
        }
    
    def list_servers(self) -> list:
        """List all running servers."""
        servers = []
        
        for server_id in self.servers:
            info = self.get_server_info(server_id)
            if info:
                servers.append(info)
        
        return servers

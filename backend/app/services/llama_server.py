import asyncio
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import httpx

from app.core.config import settings
from app.models import ServerInstance

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for llama-server instance."""
    model_path: str
    port: int
    gpu_layers: int = settings.DEFAULT_GPU_LAYERS
    context_size: int = settings.DEFAULT_CONTEXT_SIZE
    batch_size: int = settings.DEFAULT_BATCH_SIZE
    threads: int = os.cpu_count() or 4
    flash_attn: bool = True
    cache_prompt: bool = True


class LlamaServerManager:
    """Manages llama-server subprocess lifecycle."""

    def __init__(self) -> None:
        self.servers: dict[int, subprocess.Popen[bytes]] = {}
        self.configs: dict[int, ServerConfig] = {}
        self._health_check_interval = 30
        self._health_check_task: asyncio.Task[None] | None = None

    async def start_server(
        self,
        config: ServerConfig,
        server_instance: ServerInstance,
    ) -> bool:
        """Start a llama-server subprocess."""
        if config.port in self.servers:
            logger.warning(f"Server on port {config.port} already running")
            return False

        cmd = [
            settings.LLAMA_SERVER_PATH,
            "-m", config.model_path,
            "--port", str(config.port),
            "-ngl", str(config.gpu_layers),
            "-c", str(config.context_size),
            "-b", str(config.batch_size),
            "-t", str(config.threads),
        ]

        if config.flash_attn:
            cmd.append("--flash-attn")
        if config.cache_prompt:
            cmd.append("--prompt-cache")

        logger.info(f"Starting llama-server on port {config.port}: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != "nt" else None,
            )
            self.servers[config.port] = process
            self.configs[config.port] = config

            server_instance.pid = process.pid
            server_instance.status = "starting"
            server_instance.started_at = datetime.now(UTC)
            server_instance.process_command = " ".join(cmd)

            if await self._wait_for_health(config.port, timeout=60):
                server_instance.status = "running"
                server_instance.health_status = "healthy"
                server_instance.auto_shutdown_at = datetime.now(UTC).replace(
                    second=datetime.now(UTC).second + settings.SERVER_INACTIVITY_TIMEOUT
                )
                logger.info(f"Server on port {config.port} started successfully (PID: {process.pid})")
                return True
            else:
                server_instance.status = "failed"
                server_instance.error_message = "Health check failed"
                await self.stop_server(config.port)
                return False

        except Exception as e:
            logger.error(f"Failed to start server on port {config.port}: {e}")
            server_instance.status = "failed"
            server_instance.error_message = str(e)
            return False

    async def stop_server(self, port: int, graceful: bool = True) -> bool:
        """Stop a llama-server subprocess."""
        if port not in self.servers:
            logger.warning(f"No server running on port {port}")
            return False

        process = self.servers[port]
        config = self.configs.get(port)

        try:
            if graceful:
                process.terminate()
            else:
                process.kill()

            process.wait(timeout=10)
            logger.info(f"Server on port {port} stopped")
            return True

        except subprocess.TimeoutExpired:
            logger.warning(f"Server on port {port} did not terminate gracefully, killing")
            if os.name != "nt":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            return True

        except Exception as e:
            logger.error(f"Error stopping server on port {port}: {e}")
            return False

        finally:
            if port in self.servers:
                del self.servers[port]
            if config and config.port in self.configs:
                del self.configs[config.port]

    async def restart_server(self, port: int) -> bool:
        """Restart a llama-server subprocess."""
        config = self.configs.get(port)
        if not config:
            logger.error(f"No config for server on port {port}")
            return False

        await self.stop_server(port, graceful=False)
        await asyncio.sleep(1)

        server_instance = ServerInstance(
            port=port,
            model_id=config.model_id if hasattr(config, "model_id") else None,
            status="starting",
            inactivity_timeout_seconds=settings.SERVER_INACTIVITY_TIMEOUT,
        )
        return await self.start_server(config, server_instance)

    async def _wait_for_health(self, port: int, timeout: int = 60) -> bool:
        """Wait for server to become healthy."""
        start_time = time.time()
        health_url = f"http://localhost:{port}/health"

        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(health_url, timeout=5.0)
                    if response.status_code == 200:
                        return True
            except (httpx.HTTPError, httpx.ConnectError):
                pass

            await asyncio.sleep(1)

        return False

    async def check_health(self, port: int) -> Literal["healthy", "unhealthy", "unknown"]:
        """Check health of a server instance."""
        if port not in self.servers:
            return "unknown"

        process = self.servers[port]
        if process.poll() is not None:
            return "unhealthy"

        health_url = f"http://localhost:{port}/health"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    return "healthy"
        except (httpx.HTTPError, httpx.ConnectError):
            pass

        return "unhealthy"

    async def start_health_monitor(self) -> None:
        """Start background health monitoring task."""
        async def monitor() -> None:
            while True:
                for port in list(self.servers.keys()):
                    health = await self.check_health(port)
                    logger.debug(f"Server on port {port} health: {health}")

                    if health == "unhealthy":
                        config = self.configs.get(port)
                        if config:
                            logger.warning(f"Server on port {port} unhealthy, restarting...")
                            await self.restart_server(port)

                await asyncio.sleep(self._health_check_interval)

        self._health_check_task = asyncio.create_task(monitor())
        logger.info("Health monitor started")

    async def stop_health_monitor(self) -> None:
        """Stop background health monitoring task."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("Health monitor stopped")

    def get_server_stats(self, port: int) -> dict[str, int | str | bool | None] | None:
        """Get statistics for a server instance."""
        if port not in self.servers:
            return None

        process = self.servers[port]
        config = self.configs.get(port)

        return {
            "port": port,
            "pid": process.pid,
            "running": process.poll() is None,
            "model_path": config.model_path if config else None,
            "gpu_layers": config.gpu_layers if config else 0,
            "context_size": config.context_size if config else 0,
        }

    async def shutdown_all(self) -> None:
        """Shutdown all running servers."""
        logger.info(f"Shutting down {len(self.servers)} servers")

        for port in list(self.servers.keys()):
            await self.stop_server(port, graceful=True)

        await self.stop_health_monitor()
        logger.info("All servers shut down")


llama_server_manager = LlamaServerManager()

"""Manages Halogen server subprocesses behind the Matrix agent contract."""

import asyncio
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.services.event_bus import publish_event
from app.services.model_manager import model_manager

HALOGEN_REPO_ID = "peonist-ai/halogen-qwen3.8-27b"
HALOGEN_CHECKPOINT = "/models/peonist-ai/halogen-qwen3.8-27b/qwen3.8-27b-p1w4d-d2.hgn"
HALOGEN_TOKENIZER = "/models/peonist-ai/halogen-qwen3.8-27b/tokenizer"


@dataclass
class HalogenServerConfig:
    model_path: str
    port: int
    api_port: int
    engine_port: int
    options: dict[str, Any]
    cache_dir: str


class HalogenServerManager:
    """Run one independently configured Halogen process per server ID."""

    HEALTH_CHECK_INTERVAL = 10.0
    HEALTH_CHECK_TIMEOUT = 5.0

    def __init__(self) -> None:
        self.servers: dict[str, subprocess.Popen] = {}
        self.configs: dict[str, HalogenServerConfig] = {}
        self.healthy_servers: set[str] = set()
        self.server_health: dict[str, str] = {}
        self.start_times: dict[str, float] = {}
        self._health_task: asyncio.Task | None = None
        self._log_task: asyncio.Task | None = None
        self._logs: dict[str, list[str]] = {}

    @property
    def ServerConfig(self) -> type[HalogenServerConfig]:  # noqa: N802
        """Expose a constructor compatible with the legacy WS command path."""
        return HalogenServerConfig

    def start_health_monitoring(self) -> None:
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self._health_loop())

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
            for server_id in list(self.servers):
                await self._monitor(server_id)

    async def _monitor(self, server_id: str) -> None:
        process = self.servers.get(server_id)
        config = self.configs.get(server_id)
        if process is None or config is None:
            return
        exit_code = process.poll()
        if exit_code is not None:
            self._remove(server_id)
            publish_event(
                "server.error",
                {
                    "server_id": server_id,
                    "error": f"halogen exited with code {exit_code}",
                },
            )
            return
        healthy, error = await self._check_health(config.api_port)
        previous = self.server_health.get(server_id, "unknown")
        status = "healthy" if healthy else "unhealthy"
        self.server_health[server_id] = status
        if healthy:
            self.healthy_servers.add(server_id)
        else:
            self.healthy_servers.discard(server_id)
        if status != previous:
            publish_event(
                "server.health",
                {"server_id": server_id, "status": status, "error": error},
            )

    async def stop_health_monitoring(self) -> None:
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

    def start_log_forwarding(self) -> None:
        if self._log_task is None or self._log_task.done():
            self._log_task = asyncio.create_task(self._log_loop())

    async def _log_loop(self) -> None:
        while True:
            await asyncio.sleep(settings.LOG_FORWARD_INTERVAL)
            for server_id, process in list(self.servers.items()):
                if process.stderr is None:
                    continue
                try:
                    line = await asyncio.to_thread(process.stderr.readline)
                except Exception:
                    continue
                if line:
                    text = line.rstrip()
                    self._logs.setdefault(server_id, []).append(text)
                    self._logs[server_id] = self._logs[server_id][-64:]
                    logger.info("halogen %s: %s", server_id[:8], text)
                    publish_event(
                        "log.lines",
                        {
                            "server_id": server_id,
                            "lines": [{"stream": "stderr", "line": text}],
                        },
                    )

    async def start_server(self, server_id: str, config: HalogenServerConfig) -> bool:
        if server_id in self.servers:
            return False
        if len(self.servers) >= settings.HALOGEN_MAX_INSTANCES:
            raise RuntimeError("Halogen instance limit reached")

        await model_manager.download_repository(
            HALOGEN_REPO_ID, job_id=f"halogen-{server_id}"
        )

        cache_dir = Path(config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(
            {
                "HALOGEN_API_PORT": str(config.api_port),
                "HALOGEN_PORT": str(config.engine_port),
                "HALOGEN_BIND": "127.0.0.1",
                "HALOGEN_ENGINE": f"127.0.0.1:{config.engine_port}",
                "HALOGEN_CHECKPOINT": HALOGEN_CHECKPOINT,
                "HALOGEN_TOKENIZER": HALOGEN_TOKENIZER,
                "HALOGEN_CACHE_DIR": str(cache_dir),
                "XDG_CACHE_HOME": str(cache_dir),
            }
        )
        options = config.options
        mappings = {
            "drafter": "HALOGEN_DRAFTER",
            "cache_align": "HALOGEN_CACHE_ALIGN",
            "kv_slots": "HALOGEN_KV_SLOTS",
            "slot_ctx": "HALOGEN_SLOT_CTX",
            "cache_mb": "HALOGEN_CACHE_MB",
            "cache_reserve_mb": "HALOGEN_CACHE_RESERVE_MB",
            "max_tokens_cap": "HALOGEN_MAX_TOKENS_CAP",
            "queue_timeout": "HALOGEN_QUEUE_TIMEOUT",
            "w4a4": "HALOGEN_W4A4",
            "w4a4_excl": "HALOGEN_W4A4_EXCL",
            "keepalive_timeout": "HALOGEN_KEEPALIVE_TIMEOUT",
            "sse_keepalive_s": "HALOGEN_SSE_KEEPALIVE_S",
        }
        for key, variable in mappings.items():
            if key in options and options[key] is not None:
                env[variable] = str(options[key])

        command = [settings.HALOGEN_ENTRYPOINT, "all"]
        logger.info("Starting Halogen %s on API port %s", server_id, config.api_port)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        self.servers[server_id] = process
        self.configs[server_id] = config
        self.start_times[server_id] = time.time()
        try:
            await self._wait_for_health(server_id, config.api_port)
        except Exception:
            await self.stop_server(server_id, force=True)
            raise
        self.healthy_servers.add(server_id)
        self.server_health[server_id] = "healthy"
        publish_event(
            "server.started",
            {
                "server_id": server_id,
                "status": "running",
                "model_path": config.model_path,
                "port": config.api_port,
            },
        )
        return True

    async def _wait_for_health(
        self, server_id: str, port: int, timeout: float = 900
    ) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            process = self.servers.get(server_id)
            if process is None or process.poll() is not None:
                raise RuntimeError("Halogen exited before becoming healthy")
            healthy, _ = await self._check_health(port)
            if healthy:
                return
            await asyncio.sleep(1)
        raise TimeoutError("Halogen failed to become healthy")

    async def _check_health(self, port: int) -> tuple[bool, str | None]:
        try:
            async with httpx.AsyncClient(timeout=self.HEALTH_CHECK_TIMEOUT) as client:
                response = await client.get(f"http://127.0.0.1:{port}/health")
            if response.status_code == 200:
                return True, None
            return False, f"health endpoint returned HTTP {response.status_code}"
        except Exception as error:
            return False, str(error)

    async def stop_server(self, server_id: str, force: bool = False) -> bool:
        process = self.servers.get(server_id)
        if process is None:
            return False
        process.kill() if force else process.send_signal(signal.SIGTERM)
        try:
            await asyncio.to_thread(process.wait, 30)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait)
        self._remove(server_id)
        publish_event("server.stopped", {"server_id": server_id, "status": "stopped"})
        return True

    def delete_server(self, server_id: str) -> None:
        config = self.configs.get(server_id)
        cache_dir = (
            config.cache_dir if config else str(Path(settings.CACHE_PATH) / server_id)
        )
        shutil.rmtree(cache_dir, ignore_errors=True)
        self._logs.pop(server_id, None)

    def _remove(self, server_id: str) -> None:
        self.servers.pop(server_id, None)
        self.configs.pop(server_id, None)
        self.healthy_servers.discard(server_id)
        self.server_health.pop(server_id, None)
        self.start_times.pop(server_id, None)

    def list_servers(self) -> list[dict]:
        return [
            {
                "server_id": server_id,
                "model_path": config.model_path,
                "port": config.api_port,
                "status": "running",
                "health_status": self.server_health.get(server_id, "unknown"),
                "uptime_seconds": time.time()
                - self.start_times.get(server_id, time.time()),
            }
            for server_id, config in self.configs.items()
            if server_id in self.servers
        ]

    async def get_server_status(self, server_id: str) -> dict:
        if server_id not in self.servers:
            return {"status": "stopped"}
        config = self.configs[server_id]
        return {
            "status": "running",
            "server_id": server_id,
            "model_path": config.model_path,
            "port": config.api_port,
            "health_status": self.server_health.get(server_id, "unknown"),
            "uptime_seconds": time.time() - self.start_times[server_id],
        }

    def get_server_logs(self, server_id: str, lines: int = 100) -> dict:
        return {
            "server_id": server_id,
            "status": "running" if server_id in self.servers else "stopped",
            "stdout": [],
            "stderr": self._logs.get(server_id, [])[-lines:],
        }


halogen_server_manager = HalogenServerManager()

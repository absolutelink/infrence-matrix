"""Manages llama.cpp subprocess lifecycle."""

import asyncio
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.services.event_bus import publish_event


@dataclass
class ServerConfig:
    model_path: str
    port: int
    gpu_layers: int = 35
    context_size: int = 4096
    batch_size: int = 512
    cache_prompt: bool = True
    # None lets llama.cpp decide; True/False maps to --flash-attn on/off
    flash_attn: bool | None = None
    # Always-on: enables Jinja chat templates (required for tool calling,
    # used by chat/completions and OpenResponses alike)
    jinja: bool = True
    # Multimodal projector path (vision GGUF); None omits the flag
    mmproj_path: str | None = None


class LlamaServerManager:
    """Manages llama.cpp subprocesses."""

    # Number of raw log chunks (per stream) kept in memory per server
    LOG_BUFFER_CHUNKS: int = 64

    def __init__(self) -> None:
        self.servers: dict[str, subprocess.Popen] = {}
        self.configs: dict[str, ServerConfig] = {}
        self.start_times: dict[str, float] = {}
        # server_id -> list of (stream_key, raw_chunk) in order
        self._log_buffers: dict[str, list[tuple[str, str]]] = {}
        self._log_task: asyncio.Task | None = None

    def start_log_forwarding(self) -> None:
        """Start the background loop emitting log.lines events."""
        if self._log_task is None or self._log_task.done():
            self._log_task = asyncio.create_task(self._log_forward_loop())

    async def _log_forward_loop(self) -> None:
        """Periodically drain server output and emit log.lines events."""
        positions: dict[str, tuple[int, int]] = {}
        while True:
            await asyncio.sleep(settings.LOG_FORWARD_INTERVAL)
            for server_id in list(self.servers.keys()):
                from_stdout, from_stderr = positions.get(server_id, (0, 0))
                (
                    stdout_lines,
                    stderr_lines,
                    total_stdout,
                    total_stderr,
                ) = self._collect_logs(server_id, from_stdout, from_stderr)

                lines = [
                    {"stream": "stdout", "line": line} for line in stdout_lines
                ] + [{"stream": "stderr", "line": line} for line in stderr_lines]

                if lines:
                    publish_event(
                        "log.lines",
                        {
                            "server_id": server_id,
                            "lines": lines,
                        },
                    )

                positions[server_id] = (total_stdout, total_stderr)

    async def start_server(self, server_id: str, config: ServerConfig) -> bool:
        """Start a llama.cpp server subprocess."""
        if server_id in self.servers:
            return False

        cmd = [
            settings.LLAMA_SERVER_PATH,
            "--model",
            config.model_path,
            "--port",
            str(config.port),
            "--n-gpu-layers",
            str(config.gpu_layers),
            "--ctx-size",
            str(config.context_size),
            "--batch-size",
            str(config.batch_size),
        ]

        if config.flash_attn is not None:
            cmd.extend(["--flash-attn", "on" if config.flash_attn else "off"])

        if config.jinja:
            cmd.append("--jinja")

        if config.mmproj_path:
            cmd.extend(["--mmproj", config.mmproj_path])

        logger.info(
            f"Starting llama.cpp server {server_id} with command: {' '.join(cmd)}"
        )

        try:
            env = dict(os.environ)
            env.setdefault(
                "LD_LIBRARY_PATH", str(Path(settings.LLAMA_SERVER_PATH).parent)
            )
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            self.servers[server_id] = proc
            self.configs[server_id] = config
            self.start_times[server_id] = time.time()

            await self._wait_for_server(server_id, config.port)

            publish_event(
                "server.started",
                {
                    "server_id": server_id,
                    "status": "running",
                    "model_path": config.model_path,
                    "port": config.port,
                },
            )

            return True
        except Exception as e:
            logger.error(f"Failed to start server {server_id}: {e}")
            publish_event(
                "server.error",
                {
                    "server_id": server_id,
                    "error": str(e),
                },
            )
            return False

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
        self._log_buffers.pop(server_id, None)

        publish_event(
            "server.stopped",
            {
                "server_id": server_id,
                "status": "stopped",
            },
        )

        return True

    async def _wait_for_server(
        self, server_id: str, port: int, timeout: float = 30.0
    ) -> None:
        """Wait for server to be healthy."""
        start = time.time()

        while time.time() - start < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"http://localhost:{port}/health")
                    if response.status_code == 200:
                        logger.info(f"Server {server_id} is now healthy")
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

    def _drain_pipes(self, server_id: str) -> None:
        """Read any pending output into the per-server ring buffer."""
        proc = self.servers.get(server_id)
        if proc is None:
            return

        import fcntl

        for stream, key in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
            if not stream:
                continue
            try:
                fd = stream.fileno()
                orig = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, orig | os.O_NONBLOCK)
                try:
                    raw = stream.read()
                finally:
                    fcntl.fcntl(fd, fcntl.F_SETFL, orig)
                if raw:
                    self._log_buffers.setdefault(server_id, []).append((key, raw))
                    if len(self._log_buffers[server_id]) > self.LOG_BUFFER_CHUNKS:
                        self._log_buffers[server_id].pop(0)
            except TypeError:
                # Nonblocking read with no data returns None in text mode
                continue
            except Exception as e:
                logger.debug(f"Log drain error for {server_id}/{key}: {e}")

    def _collect_logs(
        self,
        server_id: str,
        from_stdout: int = 0,
        from_stderr: int = 0,
    ) -> tuple[list[str], list[str], int, int]:
        """Drain pipes and return (stdout_lines, stderr_lines, new_stdout_pos, new_stderr_pos)."""
        self._drain_pipes(server_id)

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        for stream_key, from_pos, collected in (
            ("stdout", from_stdout, stdout_lines),
            ("stderr", from_stderr, stderr_lines),
        ):
            pos = 0
            for chunk_key, chunk in self._log_buffers.get(server_id, []):
                if chunk_key != stream_key:
                    continue
                lines = chunk.splitlines()
                chunk_end = pos + len(lines)
                if chunk_end > from_pos:
                    collected.extend(lines[max(from_pos - pos, 0) :])
                pos = chunk_end

        total_stdout = sum(
            len(c.splitlines())
            for k, c in self._log_buffers.get(server_id, [])
            if k == "stdout"
        )
        total_stderr = sum(
            len(c.splitlines())
            for k, c in self._log_buffers.get(server_id, [])
            if k == "stderr"
        )

        return stdout_lines, stderr_lines, total_stdout, total_stderr

    def get_server_logs(self, server_id: str, lines: int = 100) -> dict:
        """Get recent stdout/stderr output from a llama.cpp server."""
        if server_id not in self.servers:
            return {
                "server_id": server_id,
                "status": "stopped",
                "stdout": [],
                "stderr": [],
            }

        stdout_lines, stderr_lines, _, _ = self._collect_logs(server_id)

        return {
            "server_id": server_id,
            "status": "running",
            "stdout": stdout_lines[-lines:],
            "stderr": stderr_lines[-lines:],
        }

    def list_servers(self) -> list[dict]:
        """List all running servers."""
        servers = []

        for server_id in self.servers:
            config = self.configs[server_id]
            servers.append(
                {
                    "server_id": server_id,
                    "model_path": config.model_path,
                    "port": config.port,
                    "status": "running",
                    "uptime_seconds": self.get_server_uptime(server_id),
                }
            )

        return servers

    async def get_server_status(self, server_id: str) -> dict:
        """Get detailed status of a specific server."""
        if server_id not in self.servers:
            return {"status": "stopped"}

        return {
            "status": "running",
            "server_id": server_id,
            "model_path": self.configs[server_id].model_path,
            "port": self.configs[server_id].port,
            "uptime_seconds": self.get_server_uptime(server_id),
        }


# Global instance
llama_server_manager = LlamaServerManager()

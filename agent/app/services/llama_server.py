"""Manages llama.cpp subprocess lifecycle."""

import asyncio
import os
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
    # MTP Draft N-Max value; when 0 no MTP flags are added, when > 0 add --spec-type draft-mtp --spec-draft-n-max <value>
    mtp_draft_max: int | None = None
    # Always-on: enables Jinja chat templates (required for tool calling,
    # used by chat/completions and OpenResponses alike)
    jinja: bool = True
    # Multimodal projector path (vision GGUF); None omits the flag
    mmproj_path: str | None = None
    draft_model_path: str | None = None
    options: dict[str, Any] | None = None


class LlamaServerManager:
    """Manages llama.cpp subprocesses."""

    ServerConfig = ServerConfig

    # Number of raw log chunks (per stream) kept in memory per server
    LOG_BUFFER_CHUNKS: int = 64
    HEALTH_CHECK_INTERVAL: float = 30.0
    HEALTH_CHECK_TIMEOUT: float = 5.0
    HEALTH_FAILURE_THRESHOLD: int = 3
    HEALTH_RECOVERY_THRESHOLD: int = 2

    def __init__(self) -> None:
        self.servers: dict[str, subprocess.Popen] = {}
        self.configs: dict[str, ServerConfig] = {}
        self.healthy_servers: set[str] = set()
        self.server_health: dict[str, str] = {}
        self._health_failures: dict[str, int] = {}
        self._health_successes: dict[str, int] = {}
        self._health_errors: dict[str, str | None] = {}
        self.start_times: dict[str, float] = {}
        # server_id -> lock serializing concurrent /servers/start calls for
        # the same instance (e.g. re-registration during a long download)
        self._start_locks: dict[str, asyncio.Lock] = {}
        # server_id -> list of (stream_key, raw_chunk) in order
        self._log_buffers: dict[str, list[tuple[str, str]]] = {}
        self._log_readers: dict[str, set[asyncio.Task]] = {}
        self._log_forwarding_started = False
        self._health_task: asyncio.Task | None = None

    def start_health_monitoring(self) -> None:
        """Start periodic health checks for active llama-server processes."""
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self._health_monitor_loop())

    async def _health_monitor_loop(self) -> None:
        """Detect health transitions and unexpected process exits."""
        while True:
            await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
            for server_id in list(self.servers):
                try:
                    await self._monitor_server_health(server_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Health check failed for server %s", server_id)

    async def _monitor_server_health(self, server_id: str) -> None:
        """Check one server and publish only meaningful state transitions."""
        process = self.servers.get(server_id)
        config = self.configs.get(server_id)
        if process is None or config is None:
            return

        exit_code = process.poll()
        if exit_code is not None:
            error = f"llama-server exited with code {exit_code}"
            self._remove_server_state(server_id, preserve_logs=True)
            publish_event(
                "server.error",
                {"server_id": server_id, "error": error, "exit_code": exit_code},
            )
            return

        healthy, error = await self._check_health(config.port)
        previous = self.server_health.get(server_id, "healthy")
        if healthy:
            self._health_failures[server_id] = 0
            self._health_successes[server_id] = (
                self._health_successes.get(server_id, 0) + 1
            )
            if previous == "unhealthy":
                if self._health_successes[server_id] < self.HEALTH_RECOVERY_THRESHOLD:
                    return
                self.server_health[server_id] = "healthy"
                self.healthy_servers.add(server_id)
                self._publish_health_event(server_id, previous, "healthy", None)
            return

        self._health_successes[server_id] = 0
        self._health_failures[server_id] = self._health_failures.get(server_id, 0) + 1
        self._health_errors[server_id] = error
        if previous == "unhealthy":
            return
        if self._health_failures[server_id] < self.HEALTH_FAILURE_THRESHOLD:
            return
        self.server_health[server_id] = "unhealthy"
        self.healthy_servers.discard(server_id)
        self._publish_health_event(server_id, previous, "unhealthy", error)

    def _publish_health_event(
        self,
        server_id: str,
        previous: str,
        status: str,
        error: str | None,
    ) -> None:
        """Publish a transition event for backend persistence and the UI."""
        publish_event(
            "server.health",
            {
                "server_id": server_id,
                "status": status,
                "previous_status": previous,
                "consecutive_failures": self._health_failures.get(server_id, 0),
                "error": error,
            },
        )

    def _remove_server_state(self, server_id: str, preserve_logs: bool = False) -> None:
        """Remove a dead server without emitting a normal stopped event."""
        self.servers.pop(server_id, None)
        self.configs.pop(server_id, None)
        self.healthy_servers.discard(server_id)
        self.server_health.pop(server_id, None)
        self._health_failures.pop(server_id, None)
        self._health_successes.pop(server_id, None)
        self._health_errors.pop(server_id, None)
        self.start_times.pop(server_id, None)
        readers = self._log_readers.pop(server_id, set())
        for reader in readers:
            reader.cancel()
        if not preserve_logs:
            self._log_buffers.pop(server_id, None)

    async def stop_health_monitoring(self) -> None:
        """Stop periodic health checks."""
        if self._health_task is None:
            return
        self._health_task.cancel()
        try:
            await self._health_task
        except asyncio.CancelledError:
            pass
        self._health_task = None

    def start_log_forwarding(self) -> None:
        """Start the background loop emitting log.lines events."""
        self._log_forwarding_started = True
        for server_id in self.servers:
            self._start_log_readers(server_id)

    def _start_log_readers(self, server_id: str) -> None:
        process = self.servers.get(server_id)
        if not self._log_forwarding_started or process is None:
            return
        readers = self._log_readers.setdefault(server_id, set())
        if readers:
            return
        for stream, stream_name in (
            (process.stdout, "stdout"),
            (process.stderr, "stderr"),
        ):
            if stream is None:
                continue
            reader = asyncio.create_task(
                self._read_log_lines(server_id, stream, stream_name)
            )
            readers.add(reader)

    async def _read_log_lines(self, server_id: str, stream, stream_name: str) -> None:
        process = self.servers.get(server_id)
        try:
            while process is not None and process.poll() is None:
                line = await asyncio.to_thread(stream.readline)
                if not line:
                    break
                self._log_buffers.setdefault(server_id, []).append((stream_name, line))
                if len(self._log_buffers[server_id]) > self.LOG_BUFFER_CHUNKS:
                    self._log_buffers[server_id].pop(0)
                text = line.rstrip()
                log_fn = logger.error if stream_name == "stderr" else logger.info
                log_fn(
                    "llama-server %s [%s]: %s",
                    server_id[:8],
                    stream_name,
                    text,
                )
                publish_event(
                    "log.lines",
                    {
                        "server_id": server_id,
                        "lines": [{"stream": stream_name, "line": text}],
                    },
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Log reader failed for server %s", server_id)
        finally:
            readers = self._log_readers.get(server_id)
            if readers is not None:
                readers.discard(asyncio.current_task())

    async def start_server(self, server_id: str, config: ServerConfig) -> bool:
        """Start a llama.cpp server subprocess."""
        if server_id in self.servers:
            return False

        # Serialize concurrent starts for the same instance: while one
        # caller is downloading files, a second /servers/start for the same
        # id (registration cycle) must not spawn a duplicate process. After
        # the wait, re-check so the second caller sees the running server.
        lock = self._start_locks.setdefault(server_id, asyncio.Lock())
        if lock.locked():
            async with lock:
                if server_id in self.servers:
                    return False
                logger.warning(
                    f"Server {server_id} start raced a pending start; proceeding"
                )
        async with lock:
            try:
                return await self._start_server_locked(server_id, config)
            finally:
                self._start_locks.pop(server_id, None)

    async def _start_server_locked(self, server_id: str, config: ServerConfig) -> bool:
        """Start the llama-server process (caller holds the start lock)."""
        options = config.options or {}
        # Strict Qwen MTP relies on a single sequence. Do not allow a stale or
        # incompatible parallel setting to make llama-server reject the model.
        if options.get("strict_mtp_qwen"):
            options = {**options, "parallel": 1}
        gpu_layers = options.get("gpu_layers", config.gpu_layers)
        batch_size = options.get("batch_size", config.batch_size)
        cmd = [
            settings.LLAMA_SERVER_PATH,
            "--model",
            config.model_path,
            "--port",
            str(config.port),
            "--n-gpu-layers",
            str(gpu_layers),
            "--ctx-size",
            str(options.get("context_size", config.context_size)),
            "--batch-size",
            str(batch_size),
        ]

        value_flags = {
            "threads": "--threads",
            "threads_batch": "--threads-batch",
            "ubatch_size": "--ubatch-size",
            "keep": "--keep",
            "predict": "--predict",
            "cache_type_k": "--cache-type-k",
            "cache_type_v": "--cache-type-v",
            "cache_reuse": "--cache-reuse",
            "ctx_checkpoints": "--ctx-checkpoints",
            "checkpoint_every": "--checkpoint-min-step",
            "cache_ram": "--cache-ram",
            "slot_save_path": "--slot-save-path",
            "device": "--device",
            "split_mode": "--split-mode",
            "tensor_split": "--tensor-split",
            "main_gpu": "--main-gpu",
            "fit": "--fit",
            "fit_target": "--fit-target",
            "fit_ctx": "--fit-ctx",
            "temperature": "--temperature",
            "top_k": "--top-k",
            "top_p": "--top-p",
            "min_p": "--min-p",
            "repeat_penalty": "--repeat-penalty",
            "presence_penalty": "--presence-penalty",
            "frequency_penalty": "--frequency-penalty",
            "seed": "--seed",
            "parallel": "--parallel",
            "reasoning": "--reasoning",
            "reasoning_budget": "--reasoning-budget",
            "spec_draft_p_min": "--spec-draft-p-min",
        }
        for name, flag in value_flags.items():
            if name in options and options[name] is not None:
                cmd.extend([flag, str(options[name])])

        boolean_flags = {
            "swa_full": "--swa-full",
            "kv_offload": ("--kv-offload", "--no-kv-offload"),
            "cache_prompt": ("--cache-prompt", "--no-cache-prompt"),
            "cont_batching": ("--cont-batching", "--no-cont-batching"),
            "warmup": ("--warmup", "--no-warmup"),
            "context_shift": ("--context-shift", "--no-context-shift"),
            "kv_unified": "--kv-unified",
            "no_mmap": ("--no-mmap", "--mmap"),
            "no_cache_idle_slots": (
                "--no-cache-idle-slots",
                "--cache-idle-slots",
            ),
            "strict_mtp_qwen": "--spec-mtp-strict-qwen",
        }
        for name, flags in boolean_flags.items():
            if name not in options or options[name] is None:
                continue
            if isinstance(flags, tuple):
                cmd.append(flags[0] if options[name] else flags[1])
            elif options[name]:
                cmd.append(flags)

        if config.flash_attn is not None:
            cmd.extend(["--flash-attn", "on" if config.flash_attn else "off"])

        if config.draft_model_path or (
            config.mtp_draft_max is not None and config.mtp_draft_max > 0
        ):
            cmd.extend(
                [
                    "--spec-type",
                    "draft-dflash" if config.draft_model_path else "draft-mtp",
                ]
            )
            if config.mtp_draft_max is not None and config.mtp_draft_max > 0:
                cmd.extend(["--spec-draft-n-max", str(config.mtp_draft_max)])

        if options.get("jinja", config.jinja):
            cmd.append("--jinja")

        if config.mmproj_path:
            cmd.extend(["--mmproj", config.mmproj_path])

        if config.draft_model_path:
            cmd.extend(["-md", config.draft_model_path])

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
            self._start_log_readers(server_id)

            await self._wait_for_server(server_id, config.port)
            self.healthy_servers.add(server_id)
            self.server_health[server_id] = "healthy"
            self._health_failures[server_id] = 0
            self._health_successes[server_id] = 0
            self._health_errors[server_id] = None

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
            self.healthy_servers.discard(server_id)
            self.server_health.pop(server_id, None)
            self._health_failures.pop(server_id, None)
            self._health_successes.pop(server_id, None)
            self._health_errors.pop(server_id, None)
            proc = self.servers.pop(server_id, None)
            self.configs.pop(server_id, None)
            self.start_times.pop(server_id, None)
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
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
        self.healthy_servers.discard(server_id)
        self.server_health.pop(server_id, None)
        self._health_failures.pop(server_id, None)
        self._health_successes.pop(server_id, None)
        self._health_errors.pop(server_id, None)
        self.start_times.pop(server_id, None)
        self._start_locks.pop(server_id, None)
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
            process = self.servers.get(server_id)
            if process is not None:
                exit_code = process.poll()
                if exit_code is not None:
                    self._drain_pipes(server_id)
                    _, stderr_lines, _, _ = self._collect_logs(server_id)
                    detail = "\n".join(stderr_lines[-5:])
                    message = f"Server {server_id} exited with code {exit_code}"
                    if detail:
                        message = f"{message}: {detail}"
                    raise RuntimeError(message)
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

    async def _check_health(self, port: int) -> tuple[bool, str | None]:
        """Check llama-server health and retain a useful failure reason."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{port}/health",
                    timeout=self.HEALTH_CHECK_TIMEOUT,
                )
            if response.status_code == 200:
                return True, None
            return False, f"health endpoint returned HTTP {response.status_code}"
        except Exception as error:
            return False, str(error)

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
        if not self._log_forwarding_started:
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
        if server_id not in self.servers and server_id not in self._log_buffers:
            return {
                "server_id": server_id,
                "status": "stopped",
                "stdout": [],
                "stderr": [],
            }

        stdout_lines, stderr_lines, _, _ = self._collect_logs(server_id)

        return {
            "server_id": server_id,
            "status": "running" if server_id in self.servers else "error",
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
                    "health_status": self.server_health.get(server_id, "unknown"),
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
            "health_status": self.server_health.get(server_id, "unknown"),
            "uptime_seconds": self.get_server_uptime(server_id),
        }


# Global instance
llama_server_manager = LlamaServerManager()

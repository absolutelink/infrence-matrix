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
from app.services.log_buffers import CursorLogRing


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
    slot_generation: int = 0


class LlamaServerManager:
    """Manages llama.cpp subprocesses."""

    ServerConfig = ServerConfig

    # Number of log lines kept in memory per server (cursor ring)
    LOG_BUFFER_LINES: int = 2000
    HEALTH_CHECK_INTERVAL: float = 30.0
    HEALTH_CHECK_TIMEOUT: float = 5.0
    HEALTH_FAILURE_THRESHOLD: int = 3
    HEALTH_RECOVERY_THRESHOLD: int = 2

    def __init__(self) -> None:
        self.servers: dict[str, subprocess.Popen] = {}
        self.configs: dict[str, ServerConfig] = {}
        self.slot_generations: dict[str, int] = {}
        self.healthy_servers: set[str] = set()
        self.server_health: dict[str, str] = {}
        self._health_failures: dict[str, int] = {}
        self._health_successes: dict[str, int] = {}
        self._health_errors: dict[str, str | None] = {}
        self.start_times: dict[str, float] = {}
        # server_id -> lock serializing concurrent /servers/start calls for
        # the same instance (e.g. re-registration during a long download)
        self._start_locks: dict[str, asyncio.Lock] = {}
        # server_id -> cursor ring of (seq, stream_key, line)
        self._log_buffers: dict[str, CursorLogRing] = {}
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
            slot_generation = config.slot_generation
            from app.services.inference_operations import stop_server_operations

            await stop_server_operations(self, server_id)
            self._remove_server_state(server_id, preserve_logs=True)
            publish_event(
                "server.error",
                {
                    "server_id": server_id,
                    "error": error,
                    "exit_code": exit_code,
                    "slot_generation": slot_generation,
                },
            )
            return

        healthy, error = await self._check_health(config.port)
        if (
            self.servers.get(server_id) is not process
            or self.configs.get(server_id) is not config
        ):
            return
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
                "slot_generation": self.configs[server_id].slot_generation,
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

    def _get_ring(self, server_id: str) -> CursorLogRing:
        return self._log_buffers.setdefault(
            server_id, CursorLogRing(self.LOG_BUFFER_LINES)
        )

    def _append_log(self, server_id: str, stream_name: str, text: str) -> None:
        """Store one line in the cursor ring and publish it as a log.lines event."""
        entry = self._get_ring(server_id).append(stream_name, text)
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
                "seq_start": entry.seq,
                "seq_end": entry.seq + 1,
                "lines": [{"stream": entry.stream, "line": entry.line}],
            },
        )

    async def _read_log_lines(self, server_id: str, stream, stream_name: str) -> None:
        process = self.servers.get(server_id)
        try:
            while process is not None and process.poll() is None:
                line = await asyncio.to_thread(stream.readline)
                if not line:
                    break
                self._append_log(server_id, stream_name, line.rstrip())
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
        lock = self._start_locks.setdefault(server_id, asyncio.Lock())
        try:
            async with lock:
                if server_id in self.servers:
                    current = self.configs[server_id].slot_generation
                    if config.slot_generation <= current:
                        return False
                    await self.stop_server(server_id, slot_generation=current)

                latest = self.slot_generations.get(server_id, -1)
                generation_is_fenced = latest > 0 or config.slot_generation > 0
                if generation_is_fenced and config.slot_generation <= latest:
                    return False
                return await self._start_server_locked(server_id, config)
        finally:
            if self._start_locks.get(server_id) is lock:
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
            "--metrics",
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
            self.slot_generations[server_id] = config.slot_generation
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
                    "slot_generation": config.slot_generation,
                    "effective_capacity": self.get_effective_capacity(server_id),
                    "active_inference_requests": getattr(
                        self, "_active_inference_requests", {}
                    ).get(server_id, 0),
                    "open_upstream_connections": getattr(
                        self, "_active_connections", {}
                    ).get(server_id, 0),
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
                await asyncio.to_thread(proc.wait)
            publish_event(
                "server.error",
                {
                    "server_id": server_id,
                    "error": str(e),
                    "slot_generation": config.slot_generation,
                },
            )
            return False

    async def stop_server(
        self,
        server_id: str,
        force: bool = False,
        slot_generation: int | None = None,
    ) -> bool:
        """Stop a llama.cpp server."""
        if server_id not in self.servers:
            return False

        config = self.configs[server_id]
        if slot_generation is not None and slot_generation != config.slot_generation:
            return False

        from app.services.inference_operations import stop_server_operations

        await stop_server_operations(self, server_id)

        proc = self.servers[server_id]

        if force:
            proc.kill()
        else:
            proc.send_signal(signal.SIGTERM)

        try:
            await asyncio.to_thread(proc.wait, 30)
        except subprocess.TimeoutExpired:
            proc.kill()
            await asyncio.to_thread(proc.wait)

        self.servers.pop(server_id, None)
        self.configs.pop(server_id, None)
        self.healthy_servers.discard(server_id)
        self.server_health.pop(server_id, None)
        self._health_failures.pop(server_id, None)
        self._health_successes.pop(server_id, None)
        self._health_errors.pop(server_id, None)
        self.start_times.pop(server_id, None)
        self._log_buffers.pop(server_id, None)

        publish_event(
            "server.stopped",
            {
                "server_id": server_id,
                "status": "stopped",
                "slot_generation": config.slot_generation,
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
                    ring = self._log_buffers.get(server_id)
                    stderr_lines = (
                        [e.line for e in ring.tail(100) if e.stream == "stderr"]
                        if ring
                        else []
                    )
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

    def get_effective_capacity(self, server_id: str) -> int:
        """Return the concurrency actually passed to llama-server."""
        config = self.configs[server_id]
        options = config.options or {}
        if options.get("strict_mtp_qwen"):
            return 1
        try:
            return max(int(options.get("parallel", 1)), 1)
        except (TypeError, ValueError):
            return 1

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
                    self._get_ring(server_id).extend(key, raw)
            except TypeError:
                # Nonblocking read with no data returns None in text mode
                continue
            except Exception as e:
                logger.debug(f"Log drain error for {server_id}/{key}: {e}")

    def get_server_logs(
        self, server_id: str, lines: int = 100, after: int | None = None
    ) -> dict:
        """Get recent stdout/stderr output from a llama.cpp server.

        With ``after`` (a cursor from a previous response or log.lines
        event), returns only lines with ``seq >= after`` plus ``next_cursor``.
        Without it, returns the last ``lines`` of each stream and the
        cursor just past them.
        """
        if server_id not in self.servers and server_id not in self._log_buffers:
            return {
                "server_id": server_id,
                "status": "stopped",
                "stdout": [],
                "stderr": [],
                "next_cursor": 0,
                "gap": False,
            }

        if not self._log_forwarding_started:
            self._drain_pipes(server_id)
        ring = self._log_buffers.get(server_id)
        if ring is None:
            ring = self._get_ring(server_id)

        if after is not None:
            entries, gap = ring.after(after)
            return {
                "server_id": server_id,
                "status": "running" if server_id in self.servers else "error",
                "stdout": [e.line for e in entries if e.stream == "stdout"],
                "stderr": [e.line for e in entries if e.stream == "stderr"],
                "entries": [e.to_dict() for e in entries],
                "next_cursor": entries[-1].seq + 1 if entries else after,
                "gap": gap,
            }

        tail = ring.tail(lines * 2)
        return {
            "server_id": server_id,
            "status": "running" if server_id in self.servers else "error",
            "stdout": [e.line for e in tail if e.stream == "stdout"],
            "stderr": [e.line for e in tail if e.stream == "stderr"],
            "entries": [e.to_dict() for e in tail],
            "next_cursor": ring.last_cursor,
            "gap": False,
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
                    "slot_generation": config.slot_generation,
                    "effective_capacity": self.get_effective_capacity(server_id),
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
            "slot_generation": self.configs[server_id].slot_generation,
            "effective_capacity": self.get_effective_capacity(server_id),
            "active_inference_requests": getattr(
                self, "_active_inference_requests", {}
            ).get(server_id, 0),
            "open_upstream_connections": getattr(self, "_active_connections", {}).get(
                server_id, 0
            ),
        }


# Global instance
llama_server_manager = LlamaServerManager()

"""Manages Gufo (``gufo serve llm``) subprocesses behind the Matrix agent contract."""

import asyncio
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.services.event_bus import publish_event
from app.services.log_buffers import CursorLogRing

# Maps validated engine_options keys to their ``gufo serve llm`` flags.
# Every flag is optional; omitting it lets gufo apply its own default.
VALUE_FLAGS: dict[str, str] = {
    # Model & context
    "context": "--context",
    "served_model_name": "--served-model-name",
    "mmproj": "--mmproj",
    # Sampling defaults
    "max_tokens": "--max-tokens",
    "temperature": "--temperature",
    "top_k": "--top-k",
    "top_p": "--top-p",
    "min_p": "--min-p",
    "min_keep": "--min-keep",
    "seed": "--seed",
    "repeat_penalty": "--repeat-penalty",
    "repeat_last_n": "--repeat-last-n",
    "frequency_penalty": "--frequency-penalty",
    "presence_penalty": "--presence-penalty",
    # Reasoning defaults (tri-state / enum values)
    "think": "--think",
    "reasoning_effort": "--reasoning-effort",
    "preserve_thinking": "--preserve-thinking",
    # Speculative decoding
    "speculative": "--speculative",
    "dflash_model": "--dflash-model",
    "dspark_model": "--dspark-model",
    "mtp_model": "--mtp-model",
    "draft_policy": "--draft-policy",
    "draft_tokens": "--draft-tokens",
    "min_draft_tokens": "--min-draft-tokens",
    # Scheduling and server-protection limits
    "prefill_chunk": "--prefill-chunk",
    "max_pending": "--max-pending",
    "max_pending_per_client": "--max-pending-per-client",
    "request_timeout_ms": "--request-timeout-ms",
    "max_output_bytes": "--max-output-bytes",
    "max_buffered_output_bytes": "--max-buffered-output-bytes",
    "max_buffered_output_total": "--max-buffered-output-total",
    # Disk cache
    "cache_disk_bytes": "--cache-disk-bytes",
    "cache_disk_staging_bytes": "--cache-disk-staging-bytes",
    # Server options
    "sessions": "--sessions",
    "max_connections": "--max-connections",
    "max_request_bytes": "--max-request-bytes",
    "api_key": "--api-key",
}

BOOL_FLAGS: dict[str, str] = {
    "verbose": "--verbose",
    "log_progress": "--log-progress",
}

DEFAULT_SESSIONS = 1


@dataclass
class GufoServerConfig:
    model_path: str
    port: int
    options: dict[str, Any] = field(default_factory=dict)
    slot_generation: int = 0


def build_command(config: GufoServerConfig, server_id: str) -> list[str]:
    """Assemble the ``gufo serve llm`` argv from a server config."""
    cmd = [
        settings.GUFO_SERVER_PATH,
        "serve",
        "llm",
        "--host",
        "127.0.0.1",
        "--port",
        str(config.port),
        "--model",
        config.model_path,
    ]
    options = config.options or {}
    for key, flag in VALUE_FLAGS.items():
        value = options.get(key)
        if value is not None:
            cmd.extend([flag, str(value)])
    if options.get("cache_disk") is True:
        cache_dir = Path(settings.CACHE_PATH) / server_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--cache-disk", str(cache_dir)])
    for key, flag in BOOL_FLAGS.items():
        if options.get(key) is True:
            cmd.append(flag)
    return cmd


class GufoServerManager:
    """Run one ``gufo serve llm`` process per server ID."""

    ServerConfig = GufoServerConfig

    PROCESS_LABEL = "gufo"
    LOG_BUFFER_LINES = 2000
    HEALTH_CHECK_INTERVAL = 30.0
    HEALTH_CHECK_TIMEOUT = 5.0
    HEALTH_FAILURE_THRESHOLD = 3
    HEALTH_RECOVERY_THRESHOLD = 2
    STARTUP_HEALTH_GRACE_SECONDS = 900.0

    def __init__(self) -> None:
        self.servers: dict[str, subprocess.Popen] = {}
        self.configs: dict[str, GufoServerConfig] = {}
        self.slot_generations: dict[str, int] = {}
        self.healthy_servers: set[str] = set()
        self.server_health: dict[str, str] = {}
        self._health_failures: dict[str, int] = {}
        self._health_successes: dict[str, int] = {}
        self._health_errors: dict[str, str | None] = {}
        self.start_times: dict[str, float] = {}
        self._start_locks: dict[str, asyncio.Lock] = {}
        self._log_readers: dict[str, set[asyncio.Task]] = {}
        self._log_forwarding_started = False
        self._log_buffers: dict[str, CursorLogRing] = {}
        self._health_task: asyncio.Task | None = None

    # --- Health monitoring -------------------------------------------------

    def start_health_monitoring(self) -> None:
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self._health_loop())

    async def stop_health_monitoring(self) -> None:
        if self._health_task is None:
            return
        self._health_task.cancel()
        try:
            await self._health_task
        except asyncio.CancelledError:
            pass
        self._health_task = None

    async def _health_loop(self) -> None:
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
        process = self.servers.get(server_id)
        config = self.configs.get(server_id)
        if process is None or config is None:
            return

        exit_code = process.poll()
        if exit_code is not None:
            error = f"gufo exited with code {exit_code}"
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

        started_at = self.start_times.get(server_id)
        if (
            started_at is not None
            and time.time() - started_at < self.STARTUP_HEALTH_GRACE_SECONDS
        ):
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

    async def _check_health(self, port: int) -> tuple[bool, str | None]:
        try:
            async with httpx.AsyncClient(timeout=self.HEALTH_CHECK_TIMEOUT) as client:
                response = await client.get(f"http://127.0.0.1:{port}/health")
            if response.status_code == 200:
                return True, None
            return False, f"health endpoint returned HTTP {response.status_code}"
        except Exception as error:
            return False, str(error)

    # --- Logging ---------------------------------------------------------

    def start_log_forwarding(self) -> None:
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
        entry = self._get_ring(server_id).append(stream_name, text)
        log_fn = logger.error if stream_name == "stderr" else logger.info
        log_fn("gufo %s [%s]: %s", server_id[:8], stream_name, text)
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

    # --- Lifecycle -------------------------------------------------------

    async def start_server(self, server_id: str, config: GufoServerConfig) -> bool:
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

    async def _start_server_locked(
        self, server_id: str, config: GufoServerConfig
    ) -> bool:
        if len(self.servers) >= settings.GUFO_MAX_INSTANCES:
            raise RuntimeError("Gufo instance limit reached")

        cmd = build_command(config, server_id)
        logger.info("Starting gufo server %s: %s", server_id, " ".join(cmd))

        proc: subprocess.Popen | None = None
        try:
            env = dict(os.environ)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True,
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
            logger.error(f"Failed to start gufo server {server_id}: {e}")
            self._clear_health_state(server_id)
            live = self.servers.pop(server_id, None)
            self.configs.pop(server_id, None)
            self.start_times.pop(server_id, None)
            if live is not None and live.poll() is None:
                self._kill_group(live, force=True)
                await asyncio.to_thread(live.wait)
            elif proc is not None and proc.poll() is None:
                self._kill_group(proc, force=True)
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

    def _clear_health_state(self, server_id: str) -> None:
        self.healthy_servers.discard(server_id)
        self.server_health.pop(server_id, None)
        self._health_failures.pop(server_id, None)
        self._health_successes.pop(server_id, None)
        self._health_errors.pop(server_id, None)

    async def stop_server(
        self,
        server_id: str,
        force: bool = False,
        slot_generation: int | None = None,
    ) -> bool:
        if server_id not in self.servers:
            return False

        config = self.configs[server_id]
        if slot_generation is not None and slot_generation != config.slot_generation:
            return False

        from app.services.inference_operations import stop_server_operations

        await stop_server_operations(self, server_id)

        proc = self.servers[server_id]
        self._kill_group(proc, force=force)
        try:
            await asyncio.to_thread(proc.wait, 30)
        except subprocess.TimeoutExpired:
            self._kill_group(proc, force=True)
            await asyncio.to_thread(proc.wait)

        self.servers.pop(server_id, None)
        self.configs.pop(server_id, None)
        self._clear_health_state(server_id)
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

    @staticmethod
    def _kill_group(proc: subprocess.Popen, force: bool) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            pass

    async def _wait_for_server(
        self, server_id: str, port: int, timeout: float | None = None
    ) -> None:
        """Wait until the gufo HTTP server answers its health endpoint."""
        if timeout is None:
            timeout = float(settings.SERVER_START_HEALTH_TIMEOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
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
                    message = f"Gufo server {server_id} exited with code {exit_code}"
                    if detail:
                        message = f"{message}: {detail}"
                    raise RuntimeError(message)
            healthy, _ = await self._check_health(port)
            if healthy:
                logger.info(f"Gufo server {server_id} is now healthy")
                return
            await asyncio.sleep(1)

        raise TimeoutError(f"Gufo server {server_id} failed to start")

    def _drain_pipes(self, server_id: str) -> None:
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
                continue
            except Exception as e:
                logger.debug(f"Log drain error for {server_id}/{key}: {e}")

    # --- Introspection ---------------------------------------------------

    def get_server_uptime(self, server_id: str) -> float:
        start_time = self.start_times.get(server_id)
        if not start_time:
            return 0.0
        return time.time() - start_time

    def get_effective_capacity(self, server_id: str) -> int:
        """Gufo concurrency is the number of preallocated GPU sessions."""
        config = self.configs[server_id]
        options = config.options or {}
        try:
            return max(int(options.get("sessions", DEFAULT_SESSIONS)), 1)
        except (TypeError, ValueError):
            return DEFAULT_SESSIONS

    def list_servers(self) -> list[dict]:
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

    def get_server_logs(
        self, server_id: str, lines: int = 100, after: int | None = None
    ) -> dict:
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


gufo_server_manager = GufoServerManager()

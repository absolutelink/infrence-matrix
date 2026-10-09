"""HalogenBackend: BackendDriver over a managed halogen subprocess.

Ported from the legacy agent's halogen manager, re-shaped to the
Phase 4 `BackendDriver` contract.

Halogen specifics:

- **Two ports.** The process exposes an OpenAI-compatible API port and a
  private engine port. Since the port model overhaul the driver allocates two
  distinct OS-assigned free loopback ports per backend at start (the admin never
  learns them; it dials the agent's single env `PROVIDER_PORT` and routes by
  model). Health and all /v1 traffic go to the API port; the engine port only
  appears in HALOGEN_ENGINE.
- **Env-configured.** Almost all configuration flows through HALOGEN_*
  env vars (see `provider_halogen.env`); argv is just the entrypoint
  word `all`. Phase 12 sections flatten into the semantic options dict
  via `env.flatten_options()`.
- **Capacity = kv_slots.** The lifecycle capacity comes from the
  registration response (`provider_definition.capacity`); the engine's
  effective KV-slot capacity is reported in the `backend.start` ack.
- **Artifacts.** `backend_config.artifacts.model` is the `.hgn`
  checkpoint and `artifacts.tokenizer` the tokenizer directory; both
  resolve through the lib downloader (or a plain local `{"path": ...}`).
  The legacy top-level `model` / `tokenizer` keys are accepted as
  fallback.
"""

import asyncio
import contextlib
import json
import logging
import os
import shutil
import signal
import subprocess
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from provider_lib.backend import BackendDriver
from provider_lib.config import ProviderSettings
from provider_lib.downloader import ensure_artifact
from provider_lib.log_ring import CursorLogRing
from provider_lib.schema import load_schema, validate_backend_config
from provider_lib.wire import BackendStatusValue

from provider_halogen.env import (
    DRIVER_ENV_KEYS,
    allocate_ports,
    build_argv,
    build_env,
    effective_capacity,
    flatten_options,
)

logger = logging.getLogger("provider.halogen")

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]
# (backend_status, reason) — the BackendLifecycle.report channel, bound by
# attach_status_callback. Used for long-init heartbeats only.
StatusCallback = Callable[[str, str | None], Awaitable[None]]

_LOG_BUFFER_LINES = 2000
_HEALTH_POLL_INTERVAL = 0.25
_STDERR_TAIL_LINES = 5
# Seconds between `initializing` heartbeats while waiting for the API, and
# how much of the engine's newest line travels with each one.
_BOOT_HEARTBEAT_SECONDS = 15.0
_HEARTBEAT_REASON_CHARS = 160

# The halogen committed backend_config schema (Phase 12): the sectioned
# shape the admin registers, validates against, and renders in the UI.
# Shared with main.py (single load → registration + driver validation).
SCHEMA: dict[str, Any] = load_schema("provider_halogen")


class HalogenBackend(BackendDriver):
    """Owns one halogen (api+engine) process for this provider instance."""

    def __init__(
        self,
        settings: ProviderSettings,
        backend_config: dict[str, Any],
        *,
        progress_cb: ProgressCallback | None = None,
        status_cb: StatusCallback | None = None,
    ) -> None:
        self._settings = settings
        self._config = backend_config or {}
        self._progress_cb = progress_cb
        self._status_cb = status_cb
        self._binary = settings.HALOGEN_SERVER_PATH
        # Port model overhaul: the (api, engine) pair is a private concern
        # allocated at start as two OS-assigned free loopback ports (the admin
        # never learns them). ``None`` until :meth:`start` binds it.
        self._ports: tuple[int, int] | None = None
        self._proc: subprocess.Popen[str] | None = None
        # Legacy merges stderr into stdout (single pipe); the ring labels
        # merged lines "stdout".
        self._log_ring = CursorLogRing(_LOG_BUFFER_LINES)
        self._log_reader: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None
        # Phase 9: local paths resolved during the last start, used by
        # storage.prune_unused as the "referenced artifact set".
        self.resolved_artifacts: list[str] = []

    def attach_status_callback(self, emit: StatusCallback) -> None:
        """Bind the lifecycle's out-of-band status reporter.

        Called by `BackendLifecycle.__init__` (it detects the hook). Used
        ONLY for `initializing` heartbeats during a long engine boot; state
        transitions stay the lifecycle's, so a heartbeat can never make a
        not-yet-serving backend look servable.
        """
        self._status_cb = emit

    def apply_config(self, backend_config: dict[str, Any]) -> None:
        """Adopt a (possibly updated) sectioned backend_config before start.

        Schema violations are logged as warnings, not raised: the
        authoritative gates are the admin's registration schema check and
        definition CRUD validation (docs/ws-protocol.md §2).
        """
        self._config = backend_config or {}
        for err in validate_backend_config(SCHEMA, self._config):
            logger.warning("backend_config schema violation: %s", err)
        # The (api, engine) pair is allocated at start (OS-assigned free ports),
        # never read from config — nothing to recompute here.
        # Config changed: previously resolved artifacts are stale.
        self.resolved_artifacts = []

    def _options(self) -> dict[str, Any]:
        """Flat semantic options for env building (Phase 12 sections +
        legacy `cfg["options"]`; see env.flatten_options)."""
        return flatten_options(self._config)

    @property
    def api_port(self) -> int | None:
        return self._ports[0] if self._ports is not None else None

    @property
    def engine_port(self) -> int | None:
        return self._ports[1] if self._ports is not None else None

    @property
    def backend_port(self) -> int | None:
        """The port the admin-facing streams come from (the API port)."""
        return self._ports[0] if self._ports is not None else None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    @property
    def process(self) -> subprocess.Popen[str] | None:
        return self._proc

    @property
    def effective_capacity(self) -> int:
        """Halogen concurrency = HALOGEN_KV_SLOTS (flattened kv_slots)."""
        return effective_capacity(self._options())

    # ------------------------------------------------------------------
    # Artifact resolution
    # ------------------------------------------------------------------
    def _artifact_descriptor(self, name: str) -> Any:
        """Read an artifact descriptor from the `artifacts` section
        (Phase 12), falling back to the legacy top-level key."""
        artifacts = self._config.get("artifacts")
        if isinstance(artifacts, dict) and artifacts.get(name) is not None:
            return artifacts[name]
        return self._config.get(name)

    async def _resolve_artifact(self, descriptor: Any, kind: str) -> str:
        """Resolve a halogen artifact to a local path.

        The tokenizer is a DIRECTORY artifact, so `{"path": ...}` is
        checked with exists() (file or dir) here instead of the lib's
        file-only local path; HF repo descriptors still go through
        `ensure_artifact` (with the descriptor's optional `revision`
        pinned, mirroring the llama-cpp driver).
        """
        if descriptor is None:
            raise RuntimeError(f"backend_config missing '{kind}' artifact")
        if isinstance(descriptor, str):
            p = Path(descriptor)
            if not p.exists():
                raise RuntimeError(f"local {kind} path not found: {descriptor}")
            return str(p)
        if isinstance(descriptor, dict) and descriptor.get("path"):
            p = Path(str(descriptor["path"]))
            if not p.exists():
                raise RuntimeError(f"local {kind} path not found: {p}")
            return str(p)
        # Only a non-empty string pins the HF revision (empty/absent = None).
        revision = descriptor.get("revision") if isinstance(descriptor, dict) else None
        if not isinstance(revision, str) or not revision.strip():
            revision = None
        return await ensure_artifact(
            descriptor,
            models_dir=self._settings.MODELS_DIR,
            progress_cb=self._progress_cb,
            revision=revision,
        )

    async def _resolve_artifacts(self) -> tuple[str, str]:
        checkpoint = await self._resolve_artifact(
            self._artifact_descriptor("model"), "model"
        )
        tokenizer = await self._resolve_artifact(
            self._artifact_descriptor("tokenizer"), "tokenizer"
        )
        return checkpoint, tokenizer

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Resolve artifacts, spawn halogen with HALOGEN_* env, wait health."""
        if self._proc is not None and self._proc.poll() is None:
            return
        # The argv wraps the entrypoint in stdbuf, so a missing binary
        # would otherwise surface as a confusing stdbuf exit; check first.
        if not shutil.which(self._binary) and not Path(self._binary).exists():
            raise RuntimeError(f"halogen entrypoint not found: {self._binary}")
        checkpoint_path, tokenizer_path = await self._resolve_artifacts()
        self.resolved_artifacts = [checkpoint_path, tokenizer_path]
        # Port model overhaul: allocate a private OS-assigned (api, engine) pair
        # now (the admin never sees it). A stolen port makes the engine fail to
        # bind and exit early, surfacing as a boot error the scheduler re-drives.
        self._ports = allocate_ports()
        api_port, engine_port = self._ports
        env = build_env(
            self._options(),
            api_port=api_port,
            engine_port=engine_port,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
        )
        cmd = build_argv(self._binary)
        logger.info(
            "starting halogen: %s (api=%s engine=%s)",
            " ".join(cmd),
            api_port,
            engine_port,
        )
        logger.info(
            "halogen env: %s",
            " ".join(
                f"{k}={str(env[k]).replace(chr(10), ' ').replace(chr(13), ' ')}"
                for k in sorted(DRIVER_ENV_KEYS)
                if k in env
            ),
        )
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                # Legacy: stderr merged into stdout so a single reader
                # keeps the pipes drained (avoids the stderr-pipe
                # deadlock when the entrypoint is chatty).
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,
                bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(
                f"failed to spawn halogen ({self._binary}): {exc}"
            ) from exc

        self._proc = proc
        self._start_log_reader()
        try:
            await self._wait_for_health()
        except BaseException:
            await self.stop()
            raise

    async def _wait_for_health(self) -> None:
        """Wait out the engine's start: companion downloads and index build.

        Budget is `ENGINE_BOOT_TIMEOUT` (default 1h), not
        `SERVER_START_HEALTH_TIMEOUT`: the halogen entrypoint may fetch the
        checkpoint and its companions from HuggingFace (`HALOGEN_DOWNLOAD`)
        before the API port answers, and a large checkpoint plus indexer
        build is slow even when everything is on disk. `initializing`
        heartbeats carry the engine's newest log line so the admin shows
        progress instead of a hang; a process that dies is still detected
        immediately from its exit code.
        """
        timeout = float(self._settings.ENGINE_BOOT_TIMEOUT)
        deadline = time.monotonic() + timeout
        started = time.monotonic()
        next_beat = started + _BOOT_HEARTBEAT_SECONDS
        while time.monotonic() < deadline:
            proc = self._proc
            if proc is None:
                raise RuntimeError("halogen process disappeared")
            exit_code = proc.poll()
            if exit_code is not None:
                await self._drain_final_output()
                raise RuntimeError(self._early_exit_message(exit_code))
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    logger.info("halogen API healthy on port %s", self.api_port)
                    return
            except Exception:  # noqa: BLE001 - not up yet
                pass
            now = time.monotonic()
            if self._status_cb is not None and now >= next_beat:
                next_beat = now + _BOOT_HEARTBEAT_SECONDS
                await self._heartbeat(now - started)
            await asyncio.sleep(_HEALTH_POLL_INTERVAL)
        raise TimeoutError(f"halogen failed to become healthy within {timeout:.0f}s")

    async def _heartbeat(self, elapsed: float) -> None:
        """Report `initializing` + the engine's newest line (best effort)."""
        if self._status_cb is None:  # pragma: no cover - guarded by caller
            return
        reason = f"halogen initializing ({elapsed:.0f}s elapsed)"
        line = self._latest_log_line()
        if line:
            reason = f"{reason}: {line}"
        try:
            await self._status_cb(BackendStatusValue.INITIALIZING, reason)
        except Exception:  # noqa: BLE001 - progress never breaks a boot
            logger.debug("boot heartbeat failed", exc_info=True)

    def _latest_log_line(self) -> str:
        """Newest non-empty engine log line, truncated for a status reason."""
        for entry in reversed(self._log_ring.tail(20)):
            text = entry.text.strip()
            if not text:
                continue
            if len(text) <= _HEARTBEAT_REASON_CHARS:
                return text
            return f"{text[: _HEARTBEAT_REASON_CHARS - 3]}..."
        return ""

    def _early_exit_message(self, exit_code: int | None) -> str:
        lines = [e.text for e in self._log_ring.tail(100)]
        detail = "\n".join(lines[-_STDERR_TAIL_LINES:])
        message = f"halogen exited with code {exit_code}"
        if detail:
            message = f"{message}: {detail}"
        return message

    # ------------------------------------------------------------------
    # Log capture (merged stdout+stderr ring; streamer drains it)
    # ------------------------------------------------------------------
    def _start_log_reader(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        self._log_reader = asyncio.create_task(self._read_log_lines(proc.stdout))

    async def _read_log_lines(self, stream: Any) -> None:
        proc = self._proc
        try:
            while proc is not None and proc.poll() is None:
                line = await asyncio.to_thread(stream.readline)
                if not line:
                    break
                self._append_log("stdout", line.rstrip())
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("halogen log reader failed")

    async def _drain_final_output(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        # Join the live reader task first: two concurrent readers on the
        # same merged pipe can split lines.
        if self._log_reader is not None:
            self._log_reader.cancel()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    asyncio.gather(self._log_reader, return_exceptions=True),
                    timeout=10,
                )
            self._log_reader = None
        try:
            text = await asyncio.to_thread(proc.stdout.read)
        except Exception:  # noqa: BLE001
            return
        if text:
            for line in text.splitlines():
                if line:
                    self._append_log("stdout", line)

    @property
    def log_ring(self) -> CursorLogRing:
        """Shared Phase 13 ring (provider_lib.log_stream reads this)."""
        return self._log_ring

    def _append_log(self, stream_name: str, line: str) -> None:
        # Phase 13: the ring is the single sink; the LogStreamer (wired
        # in main.py) drains it into batched backend.logs frames. No
        # per-line WS send happens here anymore.
        self._log_ring.append(stream_name, line)

    def get_logs(self, tail: int = 100) -> list[dict[str, str]]:
        return [{"stream": e.stream, "text": e.text} for e in self._log_ring.tail(tail)]

    # ------------------------------------------------------------------
    # Health / models
    # ------------------------------------------------------------------
    async def health(self) -> bool:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return False
        try:
            client = self._http_client()
            resp = await client.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        client = self._http_client()
        resp = await client.get(f"{self.base_url}/v1/models", timeout=10.0)
        if resp.status_code != 200:
            raise RuntimeError(f"halogen /v1/models returned HTTP {resp.status_code}")
        data = resp.json().get("data", [])
        return list(data)

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=None)
        return self._client

    # ------------------------------------------------------------------
    # Streaming (SSE proxy from the API port)
    # ------------------------------------------------------------------
    async def _stream_sse(
        self, path: str, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        body = dict(request)
        body["stream"] = True
        client = self._http_client()
        async with client.stream("POST", f"{self.base_url}{path}", json=body) as resp:
            if resp.status_code != 200:
                detail = await resp.aread()
                raise RuntimeError(
                    f"upstream {path} returned HTTP {resp.status_code}: "
                    f"{detail.decode(errors='replace')[:500]}"
                )
            async for raw in resp.aiter_lines():
                line = raw.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    yield json.loads(data)
                except ValueError:
                    logger.debug("skipping unparseable SSE data: %r", data[:120])
        # Exiting the `async with` (normal end or GeneratorExit) closes
        # the upstream response: the halogen request is cancelled and the
        # lifecycle releases the slot on upstream close.

    def stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        return self._stream_sse("/v1/responses", request)

    def stream_chat_completions(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        return self._stream_sse("/v1/chat/completions", request)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    async def stop(self) -> None:
        proc = self._proc
        self._proc = None
        # Port model overhaul: the (api, engine) pair is allocated per start, so
        # clear it here to keep the "None until start" invariant between boots.
        self._ports = None
        if proc is not None and proc.poll() is None:
            # The entrypoint spawns the engine + API as children; kill the
            # process group so no child survives a failed stop.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                await asyncio.to_thread(proc.wait, 30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                await asyncio.to_thread(proc.wait)
        if self._log_reader is not None:
            self._log_reader.cancel()
            with contextlib.suppress(Exception):
                # gather(return_exceptions=True) absorbs the reader task's
                # CancelledError (a bare wait_for would re-raise it out of
                # stop() and break the lifecycle's STOPPING→STOPPED path).
                await asyncio.wait_for(
                    asyncio.gather(self._log_reader, return_exceptions=True),
                    timeout=10,
                )
            self._log_reader = None

    async def aclose(self) -> None:
        await self.stop()
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

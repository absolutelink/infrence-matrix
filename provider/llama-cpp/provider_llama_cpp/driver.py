"""LlamaCppBackend: BackendDriver over a managed llama-server subprocess.

Ported from the legacy agent's llama_server manager, re-shaped to the
Phase 4 `BackendDriver` contract: the lifecycle/state machine/slots
live in provider_lib; this class only owns the subprocess, its log ring,
and the SSE proxying to the upstream OpenAI-compatible endpoints.

Critical stream-close semantics: the driver's stream generators close
the upstream httpx response in their `finally`, so when the lifecycle's
pump closes the generator (client disconnect or exhaustion) the
llama-server request is cancelled and the slot is released.
"""

import asyncio
import contextlib
import json
import logging
import os
import signal
import socket
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

from provider_llama_cpp.command import build_llama_command
from provider_llama_cpp.rates import inject_rates, parse_rate_gauges

logger = logging.getLogger("provider.llama_cpp")

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]

_LOG_BUFFER_LINES = 2000
_HEALTH_POLL_INTERVAL = 0.25
_STDERR_TAIL_LINES = 5
_METRICS_TIMEOUT = 5.0

# The llama-cpp committed backend_config schema (Phase 12): the sectioned
# shape the admin registers, validates against, and renders in the UI.
# Shared with main.py (single load → registration + driver validation).
SCHEMA: dict[str, Any] = load_schema("provider_llama_cpp")


def _pick_free_port() -> int:
    """Ask the OS for a free loopback port (bind ``127.0.0.1:0``, read back,
    release).

    Port model overhaul: each backend's llama-server engine binds a private
    local port inside the container that the admin never learns or dials. The
    agent's single ``/v1`` surface (``PROVIDER_PORT``) routes requests to this
    backend by model, so the engine port is an internal detail. A tiny TOCTOU
    window exists between releasing the probed port and the engine binding it;
    if the port is stolen the engine fails to bind and exits early, surfacing as
    a boot error that the admin scheduler re-drives on the next ``backend.start``
    (which re-picks a fresh port).
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LlamaCppBackend(BackendDriver):
    """Owns one llama-server subprocess for this provider instance."""

    def __init__(
        self,
        settings: ProviderSettings,
        backend_config: dict[str, Any],
        *,
        progress_cb: ProgressCallback | None = None,
    ) -> None:
        self._settings = settings
        self._config = backend_config or {}
        self._progress_cb = progress_cb
        self._binary = settings.LLAMA_SERVER_PATH
        # Port model overhaul: the engine's private local port is allocated at
        # start (OS-assigned free port), NOT derived from config or the agent's
        # env port. ``None`` until :meth:`start` binds it.
        self.backend_port: int | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._log_ring = CursorLogRing(_LOG_BUFFER_LINES)
        self._log_readers: list[asyncio.Task[None]] = []
        self._client: httpx.AsyncClient | None = None
        # Phase 9: local paths resolved during the last start, used by
        # storage.prune_unused as the "referenced artifact set".
        self.resolved_artifacts: list[str] = []

    def apply_config(self, backend_config: dict[str, Any]) -> None:
        """Adopt a (possibly updated) sectioned backend_config before start.

        Schema violations are logged as warnings, not raised: the
        authoritative gates are the admin's registration schema check and
        definition CRUD validation (docs/ws-protocol.md §2).
        """
        self._config = backend_config or {}
        for err in validate_backend_config(SCHEMA, self._config):
            logger.warning("backend_config schema violation: %s", err)
        # The engine port is allocated at start (OS-assigned free port), never
        # read from config — nothing to recompute here.
        # Config changed: previously resolved artifacts are stale.
        self.resolved_artifacts = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.backend_port}"

    @property
    def process(self) -> subprocess.Popen[str] | None:
        return self._proc

    @property
    def log_ring(self) -> CursorLogRing:
        """Shared Phase 13 ring (provider_lib.log_stream reads this)."""
        return self._log_ring

    # ------------------------------------------------------------------
    # Artifact resolution
    # ------------------------------------------------------------------
    async def _ensure(self, descriptor: Any, kind: str) -> str | None:
        if descriptor is None:
            return None
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"{kind} artifact descriptor must be an object")
        # The hfFile schema allows an optional `revision`; only a
        # non-empty string pins the HF revision (empty/absent = None).
        revision = descriptor.get("revision")
        if not isinstance(revision, str) or not revision.strip():
            revision = None
        return await ensure_artifact(
            descriptor,
            models_dir=self._settings.MODELS_DIR,
            progress_cb=self._progress_cb,
            revision=revision,
        )

    async def _resolve_artifacts(self) -> tuple[str, str | None, str | None]:
        # Phase 12: artifact descriptors live under the `artifacts`
        # section; the legacy top-level keys are accepted as fallback.
        artifacts = self._config.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = self._config

        def descriptor(name: str) -> Any:
            value = artifacts.get(name)
            if value is None and artifacts is not self._config:
                value = self._config.get(name)
            return value

        model = descriptor("model")
        if not model:
            raise RuntimeError("backend_config missing 'model' artifact")
        model_path = await self._ensure(model, "model")
        mmproj_path = await self._ensure(descriptor("mmproj"), "mmproj")
        draft_path = await self._ensure(descriptor("draft"), "draft")
        self.resolved_artifacts = [
            p for p in (model_path, mmproj_path, draft_path) if p
        ]
        return model_path, mmproj_path, draft_path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Resolve model files, allocate a free engine port, spawn llama-server,
        wait for /health.

        Port model overhaul: the engine binds an OS-assigned private loopback
        port (the admin never learns it). A tiny TOCTOU race exists between
        probing the free port and the engine binding it; if the port is stolen
        the engine fails to bind and exits early, which surfaces here as a boot
        error. That rare race is deliberately NOT retried in-process — the admin
        scheduler re-drives ``backend.start`` (which re-picks a fresh port) on
        the next boot attempt.
        """
        if self._proc is not None and self._proc.poll() is None:
            return
        model_path, mmproj_path, draft_path = await self._resolve_artifacts()
        self.backend_port = _pick_free_port()
        cmd = build_llama_command(
            self._config,
            model_path=model_path,
            port=self.backend_port,
            binary=self._binary,
            mmproj_path=mmproj_path,
            draft_path=draft_path,
            modality=self.modality,
        )
        logger.info("starting llama-server: %s", " ".join(cmd))
        self._spawn(cmd)
        self._start_log_readers()
        try:
            await self._wait_for_health()
        except BaseException:
            await self.stop()
            raise

    def _spawn(self, cmd: list[str]) -> None:
        """Popen the llama-server subprocess (raises OSError on exec failure)."""
        try:
            env = dict(os.environ)
            # Vulkan/containers may not provide LD_LIBRARY_PATH; default it
            # to the binary's directory (legacy behavior).
            env.setdefault("LD_LIBRARY_PATH", str(Path(self._binary).parent))
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        except OSError as exc:
            raise RuntimeError(
                f"failed to spawn llama-server ({self._binary}): {exc}"
            ) from exc

    async def _wait_for_health(self) -> None:
        timeout = float(self._settings.SERVER_START_HEALTH_TIMEOUT)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            proc = self._proc
            if proc is None:
                raise RuntimeError("llama-server process disappeared")
            exit_code = proc.poll()
            if exit_code is not None:
                await self._drain_final_output()
                raise RuntimeError(self._early_exit_message(exit_code))
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    logger.info("llama-server healthy on port %s", self.backend_port)
                    return
            except Exception:  # noqa: BLE001 - not up yet
                pass
            await asyncio.sleep(_HEALTH_POLL_INTERVAL)
        raise TimeoutError(
            f"llama-server failed to become healthy within {timeout:.0f}s"
        )

    def _early_exit_message(self, exit_code: int | None) -> str:
        stderr_lines = [
            e.text for e in self._log_ring.tail(100) if e.stream == "stderr"
        ]
        detail = "\n".join(stderr_lines[-_STDERR_TAIL_LINES:])
        message = f"llama-server exited with code {exit_code}"
        if detail:
            message = f"{message}: {detail}"
        return message

    # ------------------------------------------------------------------
    # Log capture
    # ------------------------------------------------------------------
    def _start_log_readers(self) -> None:
        proc = self._proc
        if proc is None:
            return
        for stream, name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
            if stream is None:
                continue
            self._log_readers.append(
                asyncio.create_task(self._read_log_lines(stream, name))
            )

    async def _read_log_lines(self, stream: Any, name: str) -> None:
        proc = self._proc
        try:
            while proc is not None and proc.poll() is None:
                line = await asyncio.to_thread(stream.readline)
                if not line:
                    break
                self._append_log(name, line.rstrip())
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("log reader (%s) failed", name)

    async def _drain_final_output(self) -> None:
        """Read whatever the process left in its pipes after exiting."""
        proc = self._proc
        if proc is None:
            return
        for stream, name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
            if stream is None:
                continue
            try:
                text = await asyncio.to_thread(stream.read)
            except Exception:  # noqa: BLE001
                continue
            if text:
                for line in text.splitlines():
                    if line:
                        self._append_log(name, line)

    def _append_log(self, stream_name: str, line: str) -> None:
        # Phase 13: the ring is the single sink; the LogStreamer (wired
        # in main.py) drains it into batched backend.logs frames. No
        # per-line WS send happens here anymore.
        self._log_ring.append(stream_name, line)

    def get_logs(self, tail: int = 100) -> list[dict[str, str]]:
        """Recent stdout/stderr lines as {stream, text} dicts."""
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
            raise RuntimeError(
                f"llama-server /v1/models returned HTTP {resp.status_code}"
            )
        data = resp.json().get("data", [])
        return list(data)

    async def embeddings(self, request: dict[str, Any]) -> dict[str, Any]:
        """Proxy an OpenAI embeddings request to the upstream llama-server.

        Phase 18: awaited by provider_lib's slot-admitted
        ``POST /v1/embeddings`` route. The backend must have been booted
        with ``--embedding`` (modality == "embedding") for the upstream to
        serve this path. Returns the parsed spec CreateEmbeddingResponse.
        """
        client = self._http_client()
        resp = await client.post(
            f"{self.base_url}/v1/embeddings", json=request, timeout=120.0
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"upstream /v1/embeddings returned HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            )
        return resp.json()

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=None)
        return self._client

    async def _scrape_rate_gauges(self) -> dict[str, float]:
        """Best-effort: read llama-server's Prometheus gauges. Never raises."""
        try:
            client = self._http_client()
            resp = await client.get(
                f"{self.base_url}/metrics", timeout=_METRICS_TIMEOUT
            )
            if resp.status_code != 200:
                return {}
            return parse_rate_gauges(resp.text)
        except Exception:  # noqa: BLE001 - rates are telemetry, not critical
            logger.debug("llama-cpp rate scrape failed", exc_info=True)
            return {}

    # ------------------------------------------------------------------
    # Streaming (SSE proxy)
    # ------------------------------------------------------------------
    _TERMINAL_TYPES = ("response.completed", "response.incomplete")

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
                    event = json.loads(data)
                except ValueError:
                    logger.debug("skipping unparseable SSE data: %r", data[:120])
                    continue
                if (
                    isinstance(event, dict)
                    and event.get("type") in self._TERMINAL_TYPES
                    and isinstance(event.get("response"), dict)
                    and isinstance(event["response"].get("usage"), dict)
                ):
                    # Enrich the terminal usage with the live rate gauges
                    # before the admin persists the turn (scrape happens
                    # while the request just finished, so the gauges
                    # reflect it).
                    rates = await self._scrape_rate_gauges()
                    inject_rates(event["response"]["usage"], rates)
                yield event
        # Exiting the `async with` (normal end OR GeneratorExit from an
        # early aclose) closes the upstream response, which cancels the
        # llama-server request. This is what makes the lifecycle's
        # release-on-upstream-close safe end to end.

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
        # Port model overhaul: the engine port is allocated per start, so clear
        # it here to keep the "None until start" invariant between boots.
        self.backend_port = None
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.to_thread(proc.wait, 30)
            except subprocess.TimeoutExpired:
                proc.kill()
                await asyncio.to_thread(proc.wait)
        # Terminate FIRST: once the process is gone the pipes hit EOF and the
        # blocked readline calls return. Cancelling readers before killing
        # the process would deadlock on the blocking read.
        for task in self._log_readers:
            task.cancel()
        if self._log_readers:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    asyncio.gather(*self._log_readers, return_exceptions=True),
                    timeout=10,
                )
        self._log_readers = []
        self._log_final_drain()

    def _log_final_drain(self) -> None:
        # After the process is gone, readers may have exited before the
        # final lines were flushed; the ring already holds what we saw.
        logger.debug("llama-server stopped; ring holds %d lines", len(self._log_ring))

    async def aclose(self) -> None:
        await self.stop()
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

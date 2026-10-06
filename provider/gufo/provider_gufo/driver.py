"""GufoBackend: BackendDriver over a managed ``gufo serve llm`` subprocess.

Ported from the legacy agent's gufo manager, re-shaped to the Phase 4
`BackendDriver` contract (lifecycle/state/slots live in provider_lib).

Provider-specific behaviors ported here:

- **Multi-model passthrough.** gufo serves multiple models on one
  instance; the request's ``model`` field is forwarded upstream as-is
  (llama-cpp's single-model server does not need this).
- **Native responses passthrough.** gufo speaks OpenResponses SSE
  natively; events pass through unmodified except for usage enrichment.
- **Rate-gauge scraping.** gufo reports instantaneous prompt/generation
  rates on ``GET /metrics`` (Prometheus gauges, not ``*_seconds_total``
  counters). Before the terminal ``response.completed``/``incomplete``
  event is yielded, the driver scrapes the gauges and merges them into
  the terminal usage's ``completion_tokens_details`` so the admin's
  `persist_turn` records real rates in `TokenUsageSample`.
- **Aux spec files by path.** mmproj / dflash_model / dspark_model /
  mtp_model live in ``backend_config.artifacts`` (Phase 12; the legacy
  ``options.*`` positions are still accepted) as artifact descriptors;
  they are resolved through
  `provider_lib.downloader.ensure_artifact` before spawn and rewritten
  to concrete local paths in the flattened options.
"""

import asyncio
import contextlib
import json
import logging
import os
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
from provider_lib.schema import load_schema, validate_backend_config

from provider_gufo.command import build_command, effective_capacity, flatten_args
from provider_gufo.log_ring import CursorLogRing
from provider_gufo.rates import inject_rates, parse_rate_gauges

logger = logging.getLogger("provider.gufo")

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]
LogCallback = Callable[[str, str], Awaitable[None]]

_LOG_BUFFER_LINES = 2000
_HEALTH_POLL_INTERVAL = 0.25
_STDERR_TAIL_LINES = 5
_METRICS_TIMEOUT = 5.0

# The gufo committed backend_config schema (Phase 12): the sectioned
# shape the admin registers, validates against, and renders in the UI.
# Shared with main.py (single load → registration + driver validation).
SCHEMA: dict[str, Any] = load_schema("provider_gufo")

# backend_config.artifacts keys that reference auxiliary GGUF/spec files
# (also accepted under the legacy `options.*` positions).
AUX_KEYS = ("mmproj", "dflash_model", "dspark_model", "mtp_model")


def _read_backend_port(cfg: dict[str, Any], settings: ProviderSettings) -> int:
    """Resolve the gufo listen port: `server.backend_port` (Phase 12),
    the legacy top-level `backend_port`, or PROVIDER_PORT + 1."""
    port = (cfg.get("server") or {}).get("backend_port")
    if port is None:
        port = cfg.get("backend_port")
    return int(port or (settings.PROVIDER_PORT + 1))


class GufoBackend(BackendDriver):
    """Owns one ``gufo serve llm`` subprocess for this provider instance."""

    def __init__(
        self,
        settings: ProviderSettings,
        backend_config: dict[str, Any],
        *,
        progress_cb: ProgressCallback | None = None,
        log_cb: LogCallback | None = None,
    ) -> None:
        self._settings = settings
        self._config = backend_config or {}
        self._progress_cb = progress_cb
        self._log_cb = log_cb
        self._binary = settings.GUFO_SERVER_PATH
        self.backend_port = _read_backend_port(self._config, settings)
        self._proc: subprocess.Popen[str] | None = None
        self._log_ring = CursorLogRing(_LOG_BUFFER_LINES)
        self._log_readers: list[asyncio.Task[None]] = []
        self._client: httpx.AsyncClient | None = None
        # Phase 9: local paths resolved during the last start, used by
        # storage.prune_unused as the "referenced artifact set".
        self.resolved_artifacts: list[str] = []
        # True per-instance identity for the `--cache-disk` subdirectory.
        # Set by apply_registration() from the registration response;
        # until then the driver falls back to MACHINE_UID (pre-registration
        # boots, e.g. tests). Registration always precedes backend.start
        # in the real flow, so the directory is genuinely per-instance.
        self.instance_id: str | None = None

    def set_instance_id(self, instance_id: str) -> None:
        """Record the admin-issued instance id (called at registration)."""
        self.instance_id = instance_id

    def apply_config(self, backend_config: dict[str, Any]) -> None:
        """Adopt a (possibly updated) sectioned backend_config before start.

        Schema violations are logged as warnings, not raised: the
        authoritative gates are the admin's registration schema check and
        definition CRUD validation (docs/ws-protocol.md §2).
        """
        self._config = backend_config or {}
        for err in validate_backend_config(SCHEMA, self._config):
            logger.warning("backend_config schema violation: %s", err)
        self.backend_port = _read_backend_port(self._config, self._settings)
        # Config changed: previously resolved artifacts are stale.
        self.resolved_artifacts = []

    def _options(self) -> dict[str, Any]:
        """Flat semantic options for argv building (Phase 12 sections +
        legacy `cfg["options"]`; see command.flatten_args)."""
        return flatten_args(self._config)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.backend_port}"

    @property
    def process(self) -> subprocess.Popen[str] | None:
        return self._proc

    @property
    def effective_capacity(self) -> int:
        """Gufo concurrency = preallocated GPU sessions (flattened sessions)."""
        return effective_capacity(self._options())

    # ------------------------------------------------------------------
    # Artifact resolution
    # ------------------------------------------------------------------
    def _artifact_descriptor(self, name: str) -> Any:
        """Read an artifact descriptor from the `artifacts` section
        (Phase 12), falling back to the legacy positions: top-level for
        `model`, `options.<name>` for the aux spec files."""
        artifacts = self._config.get("artifacts")
        if isinstance(artifacts, dict) and artifacts.get(name) is not None:
            return artifacts[name]
        if name == "model":
            return self._config.get(name)
        return (self._config.get("options") or {}).get(name)

    async def _resolve_artifact(self, descriptor: Any, kind: str) -> str:
        # The hfFile schema allows an optional `revision`; only a
        # non-empty string pins the HF revision (empty/absent = None).
        revision = descriptor.get("revision") if isinstance(descriptor, dict) else None
        if not isinstance(revision, str) or not revision.strip():
            revision = None
        return await ensure_artifact(
            descriptor,
            models_dir=self._settings.MODELS_DIR,
            progress_cb=self._progress_cb,
            revision=revision,
        )

    async def _resolve_main_model(self) -> str:
        model = self._artifact_descriptor("model")
        if not model:
            raise RuntimeError("backend_config missing 'model' artifact")
        if isinstance(model, str):
            p = Path(model)
            if not p.is_file():
                raise RuntimeError(f"local model file not found: {model}")
            return str(p)
        if isinstance(model, dict) and model.get("path"):
            p = Path(str(model["path"]))
            if not p.is_file():
                raise RuntimeError(f"local model file not found: {p}")
            return str(p)
        return await self._resolve_artifact(model, "model")

    async def _resolve_aux(self) -> dict[str, Any]:
        """Resolve aux descriptors to concrete paths in the flat options.

        A string value is passed through untouched (pre-placed file); a
        dict value is treated as an artifact descriptor and downloaded /
        resolved via the lib downloader, then replaced by its local path.
        Descriptors are read from `artifacts` (Phase 12) with the legacy
        `options.*` positions as fallback.
        """
        options = self._options()
        for key in AUX_KEYS:
            value = self._artifact_descriptor(key)
            if isinstance(value, dict):
                options[key] = await self._resolve_artifact(value, key)
            elif value is not None:
                options[key] = value
        return options

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Resolve model + aux files, spawn gufo, wait for /health."""
        if self._proc is not None and self._proc.poll() is None:
            return
        model_path = await self._resolve_main_model()
        options = await self._resolve_aux()
        self.resolved_artifacts = [model_path] + [
            str(options[key])
            for key in AUX_KEYS
            if isinstance(options.get(key), str) and options[key]
        ]
        cmd = build_command(
            options,
            model_path=model_path,
            port=self.backend_port,
            binary=self._binary,
            cache_dir=self._settings.CACHE_DIR,
            # Real registered instance id when available (set by
            # apply_registration); MACHINE_UID only pre-registration.
            instance_id=self.instance_id or self._settings.MACHINE_UID,
        )
        logger.info("starting gufo: %s", " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(os.environ),
                # New process group: gufo may spawn worker children;
                # stop kills the whole group (legacy semantics).
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeError(f"failed to spawn gufo ({self._binary}): {exc}") from exc

        self._proc = proc
        self._start_log_readers()
        try:
            await self._wait_for_health()
        except BaseException:
            await self.stop()
            raise

    async def _wait_for_health(self) -> None:
        timeout = float(self._settings.SERVER_START_HEALTH_TIMEOUT)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            proc = self._proc
            if proc is None:
                raise RuntimeError("gufo process disappeared")
            exit_code = proc.poll()
            if exit_code is not None:
                await self._drain_final_output()
                raise RuntimeError(self._early_exit_message(exit_code))
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    logger.info("gufo healthy on port %s", self.backend_port)
                    return
            except Exception:  # noqa: BLE001 - not up yet
                pass
            await asyncio.sleep(_HEALTH_POLL_INTERVAL)
        raise TimeoutError(f"gufo failed to become healthy within {timeout:.0f}s")

    def _early_exit_message(self, exit_code: int | None) -> str:
        stderr_lines = [
            e.line for e in self._log_ring.tail(100) if e.stream == "stderr"
        ]
        detail = "\n".join(stderr_lines[-_STDERR_TAIL_LINES:])
        message = f"gufo exited with code {exit_code}"
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
        # Join the live reader tasks first: two concurrent readers on the
        # same pipe (this drain + a blocked readline task) can split lines.
        for task in self._log_readers:
            task.cancel()
        if self._log_readers:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    asyncio.gather(*self._log_readers, return_exceptions=True),
                    timeout=10,
                )
        self._log_readers = []
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
        self._log_ring.append(stream_name, line)
        if self._log_cb is not None:
            with contextlib.suppress(RuntimeError):
                asyncio.get_running_loop().create_task(self._log_cb(stream_name, line))

    def get_logs(self, tail: int = 100) -> list[dict[str, str]]:
        """Recent stdout/stderr lines as {stream, line} dicts."""
        return [{"stream": e.stream, "line": e.line} for e in self._log_ring.tail(tail)]

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
            raise RuntimeError(f"gufo /v1/models returned HTTP {resp.status_code}")
        data = resp.json().get("data", [])
        return list(data)

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=None)
        return self._client

    async def _scrape_rate_gauges(self) -> dict[str, float]:
        """Best-effort: read gufo's Prometheus gauges. Never raises."""
        try:
            client = self._http_client()
            resp = await client.get(
                f"{self.base_url}/metrics", timeout=_METRICS_TIMEOUT
            )
            if resp.status_code != 200:
                return {}
            return parse_rate_gauges(resp.text)
        except Exception:  # noqa: BLE001 - rates are telemetry, not critical
            logger.debug("gufo rate scrape failed", exc_info=True)
            return {}

    # ------------------------------------------------------------------
    # Streaming (native OpenResponses SSE passthrough)
    # ------------------------------------------------------------------
    _TERMINAL_TYPES = ("response.completed", "response.incomplete")

    async def _stream_sse(
        self, path: str, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        body = dict(request)
        body["stream"] = True
        # NOTE: `model` is forwarded as-is. gufo is multi-model; the
        # served_model_name / alias mapping happened at registration, and
        # per-request model selection is a gufo feature we preserve.
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
                    event.get("type") in self._TERMINAL_TYPES
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
        # early aclose) closes the upstream response, cancelling the gufo
        # request — the lifecycle's release-on-upstream-close stays safe.

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
        if proc is not None and proc.poll() is None:
            # Kill the whole process group (gufo workers included).
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                await asyncio.to_thread(proc.wait, 30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                await asyncio.to_thread(proc.wait)
        for task in self._log_readers:
            task.cancel()
        if self._log_readers:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    asyncio.gather(*self._log_readers, return_exceptions=True),
                    timeout=10,
                )
        self._log_readers = []

    async def aclose(self) -> None:
        await self.stop()
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

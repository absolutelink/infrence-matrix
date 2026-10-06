"""HalogenFlashBackend: BackendDriver over a managed halogen-flash-server.

Ported from the legacy agent's halogen-flash manager, re-shaped to the
Phase 4 `BackendDriver` contract.

Halogen Flash specifics:

- **Native /v1/responses.** The backend itself speaks OpenResponses
  SSE; the driver mostly passes events through unchanged.
- **Usage normalization override.** The backend is NOT spec-compliant
  on usage (it reports chat-style counts, native timing keys, partial
  details, or nothing). Before the terminal event is yielded, the
  driver accumulates streamed characters / reasoning-delta tokens and
  runs `calculate_usage` (see `provider_halogen_flash.usage`) so the
  admin's `persist_turn` receives correct spec token counts. This is
  the canonical provider-override pattern.
- **Static ports for the disk-cache fingerprint.** The engine hashes
  its server variables — ports included — into its on-disk prompt-cache
  fingerprint; `resolve_ports()` pins a stable (api, engine) pair per
  machine (explicit in backend_config, else derived from MACHINE_UID in
  the dedicated 8200–8289 range).
- **NPU small-model pinning.** `npu.npu_models` (upstream ids; Phase 12
  section, legacy `options.npu_models` still accepted via
  `env.flatten_options`) is exported as HALOGEN_NPU_MODELS only when
  the host NPU probe passes; without an NPU the env is omitted
  gracefully (the ids are dropped with a log line, never an error).
"""

import asyncio
import contextlib
import functools
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
from provider_lib.schema import load_schema, validate_backend_config

from provider_halogen_flash.env import (
    build_argv,
    build_env,
    effective_capacity,
    flatten_options,
    resolve_ports,
)
from provider_halogen_flash.log_ring import CursorLogRing
from provider_halogen_flash.npu import (
    load_npu_pins,
    probe_npu,
    resolve_npu_download_set,
)
from provider_halogen_flash.usage import calculate_usage

logger = logging.getLogger("provider.halogen_flash")

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]
LogCallback = Callable[[str, str], Awaitable[None]]

_LOG_BUFFER_LINES = 2000
_HEALTH_POLL_INTERVAL = 0.25
_STDERR_TAIL_LINES = 5

_TERMINAL_TYPES = ("response.completed", "response.incomplete")

# The halogen-flash committed backend_config schema (Phase 12): the
# sectioned shape the admin registers, validates against, and renders in
# the UI. Shared with main.py (single load → registration + driver
# validation).
SCHEMA: dict[str, Any] = load_schema("provider_halogen_flash")


class HalogenFlashBackend(BackendDriver):
    """Owns one halogen-flash (api+engine) process for this instance."""

    def __init__(
        self,
        settings: ProviderSettings,
        backend_config: dict[str, Any],
        *,
        progress_cb: ProgressCallback | None = None,
        log_cb: LogCallback | None = None,
        npu_probe: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._settings = settings
        self._config = backend_config or {}
        self._progress_cb = progress_cb
        self._log_cb = log_cb
        self._binary = settings.HALOGEN_FLASH_SERVER_PATH
        self._npu_probe = npu_probe
        self._ports = resolve_ports(self._config, settings.MACHINE_UID)
        self._proc: subprocess.Popen[str] | None = None
        self._log_ring = CursorLogRing(_LOG_BUFFER_LINES)
        self._log_reader: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None
        # Phase 9: local paths resolved during the last start (model,
        # tokenizer, prepared NPU pins), used by storage.prune_unused.
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
        self._ports = resolve_ports(self._config, self._settings.MACHINE_UID)
        # Config changed: previously resolved artifacts are stale.
        self.resolved_artifacts = []

    def _options(self) -> dict[str, Any]:
        """Flat semantic options for env building (Phase 12 sections +
        legacy `cfg["options"]`; see env.flatten_options)."""
        return flatten_options(self._config)

    @property
    def api_port(self) -> int:
        return self._ports[0]

    @property
    def engine_port(self) -> int:
        return self._ports[1]

    @property
    def backend_port(self) -> int:
        return self._ports[0]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    @property
    def process(self) -> subprocess.Popen[str] | None:
        return self._proc

    @property
    def effective_capacity(self) -> int:
        return effective_capacity(self._options())

    # ------------------------------------------------------------------
    # NPU gating
    # ------------------------------------------------------------------
    async def _npu_models_for_env(self) -> list[str] | None:
        """Return the requested NPU model ids if the host can run them.

        Degrades gracefully: no NPU detected (or no pins file) -> None,
        with the reason logged. The request never fails on a machine
        without an NPU. The probe does blocking I/O (device checks,
        subprocess) so it runs in a worker thread.
        """
        opts = self._options()
        requested = opts.get("npu_models")
        if not requested:
            return None
        probe = self._npu_probe
        if probe is None:
            probe = functools.partial(
                probe_npu,
                device_path=self._settings.NPU_DEVICE_PATH,
                xrt_lib_dir=self._settings.NPU_XRT_LIB_DIR,
                binary_path=self._settings.NPU_BINARY_PATH,
            )
        try:
            result = await asyncio.to_thread(probe)
        except Exception:  # noqa: BLE001 - probe must never break start
            logger.warning("NPU probe crashed; omitting NPU models", exc_info=True)
            return None
        if not result.get("available"):
            logger.info(
                "NPU not available on this host (%s); omitting npu_models=%s",
                "; ".join(result.get("reasons") or ["unknown"]),
                list(requested),
            )
            return None
        return list(requested)

    # ------------------------------------------------------------------
    # NPU pre-download (Phase 9; closes the Phase 8 deferral)
    # ------------------------------------------------------------------
    async def _prepare_npu_models(self, npu_models: list[str] | None) -> list[str]:
        """Download the pinned NPU small-model files for the gated ids.

        Reads the image's pins file (``NPU_PINS_FILE``), plans the
        download set via ``resolve_npu_download_set`` (shared
        ``devices/`` programs included), and ``ensure_artifact``s each
        file into ``MODELS_DIR/npu/<id>/`` before spawn (progress
        events flow through the driver's progress callback).

        Graceful by design: a missing/unparsable pins file, or any
        planning/download error, logs and yields an empty list — the
        requested pins are omitted and start proceeds without NPU
        models. Never fails the boot.
        """
        if not npu_models:
            return []
        pins_file = self._settings.NPU_PINS_FILE
        try:
            records = await asyncio.to_thread(load_npu_pins, pins_file)
            plan = resolve_npu_download_set(records, list(npu_models))
        except (OSError, ValueError) as exc:
            logger.warning(
                "NPU pins planning failed (%s: %s); omitting npu pins",
                pins_file,
                exc,
            )
            return []
        prepared: list[str] = []
        for dir_id, (repo, revision, files) in sorted(plan.items()):
            for path, _size, _sha in files:
                try:
                    local = await ensure_artifact(
                        {"source": "hf", "repo": repo, "file": path},
                        models_dir=self._settings.MODELS_DIR,
                        progress_cb=self._progress_cb,
                        revision=revision,
                        local_subdir=f"npu/{dir_id}",
                    )
                except Exception as exc:  # noqa: BLE001 - never fail start
                    logger.warning(
                        "NPU artifact download failed (%s/%s): %s; omitting pins",
                        repo,
                        path,
                        exc,
                    )
                    return []
                prepared.append(local)
        logger.info(
            "NPU artifacts prepared: %d file(s) for %s",
            len(prepared),
            ", ".join(sorted(plan)),
        )
        return prepared

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
        """Resolve a halogen-flash artifact to a local path.

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Resolve artifacts, probe NPU, spawn the server, wait for health."""
        if self._proc is not None and self._proc.poll() is None:
            return
        # The argv wraps the entrypoint in stdbuf, so a missing binary
        # would otherwise surface as a confusing stdbuf exit; check first.
        if not shutil.which(self._binary) and not Path(self._binary).exists():
            raise RuntimeError(f"halogen-flash entrypoint not found: {self._binary}")
        checkpoint = await self._resolve_artifact(
            self._artifact_descriptor("model"), "model"
        )
        tokenizer = await self._resolve_artifact(
            self._artifact_descriptor("tokenizer"), "tokenizer"
        )
        self.resolved_artifacts = [checkpoint, tokenizer]
        # Flat semantic options (Phase 12 sections + legacy options blob).
        # artifacts.vision_tower / artifacts.mtp_head are env-mapped
        # (HALOGEN_VISION_TOWER / HALOGEN_MTP_HEAD): descriptors and
        # local paths resolve to their local path here; `true` means
        # "look beside the checkpoint" (emit 1); `false` is an explicit
        # off-switch for the vision tower (emit 0). An absent value emits
        # nothing (engine default).
        options = self._options()
        for name in ("vision_tower", "mtp_head"):
            descriptor = self._artifact_descriptor(name)
            if descriptor is None:
                continue
            if descriptor is False:
                # Only vision_tower may be a boolean per schema; `false`
                # exports the explicit disable (HALOGEN_VISION_TOWER=0).
                if name == "vision_tower":
                    options[name] = 0
                continue
            if descriptor is True:
                options[name] = 1
            else:
                local = await self._resolve_artifact(descriptor, name)
                options[name] = local
                self.resolved_artifacts.append(local)
        api_port, engine_port = self._ports
        cache_dir = Path(self._settings.CACHE_DIR) / "halogen-flash"
        npu_models = await self._npu_models_for_env()
        if npu_models:
            # Phase 9: pre-download the pinned NPU small-model files so
            # the engine finds them locally at spawn. If the pins cannot
            # be prepared, omit the pins entirely (graceful degradation).
            prepared = await self._prepare_npu_models(npu_models)
            if prepared:
                self.resolved_artifacts.extend(prepared)
            else:
                npu_models = None
        env = build_env(
            options,
            api_port=api_port,
            engine_port=engine_port,
            checkpoint_path=checkpoint,
            tokenizer_path=tokenizer,
            cache_dir=cache_dir,
            npu_models=npu_models,
        )
        cmd = build_argv(self._binary)
        logger.info(
            "starting halogen-flash: %s (api=%s engine=%s)",
            " ".join(cmd),
            api_port,
            engine_port,
        )
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,
                bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(
                f"failed to spawn halogen-flash ({self._binary}): {exc}"
            ) from exc

        self._proc = proc
        self._start_log_reader()
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
                raise RuntimeError("halogen-flash process disappeared")
            exit_code = proc.poll()
            if exit_code is not None:
                await self._drain_final_output()
                raise RuntimeError(self._early_exit_message(exit_code))
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(f"{self.base_url}/health")
                if resp.status_code == 200:
                    logger.info("halogen-flash API healthy on port %s", self.api_port)
                    return
            except Exception:  # noqa: BLE001 - not up yet
                pass
            await asyncio.sleep(_HEALTH_POLL_INTERVAL)
        raise TimeoutError(
            f"halogen-flash failed to become healthy within {timeout:.0f}s"
        )

    def _early_exit_message(self, exit_code: int | None) -> str:
        lines = [e.line for e in self._log_ring.tail(100)]
        detail = "\n".join(lines[-_STDERR_TAIL_LINES:])
        message = f"halogen-flash exited with code {exit_code}"
        if detail:
            message = f"{message}: {detail}"
        return message

    # ------------------------------------------------------------------
    # Log capture
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
            logger.exception("halogen-flash log reader failed")

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

    def _append_log(self, stream_name: str, line: str) -> None:
        self._log_ring.append(stream_name, line)
        if self._log_cb is not None:
            with contextlib.suppress(RuntimeError):
                asyncio.get_running_loop().create_task(self._log_cb(stream_name, line))

    def get_logs(self, tail: int = 100) -> list[dict[str, str]]:
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
            raise RuntimeError(
                f"halogen-flash /v1/models returned HTTP {resp.status_code}"
            )
        data = resp.json().get("data", [])
        return list(data)

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=None)
        return self._client

    # ------------------------------------------------------------------
    # Streaming: native OpenResponses passthrough + usage override
    # ------------------------------------------------------------------
    def stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        return self._stream_native_responses(request)

    async def _stream_native_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        body = dict(request)
        body["stream"] = True
        # Accumulators for the usage override: what the stream itself
        # produced, used only when the backend under-reports.
        text_chars = 0
        reasoning_tokens = 0
        client = self._http_client()
        async with client.stream(
            "POST", f"{self.base_url}/v1/responses", json=body
        ) as resp:
            if resp.status_code != 200:
                detail = await resp.aread()
                raise RuntimeError(
                    f"upstream /v1/responses returned HTTP {resp.status_code}: "
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
                etype = event.get("type")
                if etype == "response.output_text.delta":
                    text_chars += len(event.get("delta") or "")
                elif etype in (
                    "response.reasoning_text.delta",
                    "response.reasoning_summary_text.delta",
                ):
                    # Rough token estimate from reasoning deltas: ~4 chars.
                    reasoning_tokens += len(event.get("delta") or "") // 4
                if etype in _TERMINAL_TYPES and isinstance(event.get("response"), dict):
                    response = event["response"]
                    raw_usage = response.get("usage")
                    if not isinstance(raw_usage, dict):
                        raw_usage = response
                    elif isinstance(response.get("timings"), dict):
                        # Some Flash builds report timing rates beside, rather
                        # than inside, the usage object.
                        raw_usage = {**raw_usage, "timings": response["timings"]}
                    event["response"]["usage"] = calculate_usage(
                        raw_usage,
                        fallback_chars=text_chars,
                        reasoning_tokens=reasoning_tokens,
                    )
                yield event

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    async def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
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

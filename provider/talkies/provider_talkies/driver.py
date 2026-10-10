"""TalkiesBackend: BackendDriver over a managed talkies speech-server subprocess.

One talkies process per backend definition (per-definition lifecycles are
the locked Phase 24 shape: talkies reads its registry/enabled set only
at import, so a definition's slug set is process-scoped). Phase 25
multi-model: that ONE process now serves the definition's whole set of
served models (each client-facing name maps to a talkies registry slug);
a legacy single-slug definition serves exactly one. The driver:

- **Prefetches** every served model's snapshot into
  ``$TALKIES_DATA_DIR/models/<slug>`` before spawn (image-entrypoint
  parity — the server itself runs ``HF_HUB_OFFLINE=1``).
- **Spawns** ``<TALKIES_PYTHON> -m uvicorn talkies.server:app --host
  127.0.0.1 --port <OS-assigned>`` in its own session with the full
  ``TALKIES_*`` env (all-slug ENABLED + PRELOAD + per-slug concurrency +
  TTL 0 pinning).
- **Health** = ``/healthz`` 200 AND ``/api/ps`` lists EVERY served slug
  (pin semantics extended to the set: running <=> all resident).
- **Proxies** the Phase 24 audio surface: ``speech`` (served name -> slug
  rewrite, streamed bytes + relayed headers), ``transcribe`` (multipart
  relay), ``voices`` / ``voice_put`` / ``voice_delete`` (catalog +
  custom-voices enrollment on the shared data dir), and
  ``transcribe_stream`` (bidirectional WS bridge to the engine).

There is no LLM surface: ``stream_responses`` / ``stream_chat_completions``
keep the ABC defaults (NotImplementedError -> 501).

**Speech error relay (documented choice):** the S1 ``SpeechStream``
contract fixes response headers synchronously, before any upstream byte
is observed, and the route commits HTTP 200 to the client before the
pump ever runs the generator — so an upstream non-2xx CANNOT be remapped
to a real HTTP status on the speech path. The driver raises
``fastapi.HTTPException`` with the upstream status + detail from the
chunk generator anyway: it aborts the stream (the client sees a
truncated/empty body rather than an error body silently masquerading as
audio bytes) and the lifecycle's pump releases the slot in its
``finally`` exactly like any upstream close. A faithful status relay needs
an async speech seam (or a status field on ``SpeechStream``) in
provider_lib — deliberately out of S2 scope.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import socket
import subprocess
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import websockets
from fastapi import HTTPException, WebSocket
from provider_lib.backend import BackendDriver, SpeechStream
from provider_lib.config import ProviderSettings
from provider_lib.log_ring import CursorLogRing
from provider_lib.models import ModelSpec
from provider_lib.schema import load_schema, validate_backend_config
from provider_lib.wire import BackendStatusValue

from provider_talkies import env as talkies_env
from provider_talkies.prefetch import load_registry, prefetch_model

logger = logging.getLogger("provider.talkies")

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]
StatusCallback = Callable[[str, str | None], Awaitable[None]]

_LOG_BUFFER_LINES = 2000
_HEALTH_POLL_INTERVAL = 0.25
_STDERR_TAIL_LINES = 5
# Seconds between `initializing` heartbeats during the (potentially long)
# prefetch + first-load wait.
_BOOT_HEARTBEAT_SECONDS = 15.0
_HEARTBEAT_REASON_CHARS = 160

# The talkies committed backend_config schema (Phase 24): the sectioned
# shape the admin registers, validates against, and renders in the UI.
# Shared with main.py (single load -> registration + driver validation).
SCHEMA: dict[str, Any] = load_schema("provider_talkies")

# Saved-voice names must be single filesystem-safe path segments (the agent
# route pre-validates; the driver re-validates defensively — defense in
# depth, same rule as provider_lib.app_factory).
_VOICE_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")


def _pick_free_port() -> int:
    """Ask the OS for a free loopback port (bind ``127.0.0.1:0``, read back,
    release).

    Port model: the talkies engine binds a private local port inside the
    container that the admin never learns or dials; the agent's single
    ``/v1`` surface routes to this backend by model. A stolen port makes
    uvicorn fail to bind and exit early, surfacing as a boot error the
    scheduler re-drives on the next ``backend.start``.
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class TalkiesBackend(BackendDriver):
    """Owns one talkies speech-server subprocess for this provider instance."""

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
        # Port model: the engine's private loopback port is allocated at
        # start (OS-assigned), ``None`` until then.
        self.backend_port: int | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._log_ring = CursorLogRing(_LOG_BUFFER_LINES)
        self._log_readers: list[asyncio.Task[None]] = []
        self._client: httpx.AsyncClient | None = None
        # Resolved at start(): the definition's primary slug (legacy single
        # slug, or the first enabled served slug), the bearer the engine
        # enforces, and the data root it reads.
        self.slug: str | None = None
        self.alias: str | None = None
        self._auth_token: str = ""
        self._data_dir: Path | None = None
        # Phase 25 multi-model: the enabled served-model list adopted via
        # ``set_models`` plus the name -> slug map derived from it. Empty
        # until ``set_models`` runs (legacy path / direct construction), in
        # which case every accessor falls back to the single ``model.slug``
        # resolved from ``self._config`` — byte-identical to Phase 24.
        self._models: list[ModelSpec] = []
        self._name_to_slug: dict[str, str] = {}
        # The slug set the CURRENTLY-RUNNING process was actually spawned with
        # (recorded in start(), cleared in stop()). health() pins THIS, not the
        # live ``self.slugs``: talkies loads its registry only at import, so a
        # served name added live via ``set_models`` whose slug was never spawned
        # is not resident and must NOT flip an otherwise-healthy backend to
        # unhealthy. A genuinely new slug reaches the engine only via the
        # config.update RESTART path (which re-runs start with the new set).
        self._spawned_slugs: list[str] = []
        # Phase 9: local paths recorded during the last start (prefetched
        # model dir + custom-voices sidecar store) for storage.prune_unused.
        self.resolved_artifacts: list[str] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def attach_status_callback(self, emit: StatusCallback) -> None:
        """Bind the lifecycle's out-of-band status reporter (long-init
        heartbeats only; state transitions stay the lifecycle's)."""
        self._status_cb = emit

    def set_alias(self, alias: str | None) -> None:
        """Record the definition alias this backend serves (list_models id)."""
        self.alias = alias

    def set_models(self, models: list[ModelSpec]) -> None:
        """Adopt the served-model list (Phase 25 multi-model).

        Stores the ENABLED specs and rebuilds the ``name -> slug`` map (slug
        read from each spec's per-model ``backend_config``, falling back to a
        legacy sectioned ``model.slug``). When the list is empty the driver
        behaves exactly as a single-slug definition (every accessor falls back
        to ``resolve_slug(self._config)``). Called at handle build and on every
        assignment / config-update refresh **before** any restart.

        A live (no-restart) add is only fully served when the new name maps to
        a slug the running process ALREADY loaded — talkies reads its registry
        at import only, so a genuinely NEW slug cannot be hot-loaded. Adding a
        new-slug served name must therefore go through the ``provider.config
        .update`` RESTART path (which re-runs :meth:`start` with the new set);
        the health pin deliberately tracks the spawned slug set (:attr:`
        _spawned_slugs`), not this live list, so a not-yet-spawned slug never
        flips a healthy backend unhealthy in the interim.
        """
        self._models = [m for m in models if m.enabled]
        self._name_to_slug = {}
        for spec in self._models:
            slug = talkies_env.model_slug(spec.backend_config)
            if slug:
                self._name_to_slug[spec.name] = slug

    @property
    def slugs(self) -> list[str]:
        """Ordered-unique slugs across the enabled served models (legacy: the
        single ``model.slug``)."""
        return talkies_env.resolve_slugs(self._config, self._models)

    def _slug_for_name(self, name: Any) -> str:
        """Map an inbound request ``model`` (a served NAME) to its engine slug.

        The registry already 404s unroutable names before the driver sees them,
        so a map miss is defensive only: with exactly one slug (legacy / single
        served model) it is returned regardless of name; with several an unknown
        name is a clean 404.
        """
        if isinstance(name, str):
            slug = self._name_to_slug.get(name)
            if slug:
                return slug
        slugs = self.slugs
        if len(slugs) == 1:
            return slugs[0]
        raise HTTPException(status_code=404, detail=f"unknown model {name!r}")

    def _defaults_for_name(self, name: Any) -> dict[str, Any]:
        """Effective request defaults for a served name: the shared config's
        ``defaults`` overridden by that model's per-model ``defaults``."""
        shared = talkies_env.resolve_defaults(self._config)
        if isinstance(name, str):
            for spec in self._models:
                if spec.name == name:
                    per = talkies_env.resolve_defaults(spec.backend_config or {})
                    return {**shared, **per}
        return shared

    def apply_config(self, backend_config: dict[str, Any]) -> None:
        """Adopt a (possibly updated) sectioned backend_config before start.

        The config-update flow always stops the backend first (drain gate),
        so nothing live is affected here — the next start re-resolves slug,
        registry, data dir, and auth token from the new config. Schema
        violations are logged as warnings, not raised: the authoritative
        gates are the admin's registration schema check and definition CRUD
        validation (docs/ws-protocol.md §2).
        """
        self._config = backend_config or {}
        for err in validate_backend_config(SCHEMA, self._config):
            logger.warning("backend_config schema violation: %s", err)
        # The engine port is allocated at start, never read from config.
        # Config changed: previously resolved artifacts are stale.
        self.slug = None
        self._auth_token = ""
        self._data_dir = None
        self.resolved_artifacts = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.backend_port}"

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.backend_port}"

    @property
    def process(self) -> subprocess.Popen[str] | None:
        return self._proc

    @property
    def effective_capacity(self) -> int:
        """Engine admission = per-model concurrency (TALKIES_MODEL_CONCURRENCY).

        Multi-model: the maximum across served slugs (the widest single-model
        pool the shared process admits). Legacy: the single slug's value.
        """
        concurrency_map = talkies_env.resolve_model_concurrency_map(
            self._config, self._models
        )
        return max(concurrency_map.values(), default=1)

    def _require_slug(self) -> str:
        if not self.slug:
            self.slug = talkies_env.resolve_slug(self._config)
        return self.slug

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._auth_token}"}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Resolve + prefetch every served slug, spawn uvicorn, await readiness.

        Phase 25 multi-model: one process serves the whole slug set — every
        enabled slug is registry-validated and prefetched BEFORE spawn (the
        server runs ``HF_HUB_OFFLINE=1`` and reads the registry only at
        import), then exported together as ``TALKIES_ENABLED_MODELS`` /
        ``TALKIES_PRELOAD``. A legacy single-slug definition resolves exactly
        one slug, so the loop is a one-iteration pass-through.
        """
        if self._proc is not None and self._proc.poll() is None:
            return
        slugs = self.slugs
        self.slug = slugs[0]  # primary slug (back-compat / legacy single)
        data_dir = talkies_env.resolve_data_dir(self._settings)
        self._data_dir = data_dir
        self._auth_token = talkies_env.resolve_auth_token(self._config)
        revisions = talkies_env.resolve_revisions(self._config, self._models)
        label = ", ".join(slugs)

        # Registry validation + prefetch (initializing phase, BEFORE spawn:
        # the server runs HF_HUB_OFFLINE=1 and only reads the registry at
        # import). Every slug must resolve — one unknown slug fails the boot.
        await self._heartbeat(f"talkies initializing: resolving {label}")
        registry = await asyncio.to_thread(
            load_registry, Path(self._settings.TALKIES_MODELS_FILE)
        )
        for slug in slugs:
            entry = registry.get(slug)
            if not isinstance(entry, dict):
                known = ", ".join(sorted(registry)) or "<empty>"
                raise RuntimeError(
                    f"talkies slug {slug!r} not in registry "
                    f"{self._settings.TALKIES_MODELS_FILE} (known: {known})"
                )
        model_dirs: list[Path] = []
        for slug in slugs:
            model_dirs.append(
                await prefetch_model(
                    slug,
                    registry[slug],
                    data_dir,
                    revision=revisions.get(slug),
                    progress_cb=self._progress_cb,
                )
            )
        for sub in ("models", "files", "custom-voices", "hf"):
            (data_dir / sub).mkdir(parents=True, exist_ok=True)
        self.resolved_artifacts = [str(md) for md in model_dirs] + [
            str(data_dir / "custom-voices")
        ]

        self.backend_port = _pick_free_port()
        cmd = talkies_env.build_command(
            self._settings.TALKIES_PYTHON, self.backend_port
        )
        spawn_env = talkies_env.build_env(
            self._config,
            settings=self._settings,
            data_dir=data_dir,
            auth_token=self._auth_token,
            models=self._models,
        )
        logger.info("starting talkies (%s): %s", label, " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=spawn_env,
                # New session: uvicorn may spawn workers; stop() kills the
                # whole process group.
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeError(
                f"failed to spawn talkies ({self._settings.TALKIES_PYTHON}): {exc}"
            ) from exc

        self._proc = proc
        # Record the slug set this process was actually spawned with: the
        # health pin uses THIS (not the live self.slugs), so a served name
        # added live via set_models whose slug was never loaded here cannot
        # spuriously fail health. A new-slug change needs the restart path.
        self._spawned_slugs = list(slugs)
        self._start_log_readers()
        try:
            await self._wait_for_health()
        except BaseException:
            await self.stop()
            raise

    async def _wait_for_health(self) -> None:
        """Wait out the engine's start: preload of the pinned model.

        Budget is ``TALKIES_BOOT_TIMEOUT`` (default 600s): a cold first
        load (weights already prefetched, but device init + model load of a
        multi-GB checkpoint is slow) must not fail the boot. `initializing`
        heartbeats carry the engine's newest log line so the admin shows
        progress instead of a hang; a process that dies is detected
        immediately from its exit code.
        """
        timeout = float(self._settings.TALKIES_BOOT_TIMEOUT)
        deadline = time.monotonic() + timeout
        started = time.monotonic()
        next_beat = started + _BOOT_HEARTBEAT_SECONDS
        while time.monotonic() < deadline:
            proc = self._proc
            if proc is None:
                raise RuntimeError("talkies process disappeared")
            exit_code = proc.poll()
            if exit_code is not None:
                await self._drain_final_output()
                raise RuntimeError(self._early_exit_message(exit_code))
            if await self.health():
                logger.info("talkies healthy on port %s", self.backend_port)
                return
            now = time.monotonic()
            if self._status_cb is not None and now >= next_beat:
                next_beat = now + _BOOT_HEARTBEAT_SECONDS
                await self._heartbeat(
                    f"talkies initializing ({now - started:.0f}s elapsed)"
                )
            await asyncio.sleep(_HEALTH_POLL_INTERVAL)
        raise TimeoutError(
            f"talkies failed to become healthy within {timeout:.0f}s "
            f"(healthz + /api/ps pin of {', '.join(self._spawned_slugs or self.slugs)!r})"
        )

    async def _heartbeat(self, reason: str) -> None:
        """Report `initializing` (best effort; progress never breaks a boot)."""
        if self._status_cb is None:
            return
        line = ""
        for entry in reversed(self._log_ring.tail(20)):
            text = entry.text.strip()
            if text:
                line = text[:_HEARTBEAT_REASON_CHARS]
                break
        if line:
            reason = f"{reason}: {line}"
        try:
            await self._status_cb(BackendStatusValue.INITIALIZING, reason)
        except Exception:  # noqa: BLE001
            logger.debug("boot heartbeat failed", exc_info=True)

    def _early_exit_message(self, exit_code: int | None) -> str:
        stderr_lines = [
            e.text for e in self._log_ring.tail(100) if e.stream == "stderr"
        ]
        detail = "\n".join(stderr_lines[-_STDERR_TAIL_LINES:])
        message = f"talkies exited with code {exit_code}"
        if detail:
            message = f"{message}: {detail}"
        return message

    # ------------------------------------------------------------------
    # Log capture (Phase 13 contract: driver -> ring)
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

    @property
    def log_ring(self) -> CursorLogRing:
        """Shared Phase 13 ring (provider_lib.log_stream reads this)."""
        return self._log_ring

    def _append_log(self, stream_name: str, line: str) -> None:
        self._log_ring.append(stream_name, line)

    def get_logs(self, tail: int = 100) -> list[dict[str, str]]:
        """Recent stdout/stderr lines as {stream, text} dicts."""
        return [{"stream": e.stream, "text": e.text} for e in self._log_ring.tail(tail)]

    # ------------------------------------------------------------------
    # Health / models
    # ------------------------------------------------------------------
    async def health(self) -> bool:
        """Ready <=> process alive AND /healthz 200 AND /api/ps pins every SPAWNED slug.

        ``/api/ps`` is driver-internal (readiness), never client-facing: with
        ``TALKIES_MODEL_TTL=0`` + PRELOAD every slug the process was spawned
        with must appear the moment its load finishes, so a green health check
        means all those weights are resident and the admin's VRAM ledger stays
        exact. The pin is on :attr:`_spawned_slugs` (what THIS process loaded),
        NOT the live ``self.slugs`` — a served name added live via
        ``set_models`` whose slug was never spawned here is not resident and
        must not flip an otherwise-healthy backend unhealthy (that change
        reaches the engine via the config.update restart path). Legacy
        single-slug definitions pin exactly one slug (Phase 24 semantics).
        """
        proc = self._proc
        if proc is None or proc.poll() is not None or self.backend_port is None:
            return False
        try:
            client = self._http_client()
            resp = await client.get(f"{self.base_url}/healthz", timeout=5.0)
            if resp.status_code != 200:
                return False
            required = set(self._spawned_slugs) or set(self.slugs)
            ps = await client.get(
                f"{self.base_url}/api/ps", headers=self._auth_headers(), timeout=5.0
            )
            if ps.status_code != 200:
                return False
            loaded = {
                m.get("id")
                for m in (ps.json().get("models") or [])
                if isinstance(m, dict)
            }
            return required <= loaded
        except Exception:  # noqa: BLE001 - not up yet
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """The OpenAI model objects for this definition (one per served name).

        Multi-model: one entry per enabled served name with its own modality.
        Legacy (no served list): the single alias/slug entry (Phase 24).
        """
        if self._models:
            return [
                {
                    "id": spec.name,
                    "object": "model",
                    "owned_by": "talkies",
                    "modality": spec.modality,
                }
                for spec in self._models
            ]
        slug = self._require_slug()
        return [
            {
                "id": self.alias or slug,
                "object": "model",
                "owned_by": "talkies",
                "modality": self.modality,
            }
        ]

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=None)
        return self._client

    # ------------------------------------------------------------------
    # No LLM surface: talkies is a speech engine. The ABC defaults for
    # stream_chat_completions / embeddings already raise NotImplementedError
    # (-> 501); stream_responses is abstract and must be spelled out.
    # ------------------------------------------------------------------
    def stream_responses(
        self, request: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError("talkies has no OpenResponses surface")

    # ------------------------------------------------------------------
    # Phase 24: speech (TTS)
    # ------------------------------------------------------------------
    def _apply_defaults(
        self,
        body: dict[str, Any],
        defaults: dict[str, Any],
        allowed: set[str] | None = None,
    ) -> dict[str, Any]:
        """Fill recorded ``defaults`` into a request that omits them.

        ``defaults`` is the effective default map for the routed served name
        (shared config merged with the per-model ``defaults``). ``allowed``
        restricts the merge to keys meaningful for the endpoint (e.g. a speech
        ``response_format="pcm"`` default must never leak into a transcription
        form, where it is an invalid value).
        """
        merged = dict(body)
        for key, value in defaults.items():
            if allowed is not None and key not in allowed:
                continue
            if merged.get(key) in (None, ""):
                merged[key] = value
        return merged

    def speech(self, request: dict[str, Any]) -> SpeechStream:
        """Proxy POST /v1/audio/speech with the name -> slug rewrite.

        Response headers are derived from the (default-merged) request:
        talkies' content-type is a pure function of ``response_format``,
        and PCM is always 24 kHz mono, so the SpeechStream contract's
        synchronous header requirement is met without observing the
        upstream response. Upstream non-2xx raises HTTPException from the
        generator — the stream aborts and the slot releases, but the
        already-committed 200 cannot be remapped (module docstring).
        """
        name = request.get("model")
        body = self._apply_defaults(dict(request), self._defaults_for_name(name))
        body["model"] = self._slug_for_name(name)
        fmt = str(body.get("response_format") or "mp3").lower()
        headers = {
            "Content-Type": talkies_env.CONTENT_TYPES.get(
                fmt, "application/octet-stream"
            )
        }
        if fmt == "pcm":
            headers["X-Sample-Rate"] = talkies_env.TALKIES_PCM_SAMPLE_RATE
        return SpeechStream(headers=headers, chunks=self._speech_chunks(body))

    async def _speech_chunks(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        client = self._http_client()
        async with client.stream(
            "POST",
            f"{self.base_url}/v1/audio/speech",
            json=body,
            headers=self._auth_headers(),
        ) as resp:
            if resp.status_code != 200:
                detail = await resp.aread()
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=detail.decode(errors="replace")[:500],
                )
            async for chunk in resp.aiter_raw():
                yield chunk
        # Exiting the `async with` (normal end OR GeneratorExit from the
        # pump's early close) closes the upstream response — the
        # release-on-upstream-close invariant stays driven by the engine's
        # stream, never the downstream client.

    # ------------------------------------------------------------------
    # Phase 24: transcription (ASR, HTTP multipart relay)
    # ------------------------------------------------------------------
    async def transcribe(
        self,
        form: dict[str, str],
        files: list[tuple[str, str, bytes, str]],
    ) -> tuple[int, dict[str, str], bytes]:
        """Relay a multipart transcription with the alias -> slug rewrite.

        Upstream status/headers/body pass through untouched (the route
        strips hop-by-hop headers), so every response_format (json / text
        / verbose_json / srt / vtt) and every engine error shape relays
        faithfully. Only ASR-meaningful recorded defaults (``language``)
        are merged in — speech-only knobs (voice / response_format=pcm /
        speed / instructions) must never leak into the form. The relay is
        bounded by ``TALKIES_BOOT_TIMEOUT`` (a long file transcribes for
        minutes; an unbounded hang would pin the inference slot forever).
        """
        data = {k: v for k, v in form.items() if k != "model"}
        name = form.get("model")
        merged = self._apply_defaults(
            data, self._defaults_for_name(name), allowed={"language"}
        )
        merged["model"] = self._slug_for_name(name)
        client = self._http_client()
        resp = await client.post(
            f"{self.base_url}/v1/audio/transcriptions",
            data=merged,
            files=[
                (field, (filename, content, ctype))
                for field, filename, content, ctype in files
            ],
            headers=self._auth_headers(),
            timeout=float(self._settings.TALKIES_BOOT_TIMEOUT),
        )
        return resp.status_code, dict(resp.headers), resp.content

    # ------------------------------------------------------------------
    # Phase 24: voice catalog + enrollment
    # ------------------------------------------------------------------
    def _voice_slug(self, model: str | None) -> str:
        """Resolve a voice op's served NAME to its engine slug.

        ``None`` (legacy single-slug / a direct call with no name) uses the
        primary slug; otherwise the name -> slug map applies (the route already
        gated the name as routable to this handle).
        """
        if model is None:
            return self.slugs[0]
        return self._slug_for_name(model)

    async def voices(self, model: str | None = None) -> dict[str, Any]:
        """GET /v1/audio/voices passthrough, filtered to the routed model.

        ``model`` is the served NAME the route resolved this request to; it is
        mapped to its engine slug (name -> slug) before forwarding. Upstream
        talkies ignores its own ``model`` query filter and returns the
        catalogs of ALL loaded models (multi-model processes), so the response
        is filtered here to the routed slug and each entry's ``model`` field
        is rewritten back to the served NAME the client asked with. When
        ``model`` is absent (legacy single-slug / direct call) the primary slug
        is used. Custom-voices enrollment stays per-definition (shared data
        dir, spec §7).
        """
        client = self._http_client()
        slug = self._voice_slug(model)
        resp = await client.get(
            f"{self.base_url}/v1/audio/voices",
            params={"model": slug},
            headers=self._auth_headers(),
            timeout=15.0,
        )
        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=resp.text[:500],
            )
        data = resp.json()
        served_name = model if isinstance(model, str) and model else slug
        voices = data.get("voices")
        if isinstance(voices, list):
            data["voices"] = [
                {**v, "model": served_name}
                for v in voices
                if isinstance(v, dict) and v.get("model") in (slug, served_name)
            ]
        return data

    def _custom_voices_dir(self) -> Path:
        data_dir = self._data_dir or talkies_env.resolve_data_dir(self._settings)
        return data_dir / "custom-voices"

    @staticmethod
    def _validate_voice_name(name: str) -> str:
        """Reject traversal / separator names before touching the filesystem.

        The agent route pre-validates; this is the driver-side second gate
        (defense in depth — the driver may be exercised directly).
        """
        if (
            not isinstance(name, str)
            or name in (".", "..")
            or "/" in name
            or "\\" in name
            or not _VOICE_NAME_RE.fullmatch(name)
        ):
            raise HTTPException(status_code=400, detail=f"invalid voice name: {name!r}")
        return name

    async def voice_put(
        self,
        name: str,
        wav: bytes,
        ref_text: str | None,
        language: str | None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Write a cloned voice into ``custom-voices/`` (talkies layout).

        ``<name>.wav`` plus optional sibling sidecars: ``<name>.txt``
        (reference transcript, Qwen3 pairing) and ``<name>.lang``
        (language label). talkies discovers them on the next catalog read.
        The enrollment is attributed to the routed served model (name -> slug);
        the on-disk store is the shared per-definition dir (spec §7).
        """
        self._validate_voice_name(name)
        voices_dir = self._custom_voices_dir()
        voices_dir.mkdir(parents=True, exist_ok=True)
        wav_path = voices_dir / f"{name}.wav"
        await asyncio.to_thread(wav_path.write_bytes, wav)
        txt_path = voices_dir / f"{name}.txt"
        lang_path = voices_dir / f"{name}.lang"
        if ref_text:
            await asyncio.to_thread(txt_path.write_text, ref_text, "utf-8")
        elif txt_path.exists():
            txt_path.unlink(missing_ok=True)
        if language:
            await asyncio.to_thread(lang_path.write_text, language, "utf-8")
        elif lang_path.exists():
            lang_path.unlink(missing_ok=True)
        return {
            "object": "voice",
            "voice_id": name,
            "name": name,
            "model": self._voice_slug(model),
            "language": language,
            "ref_text": ref_text,
            "size_bytes": len(wav),
            "path": str(wav_path),
            "deleted": False,
        }

    async def voice_delete(self, name: str, model: str | None = None) -> dict[str, Any]:
        """Remove a saved voice (wav + sidecars) from ``custom-voices/``.

        The removal is attributed to the routed served model (name -> slug).
        """
        self._validate_voice_name(name)
        voices_dir = self._custom_voices_dir()
        existed = False
        for suffix in (".wav", ".txt", ".lang"):
            path = voices_dir / f"{name}{suffix}"
            if path.exists():
                path.unlink(missing_ok=True)
                if suffix == ".wav":
                    existed = True
        return {
            "object": "voice",
            "voice_id": name,
            "model": self._voice_slug(model),
            "deleted": existed,
        }

    # ------------------------------------------------------------------
    # Phase 24: live ASR WebSocket bridge
    # ------------------------------------------------------------------
    def _rewrite_ws_text(self, text: str) -> str:
        """Rewrite the ``model`` field of a control frame (the start
        message carries the client-facing served name; the engine only knows
        its slug). Non-JSON / model-less frames pass through untouched."""
        try:
            payload = json.loads(text)
        except ValueError:
            return text
        if isinstance(payload, dict) and "model" in payload:
            payload["model"] = self._slug_for_name(payload["model"])
            return json.dumps(payload)
        return text

    async def transcribe_stream(self, websocket: WebSocket) -> None:
        """Bridge the agent-facing WS to the engine's live-ASR WS.

        Bidirectional pump (text + binary frames) with asyncio.gather;
        whichever side ends first cancels the sibling and both sockets
        close. The lifecycle holds the inference slot for the whole
        connection and releases it when this coroutine returns. The engine
        URL carries ``?model=<slug>`` (harmless to talkies, which selects
        the model from the start frame — the driver rewrites that field
        too). The requested served name is read from the inbound query
        params (the route gated the connection on it) and mapped to its slug.
        """
        requested = getattr(websocket, "query_params", {}).get("model")
        slug = self._slug_for_name(requested) if requested else self.slugs[0]
        url = f"{self.ws_url}/v1/audio/transcriptions/stream?model={slug}"
        async with websockets.connect(
            url, additional_headers=self._auth_headers()
        ) as engine_ws:

            async def agent_to_engine() -> None:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    text = message.get("text")
                    if text is not None:
                        await engine_ws.send(self._rewrite_ws_text(text))
                    else:
                        await engine_ws.send(message.get("bytes") or b"")

            async def engine_to_agent() -> None:
                async for raw in engine_ws:
                    if isinstance(raw, str):
                        await websocket.send_text(raw)
                    else:
                        await websocket.send_bytes(raw)

            pumps = [
                asyncio.create_task(agent_to_engine()),
                asyncio.create_task(engine_to_agent()),
            ]
            try:
                done, pending = await asyncio.wait(
                    pumps, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pumps, return_exceptions=True)
                for task in done:
                    exc = task.exception()
                    if exc is not None:
                        logger.debug("talkies WS bridge pump ended: %r", exc)
            finally:
                with contextlib.suppress(Exception):
                    await engine_ws.close()

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    async def stop(self) -> None:
        proc = self._proc
        self._proc = None
        self.backend_port = None
        # No process => nothing resident; clear the health-pin set so a stale
        # spawned slug set can never leak into the next boot's readiness gate.
        self._spawned_slugs = []
        if proc is not None and proc.poll() is None:
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

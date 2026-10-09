"""Model prefetch: replicate the talkies image entrypoint in-driver.

The talkies server runs with ``HF_HUB_OFFLINE=1`` and reads its model
snapshots from ``$TALKIES_DATA_DIR/models/<slug>`` (a flat directory —
its mere non-emptiness is the "cached" signal). The image entrypoint
normally prefetches the enabled slugs before exec'ing the server; because
this provider launches uvicorn directly (private port), the driver must
do the equivalent fetch during the backend's initializing phase, BEFORE
spawn.

Logic mirrors ``entrypoint.sh`` of psyb0t/docker-talkies exactly:
resolve the slug entry in the registry JSON, ``snapshot_download`` the
repo into the flat dir (honoring ``download_patterns`` / the legacy
``gguf_file`` narrowing), then dependency repos into the standard HF
cache. ``HF_HUB_OFFLINE`` is cleared in-process for the prefetch only
(same trick as ``provider_lib.downloader``) — the spawned server keeps
the offline env and never reaches for the Hub at request time.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("provider.talkies.prefetch")

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


class RegistryError(RuntimeError):
    """The registry file is missing/unparsable or lacks the slug."""


def load_registry(models_file: Path) -> dict[str, Any]:
    """Read the talkies registry JSON ({"models": {slug: entry}})."""
    try:
        raw = json.loads(Path(models_file).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RegistryError(f"talkies registry file not found: {models_file}") from exc
    except (OSError, ValueError) as exc:
        raise RegistryError(
            f"talkies registry unreadable: {models_file}: {exc}"
        ) from exc
    models = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(models, dict):
        raise RegistryError(f"talkies registry {models_file} has no 'models' object")
    return models


async def _emit_progress(cb: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if cb is None:
        return
    try:
        await cb(payload)
    except Exception:  # noqa: BLE001 - progress events are best-effort
        logger.debug("download.progress emit failed", exc_info=True)


def _snapshot_kwargs(entry: dict[str, Any], revision: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision
    elif entry.get("revision"):
        kwargs["revision"] = entry["revision"]
    patterns = entry.get("download_patterns")
    if isinstance(patterns, list) and patterns:
        kwargs["allow_patterns"] = [p for p in patterns if isinstance(p, str) and p]
    else:
        # parakeet.cpp-era slugs predate download_patterns; keep their
        # narrow contract (entrypoint.sh parity).
        gguf_file = entry.get("gguf_file")
        if isinstance(gguf_file, str) and gguf_file:
            kwargs["allow_patterns"] = [
                gguf_file,
                "README.md",
                "LICENSE",
                "config.json",
            ]
    return kwargs


def _prefetch_sync(
    slug: str,
    entry: dict[str, Any],
    models_root: Path,
    revision: str | None,
    hf_home: Path,
) -> Path:
    """Blocking prefetch (runs in a worker thread)."""
    from huggingface_hub import constants as hf_constants
    from huggingface_hub import snapshot_download

    # The server env keeps HF_HUB_OFFLINE=1; clear only the hub's
    # in-process switch for this fetch (re-read per request inside
    # huggingface_hub), never os.environ.
    hf_constants.HF_HUB_OFFLINE = False

    target = models_root / slug
    if target.is_dir() and any(target.iterdir()):
        logger.info("talkies model cached: %s -> %s", slug, target)
        return target
    repo = entry.get("repo")
    if not isinstance(repo, str) or not repo:
        raise RegistryError(f"registry entry for {slug!r} has no 'repo'")
    target.mkdir(parents=True, exist_ok=True)
    logger.info("downloading talkies model %s (%s) -> %s", slug, repo, target)
    snapshot_download(repo, local_dir=str(target), **_snapshot_kwargs(entry, revision))
    for dep_repo in entry.get("dependencies") or []:
        if isinstance(dep_repo, str) and dep_repo:
            logger.info("prefetching dependency %s for %s", dep_repo, slug)
            # Explicit cache_dir (not the ambient HF_HOME): this directory
            # IS the hub cache root, and it must equal the HF_HUB_CACHE the
            # driver exports for the engine (env.build_env pins both to
            # $DATA/hf) — the offline server only finds deps under its
            # effective hub cache.
            snapshot_download(dep_repo, cache_dir=str(hf_home))
    return target


async def prefetch_model(
    slug: str,
    entry: dict[str, Any],
    data_dir: Path,
    *,
    revision: str | None = None,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Ensure ``$data_dir/models/<slug>`` holds the snapshot; return it.

    Emits downloader-style ``download.progress`` events (start + terminal
    finished) around the blocking snapshot download.
    """
    models_root = Path(data_dir) / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    hf_home = Path(data_dir) / "hf"
    hf_home.mkdir(parents=True, exist_ok=True)
    target = models_root / slug
    cached = target.is_dir() and any(target.iterdir())
    await _emit_progress(
        progress_cb,
        {
            "file": f"talkies:{slug}",
            "bytes_downloaded": 0,
            "total": None,
            "progress_percent": 100.0 if cached else 0.0,
            "finished": False,
        },
    )
    try:
        result = await asyncio.to_thread(
            _prefetch_sync, slug, entry, models_root, revision, hf_home
        )
    except Exception:
        await _emit_progress(
            progress_cb,
            {
                "file": f"talkies:{slug}",
                "bytes_downloaded": 0,
                "total": None,
                "progress_percent": 0.0,
                "finished": False,
            },
        )
        raise
    await _emit_progress(
        progress_cb,
        {
            "file": f"talkies:{slug}",
            "bytes_downloaded": 0,
            "total": None,
            "progress_percent": 100.0,
            "finished": True,
        },
    )
    return result

"""Shared Phase 9 handlers: ``provider.config.update`` / ``cache.clear`` /
``storage.prune_unused``.

Every provider package wires these into its
``AdminClient.install_command_handlers`` via
:func:`install_config_handlers`; the flow is identical across provider
types and only the driver differs.

Prompt-cache directory convention (see provider/README.md):
``CACHE_DIR/prompt_cache/<config_fingerprint>/`` — providers that write
engine prompt caches write them under the directory keyed by the
fingerprint they were configured with. A fingerprint change therefore
*naturally orphans* the old directory, and clearing = deleting the old
fingerprint's dir (never ``MODELS_DIR``, never model files).

``provider.config.update`` flow (docs/ws-protocol.md §4):

1. Validate the payload (fingerprint string + backend_config object).
2. Adopt the pushed ``capacity`` whenever it differs — capacity is
   enforced here at the provider and never needs a restart, so this runs
   *before* the noop/drain gates.
3. No-op when the received fingerprint equals the applied fingerprint:
   ack ``{"ok": true, "detail": {"noop": true, "capacity_adopted":
   bool, ...}}`` — always safely ackable regardless of load.
4. Otherwise drain: ``lifecycle.stop_if_idle()`` checks busy and
   transitions STOPPING -> STOPPED atomically under the lifecycle lock (a
   concurrent ``acquire_slot()`` can never slip between check and stop).
   Busy -> NAK ``{"ok": false, "error": "backend_in_use", "detail":
   {"step": "drain", "retry_after": N, "in_flight": n}}``; the admin
   retries per its policy.
5. Emit ``provider.status initializing``, clear the old fingerprint's
   prompt-cache dirs (and any provider-specific engine cache dirs via
   ``extra_cache_dirs``), and ``driver.apply_config(new)``.
6. Start the backend (the driver resolves/downloads model artifacts;
   ``download.progress`` events flow).
7. Scrape ``driver.list_models()`` into the ack ``model_metadata``.
8. Record the applied fingerprint in :class:`ConfigState` and ack
   ``{"ok": true, "config_fingerprint": fp, "capacity": c,
   "model_metadata": [...]}``.
9. Any failure: NAK with the step name and error message; the admin
   records the per-instance failure (the provider stays in whatever
   state the lifecycle reached, usually ``error``/``stopped``).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from provider_lib.admin_client import AdminClient
from provider_lib.backend import BackendBusy, BackendLifecycle
from provider_lib.config import ProviderSettings
from provider_lib.wire import BackendStatusValue, Frame

logger = logging.getLogger("provider.config_update")

PROMPT_CACHE_SUBDIR = "prompt_cache"

# Seconds the admin should wait before retrying a drain-refused update.
RETRY_AFTER_SECONDS = 10

# Steps of the config.update flow (reported in NAK detail.step).
_STEP_VALIDATE = "validate"
_STEP_DRAIN = "drain"
_STEP_CACHE_CLEAR = "cache_clear"
_STEP_APPLY_CONFIG = "apply_config"
_STEP_START = "start"


class ConfigState:
    """Provider-side record of the currently applied backend_config.

    ``applied_fingerprint`` is seeded from the registration response
    (``provider_definition.config_fingerprint``) and updated only after a
    successful ``provider.config.update`` apply, so the admin's ack echo
    and the provider's own view stay consistent. Note (N-7): the
    fingerprint means the config is *adopted*, not that a backend is
    *running* under it — registration seeds it before the first start,
    and a later stop leaves the fingerprint in place.
    """

    def __init__(self, applied_fingerprint: str | None = None) -> None:
        self.applied_fingerprint = applied_fingerprint


def prompt_cache_root(settings: ProviderSettings) -> Path:
    """The provider's prompt-cache root: ``CACHE_DIR/prompt_cache``."""
    return Path(settings.CACHE_DIR) / PROMPT_CACHE_SUBDIR


def prompt_cache_dir(settings: ProviderSettings, fingerprint: str) -> Path:
    """The per-fingerprint prompt-cache directory (the convention)."""
    return prompt_cache_root(settings) / fingerprint


def delete_tree(path: Path) -> tuple[list[str], int]:
    """Delete a directory tree (or file). Returns (deleted_paths, bytes).

    Missing paths are a no-op. Never follows symlinks out of the tree.
    """
    deleted: list[str] = []
    freed = 0
    if not path.exists():
        return deleted, freed
    if path.is_file() or path.is_symlink():
        size = _safe_size(path)
        path.unlink()
        return [str(path)], size
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            fp = Path(root) / name
            size = _safe_size(fp)
            fp.unlink(missing_ok=True)
            freed += size
            deleted.append(str(fp))
        for name in dirs:
            dp = Path(root) / name
            if dp.is_dir() and not dp.is_symlink():
                with_removed = _rmdir(dp)
                deleted.extend(with_removed)
    with_removed = _rmdir(path)
    deleted.extend(with_removed)
    return deleted, freed


def _rmdir(path: Path) -> list[str]:
    try:
        path.rmdir()
        return [str(path)]
    except OSError:
        return []


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def clear_prompt_cache(
    settings: ProviderSettings,
    fingerprints: Iterable[str] = (),
    extra_dirs: Iterable[Path] = (),
) -> tuple[list[str], int]:
    """Delete prompt-cache dirs: the given fingerprints' dirs plus the
    shared root and any provider-specific extra dirs.

    Never touches ``MODELS_DIR``. Returns (deleted_paths, bytes_freed).
    """
    deleted: list[str] = []
    freed = 0
    for fp in fingerprints:
        if not fp:
            continue
        d, n = delete_tree(prompt_cache_dir(settings, fp))
        deleted.extend(d)
        freed += n
    roots = [prompt_cache_root(settings), *(Path(p) for p in extra_dirs)]
    for root in roots:
        d, n = delete_tree(root)
        deleted.extend(d)
        freed += n
    return deleted, freed


def plan_prompt_cache_clear(
    settings: ProviderSettings, extra_dirs: Iterable[Path] = ()
) -> tuple[list[str], int]:
    """Compute what :func:`clear_prompt_cache` would delete (dry run)."""
    deleted: list[str] = []
    freed = 0
    roots = [prompt_cache_root(settings), *(Path(p) for p in extra_dirs)]
    for root in roots:
        for f in _iter_files(root):
            deleted.append(str(f))
            freed += _safe_size(f)
    return deleted, freed


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    if root.is_file() or root.is_symlink():
        return [root]
    return [Path(dir_) / name for dir_, _, names in os.walk(root) for name in names]


def _is_referenced(path: Path, referenced: set[str]) -> bool:
    """A file is referenced if it *is* a referenced path or lives under a
    referenced directory (tokenizer dirs).

    Both sides are ``os.path.realpath``-normalized before comparing so a
    non-normalized recorded path (``//``, ``..``, symlinked mounts) can
    never cause a live referenced file to be missed and deleted.
    """
    rp = os.path.realpath(str(path))
    if rp in referenced:
        return True
    for ref in referenced:
        if rp.startswith(ref.rstrip(os.sep) + os.sep):
            return True
    return False


def plan_prune(
    models_dir: Path, referenced: Iterable[str]
) -> tuple[list[str], list[str], int]:
    """Classify ``MODELS_DIR`` files into (deletable, kept, bytes).

    Only files under ``models_dir`` are considered. Empty directories
    under the models root are not deleted (harmless; only files count).
    The reference set is realpath-normalized (see :func:`_is_referenced`).
    """
    ref_set = {os.path.realpath(str(p)) for p in referenced if p}
    deletable: list[str] = []
    kept: list[str] = []
    freed = 0
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return deletable, kept, freed
    for f in _iter_files(models_dir):
        if _is_referenced(f, ref_set):
            kept.append(str(f))
        else:
            deletable.append(str(f))
            freed += _safe_size(f)
    return deletable, kept, freed


def install_config_handlers(
    client: AdminClient,
    lifecycle: BackendLifecycle,
    state: ConfigState,
    settings: ProviderSettings,
    *,
    extra_cache_dirs: (Callable[[], Iterable[Path]] | Iterable[Path] | None) = None,
) -> None:
    """Wire the three Phase 9 admin commands on the provider client.

    ``extra_cache_dirs`` (a list or a zero-arg callable returning one)
    lets a provider add its engine-managed cache directories (e.g.
    gufo's per-instance ``--cache-disk`` dir) to the cleared set; the
    shared prompt-cache root is always included.
    """

    def extras() -> list[Path]:
        if extra_cache_dirs is None:
            return []
        if callable(extra_cache_dirs):
            return list(extra_cache_dirs())
        return list(extra_cache_dirs)

    async def on_config_update(frame: Frame) -> dict[str, Any]:
        payload = frame.payload or {}
        new_fp = payload.get("config_fingerprint")
        backend_config = payload.get("backend_config")
        if not isinstance(new_fp, str) or not new_fp:
            return {
                "ok": False,
                "error": "invalid payload: config_fingerprint (str) required",
                "detail": {"step": _STEP_VALIDATE},
            }
        if not isinstance(backend_config, dict):
            return {
                "ok": False,
                "error": "invalid payload: backend_config must be an object",
                "detail": {"step": _STEP_VALIDATE},
            }

        # 2. Adopt capacity whenever it differs. Capacity is enforced at
        #    the provider and needs no restart, so this runs before the
        #    noop/drain gates: a capacity-only change (same fingerprint)
        #    is adopted without touching the backend.
        #    NOTE: ``idle_timeout_seconds`` is intentionally NOT adopted
        #    here — the idle reaper is admin-side only
        #    (TODO(phase6-idle-reaper) in the scheduler); the provider
        #    consumes no idle setting.
        capacity_adopted = False
        capacity = payload.get("capacity")
        if (
            isinstance(capacity, int)
            and capacity >= 1
            and capacity != lifecycle.capacity
        ):
            lifecycle.capacity = capacity
            capacity_adopted = True

        # 3. Fingerprint compare: nothing to do (a same-fp update is
        #    always safely ackable regardless of load — no stop needed).
        if state.applied_fingerprint == new_fp:
            return {
                "ok": True,
                "detail": {
                    "noop": True,
                    "capacity_adopted": capacity_adopted,
                    "config_fingerprint": new_fp,
                    "capacity": lifecycle.capacity,
                },
            }

        old_fp = state.applied_fingerprint
        step = _STEP_DRAIN
        driver = lifecycle.driver
        # 4. Drain: atomically refuse to stop while a slot is held
        #    (check + STOPPING transition under the lifecycle lock, no
        #    await in between, so no request can slip in).
        try:
            await lifecycle.stop_if_idle()
        except BackendBusy as exc:
            return {
                "ok": False,
                "error": "backend_in_use",
                "detail": {
                    "step": _STEP_DRAIN,
                    "retry_after": RETRY_AFTER_SECONDS,
                    "in_flight": exc.in_flight if exc.in_flight is not None else 0,
                },
            }
        try:
            # 5. Announce initializing.
            await client.send_event(
                "provider.status",
                {
                    "instance_status": "initializing",
                    "backend_status": lifecycle.backend_status,
                },
            )
            step = _STEP_CACHE_CLEAR
            # 6. Clear the orphaned prompt cache (never model files).
            cleared_paths, cleared_bytes = clear_prompt_cache(
                settings, fingerprints=[old_fp] if old_fp else [], extra_dirs=extras()
            )
            # 7. Apply the new config.
            step = _STEP_APPLY_CONFIG
            apply = getattr(driver, "apply_config", None)
            if callable(apply):
                apply(backend_config)
            # 8. Start the backend (artifact resolution + download
            #    happens inside the driver; progress events flow).
            step = _STEP_START
            await lifecycle.start()
            # 9. Scrape discovered model metadata (best-effort: a
            #    metadata scrape failure must not fail the update).
            models: list[dict[str, Any]] = []
            try:
                models = await driver.list_models()
            except Exception:  # noqa: BLE001
                logger.warning("config.update: list_models failed", exc_info=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("config.update failed at step %s", step)
            with_suppressed_errors = {
                "instance_status": "error",
                "backend_status": lifecycle.backend_status,
                "error_message": f"config.update failed at {step}: {exc}",
            }
            try:
                await client.send_event("provider.status", with_suppressed_errors)
            except Exception:  # noqa: BLE001
                logger.warning("config.update: error status emit failed", exc_info=True)
            return {"ok": False, "error": str(exc), "detail": {"step": step}}

        # 10. Commit the applied fingerprint and ack.
        state.applied_fingerprint = new_fp
        await client.send_event(
            "provider.status",
            {
                "instance_status": "running",
                "backend_status": lifecycle.backend_status,
            },
        )
        logger.info(
            "config.update applied: %s -> %s (cache cleared %d bytes)",
            old_fp,
            new_fp,
            cleared_bytes,
        )
        return {
            "ok": True,
            "detail": {
                "config_fingerprint": new_fp,
                "capacity": lifecycle.capacity,
                "model_metadata": models,
                "prompt_cache_deleted": cleared_paths,
                "prompt_cache_bytes_freed": cleared_bytes,
            },
        }

    async def on_cache_clear(frame: Frame) -> dict[str, Any]:
        payload = frame.payload or {}
        dry_run = bool(payload.get("dry_run", False))
        force = bool(payload.get("force", False))
        # N-4: refuse under load unless force is set — clearing engine
        # caches while the backend is mid-request can cause I/O errors in
        # live streams. dry_run never touches files and is always allowed.
        # (Reads the counters without the lifecycle lock: this is an
        # operator-initiated best-effort gate, not a correctness race
        # like the config.update stop — a slot acquired right after this
        # check simply races the clear, which is what `force` means.)
        if (
            not dry_run
            and not force
            and (
                lifecycle.in_flight > 0
                or lifecycle.backend_status == BackendStatusValue.IN_USE
            )
        ):
            return {
                "ok": False,
                "error": "backend_in_use",
                "detail": {
                    "step": "drain",
                    "retry_after": RETRY_AFTER_SECONDS,
                    "in_flight": lifecycle.in_flight,
                },
            }
        extra = extras()
        if dry_run:
            planned, size = plan_prompt_cache_clear(settings, extra)
            return {
                "ok": True,
                "detail": {
                    "dry_run": True,
                    "deleted": planned,
                    "bytes_freed": size,
                },
            }
        deleted, freed = clear_prompt_cache(settings, extra_dirs=extra)
        logger.info("cache.clear freed %d bytes", freed)
        return {
            "ok": True,
            "detail": {"dry_run": False, "deleted": deleted, "bytes_freed": freed},
        }

    async def on_prune_unused(frame: Frame) -> dict[str, Any]:
        payload = frame.payload or {}
        dry_run = bool(payload.get("dry_run", False))
        driver = lifecycle.driver
        referenced = [str(p) for p in getattr(driver, "resolved_artifacts", [])]
        if not referenced:
            # Nothing trustworthy to keep against: pruning now would delete
            # every model on the box. Refuse until the driver has resolved
            # its artifact set (i.e. started at least once).
            return {
                "ok": False,
                "error": (
                    "no_resolved_artifacts: backend has not resolved its "
                    "artifact set; refusing to prune"
                ),
                "detail": {"step": "validate"},
            }
        deletable, kept, size = plan_prune(Path(settings.MODELS_DIR), referenced)
        deleted: list[str] = []
        freed = 0
        if not dry_run:
            for path_str in deletable:
                p = Path(path_str)
                try:
                    freed += _safe_size(p)
                    p.unlink(missing_ok=True)
                    deleted.append(path_str)
                except OSError as exc:
                    logger.warning("prune could not delete %s: %s", path_str, exc)
        logger.info(
            "storage.prune_unused %s: %d files, %d bytes",
            "planned" if dry_run else "deleted",
            len(deleted if not dry_run else deletable),
            freed if not dry_run else size,
        )
        return {
            "ok": True,
            "detail": {
                "dry_run": dry_run,
                "deleted": deleted if not dry_run else deletable,
                "bytes_freed": freed if not dry_run else size,
                "kept": kept,
            },
        }

    client.on_command("provider.config.update", on_config_update)
    client.on_command("cache.clear", on_cache_clear)
    client.on_command("storage.prune_unused", on_prune_unused)

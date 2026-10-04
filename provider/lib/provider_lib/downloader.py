"""Model artifact downloads (HuggingFace) with progress events.

A provider instance has no database; the admin's `backend_config` names
artifacts as `{"source": "hf", "repo": ..., "file": ...}` or as a plain
local `{"path": ...}`. This module turns such a descriptor into a local
file path, downloading when needed and publishing throttled
`download.progress` events through an optional async callback.

Storage layout mirrors the repository: `MODELS_DIR/<repo>/<file>`. The
FLAT gotcha: operators also drop `.gguf` files directly under
`MODELS_DIR`, so a flat `MODELS_DIR/<file>` hit short-circuits the
download too.

`huggingface_hub` is imported lazily inside the download path so the
rest of provider_lib (and packages that never download) stays importable
without it.
"""

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, ClassVar, Self

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]

# In-flight download registry: "repo/file" -> task. Module-level so
# concurrent ensure_artifact() calls for the same artifact across the
# process share one download (legacy pattern).
_DOWNLOAD_TASKS: dict[str, asyncio.Task[str]] = {}


class DownloadError(RuntimeError):
    """Raised when an artifact cannot be resolved or downloaded."""


def _validate_relative(value: str, kind: str) -> Path:
    p = Path(value)
    if p.is_absolute() or ".." in p.parts:
        raise DownloadError(f"{kind} must be relative to its storage root: {value!r}")
    return p


def storage_path(models_dir: Path, repo: str, filename: str) -> Path:
    """Local path mirroring the repository layout, with traversal guards."""
    repo_path = _validate_relative(repo, "repo")
    file_path = _validate_relative(filename, "filename")
    return Path(models_dir) / repo_path / file_path


class _EventPublishingTqdm:
    """Stand-in for tqdm that publishes download.progress events.

    `hf_hub_download` instantiates this class (via `tqdm_class`) and calls
    `update(n)` per downloaded chunk, `refresh()` / `set_postfix_str()`
    on xet transfers, and `close()` at the end. It may also use the bar
    as a context manager and probe tqdm's `get_lock`/`set_lock` API.
    Ported faithfully from the legacy agent's model manager.
    """

    _lock: ClassVar[threading.Lock] = threading.Lock()
    filename: ClassVar[str] = ""
    progress_cb: ClassVar[ProgressCallback | None] = None
    throttle: ClassVar[float] = 1.0
    _loop: ClassVar[asyncio.AbstractEventLoop | None] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.total: int = int(kwargs.get("total") or 0)
        self.n: int = int(kwargs.get("initial") or 0)
        self.closed = False
        self._last_publish = 0.0
        self._last_ts = time.monotonic()

    # -- tqdm API used by huggingface_hub -----------------------------------

    def update(self, n: int = 1) -> None:
        if self.closed:
            return
        self.n += n
        now = time.monotonic()
        if now - self._last_publish >= self.throttle:
            self._last_publish = now
            self._publish(finished=False)

    def update_transfer(self, n: int = 1) -> None:  # noqa: ARG002
        """No-op: transfer bytes are already counted by update() on xet bars."""

    def refresh(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_postfix_str(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_transfer_postfix_str(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_description_str(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_description(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def format_dict(self) -> dict[str, Any]:
        return {"n": self.n, "total": self.total, "rate": 0}

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._publish(finished=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @classmethod
    def set_lock(cls, lock: threading.Lock) -> None:  # pragma: no cover
        cls._lock = lock

    @classmethod
    def get_lock(cls) -> threading.Lock:  # pragma: no cover
        return cls._lock

    # -- internals -----------------------------------------------------------

    def _publish(self, *, finished: bool) -> None:
        if self.progress_cb is None:
            return
        progress = (self.n / self.total * 100.0) if self.total else 0.0
        payload = {
            "file": self.filename,
            "bytes_downloaded": self.n,
            "total": self.total or None,
            "progress_percent": round(min(progress, 100.0), 2),
            "finished": finished,
        }
        # The tqdm callbacks run on the download thread; hop back onto the
        # owning event loop without blocking the transfer. `progress_cb`
        # is a class attribute holding a plain function, so fetch it from
        # the class to avoid implicit `self` binding.
        cb = type(self).progress_cb
        loop = type(self)._loop
        if cb is None or loop is None:  # pragma: no cover - set by maker
            return
        asyncio.run_coroutine_threadsafe(cb(payload), loop)


def _make_tqdm_class(
    filename: str, progress_cb: ProgressCallback, loop: asyncio.AbstractEventLoop
) -> type[_EventPublishingTqdm]:
    """Create a tqdm-compatible class bound to one download job."""

    class _BoundTqdm(_EventPublishingTqdm):
        pass

    _BoundTqdm.filename = filename
    _BoundTqdm.progress_cb = progress_cb
    _BoundTqdm.throttle = _throttle_seconds()
    # run_coroutine_threadsafe needs the loop that owns the callback.
    _BoundTqdm._loop = loop  # type: ignore[attr-defined]
    return _BoundTqdm


def _throttle_seconds() -> float:
    from provider_lib.config import ProviderSettings

    try:
        return float(ProviderSettings().DOWNLOAD_PROGRESS_INTERVAL)
    except Exception:  # noqa: BLE001 - settings may be unavailable in odd envs
        return 1.0


async def _download_hf(
    repo: str,
    filename: str,
    models_dir: Path,
    progress_cb: ProgressCallback | None,
    revision: str | None = None,
    local_subdir: Path | None = None,
) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on install
        raise DownloadError(
            "huggingface_hub is not installed; cannot download HF artifacts "
            f"({repo}/{filename})"
        ) from exc

    if local_subdir is not None:
        local_dir = Path(models_dir) / local_subdir
    else:
        local_dir = Path(models_dir) / _validate_relative(repo, "repo")
    loop = asyncio.get_running_loop()
    tqdm_class = _make_tqdm_class(filename, progress_cb, loop) if progress_cb else None
    kwargs: dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision
    return await asyncio.to_thread(
        lambda: hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(local_dir),
            **({"tqdm_class": tqdm_class} if tqdm_class else {}),
            **kwargs,
        )
    )


async def ensure_artifact(
    artifact: dict[str, Any],
    *,
    models_dir: Path,
    progress_cb: ProgressCallback | None = None,
    revision: str | None = None,
    local_subdir: str | None = None,
) -> str:
    """Resolve an artifact descriptor to a local file path.

    Accepted shapes:
      - `{"path": "/local/file.gguf"}` — validated to exist, returned as-is.
      - `{"source": "hf"|"huggingface", "repo": ..., "file": ...}` —
        candidate resolution first (`models_dir/repo/file`, then flat
        `models_dir/file`), then a de-duplicated HuggingFace download.

    Optional `revision` pins the HF revision, and `local_subdir` places
    the file under `models_dir/<local_subdir>/<file>` instead of the
    default repo-mirrored layout (used by the halogen-flash NPU pin
    downloads into `MODELS_DIR/npu/<id>/`).

    Raises `DownloadError` on unknown shapes, traversal attempts, or a
    missing local file.
    """
    if not isinstance(artifact, dict):
        raise DownloadError(f"artifact descriptor must be an object, got {artifact!r}")

    if artifact.get("path"):
        raw = str(artifact["path"])
        p = Path(raw)
        if not p.is_file():
            raise DownloadError(f"local model file not found: {raw}")
        return str(p)

    repo = artifact.get("repo")
    filename = artifact.get("file")
    source = (artifact.get("source") or "hf").lower()
    if source not in {"hf", "huggingface"}:
        raise DownloadError(f"unsupported artifact source: {source!r}")
    if not repo or not filename:
        raise DownloadError("HF artifact requires both 'repo' and 'file'")

    models_dir = Path(models_dir)
    if local_subdir is not None:
        subdir = _validate_relative(local_subdir, "local_subdir")
        candidates = [models_dir / subdir / _validate_relative(filename, "filename")]
    else:
        # Candidate resolution: repo layout first, then FLAT under MODELS_DIR
        # (operators store .gguf files directly in the models root while DB
        # paths look like /models/{repo}/{file}).
        candidates = [
            storage_path(models_dir, repo, filename),
            models_dir / _validate_relative(filename, "filename"),
            models_dir / Path(filename).name,
        ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    key = (
        f"{repo}/{filename}"
        if local_subdir is None
        else f"{repo}/{filename}@{local_subdir}"
    )
    if revision:
        key = f"{key}#{revision}"
    in_flight = _DOWNLOAD_TASKS.get(key)
    if in_flight is not None and not in_flight.done():
        return await in_flight

    task: asyncio.Task[str] = asyncio.create_task(
        _download_hf(
            repo,
            filename,
            models_dir,
            progress_cb,
            revision=revision,
            local_subdir=(
                _validate_relative(local_subdir, "local_subdir")
                if local_subdir is not None
                else None
            ),
        )
    )
    _DOWNLOAD_TASKS[key] = task
    try:
        return await asyncio.shield(task)
    finally:
        if _DOWNLOAD_TASKS.get(key) is task:
            _DOWNLOAD_TASKS.pop(key, None)

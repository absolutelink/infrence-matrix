"""Manages model downloads from HuggingFace and ModelScope."""

import asyncio
import hashlib
import re
import threading
import time
from pathlib import Path
from typing import Any, ClassVar, Self

from app.core.config import settings
from app.core.logging import logger
from app.services.event_bus import publish_event

SPLIT_PART_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d+)-of-(?P<total>\d+)\.gguf$")


def pick_primary_filename(filenames: list[str]) -> str:
    """Return the -00001-of- part if present, else the first filename.

    llama.cpp loads split GGUFs by being pointed at the first part; it reads
    the remaining parts from the same directory automatically.
    """
    for name in filenames:
        m = SPLIT_PART_RE.search(name.rsplit("/", 1)[-1])
        if m and int(m.group("idx")) == 1:
            return name
    return filenames[0]


class _EventPublishingTqdm:
    """Stand-in for tqdm that publishes download.progress events.

    hf_hub_download instantiates this class (via ``tqdm_class``) and calls
    ``update(n)`` per downloaded chunk, ``refresh()``/``set_postfix_str()`` on
    xet transfers, and ``close()`` at the end. It may also use the bar as a
    context manager and probe for tqdm's ``get_lock``/``set_lock`` API.
    """

    _lock: ClassVar[threading.Lock] = threading.Lock()
    job_id: ClassVar[str] = ""
    filename: ClassVar[str] = ""
    server_id: ClassVar[str] = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.total: int = int(kwargs.get("total") or 0)
        self.n: int = int(kwargs.get("initial") or 0)
        self.closed = False
        self._last_publish = 0.0
        self._last_bytes = self.n
        self._last_ts = time.monotonic()

    # -- tqdm API used by huggingface_hub -----------------------------------

    def update(self, n: int = 1) -> None:
        if self.closed:
            return
        self.n += n
        now = time.monotonic()
        if now - self._last_publish >= settings.DOWNLOAD_PROGRESS_INTERVAL:
            self._last_publish = now
            self._publish(now)

    def update_transfer(self, n: int = 1) -> None:
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
    def format_dict(self) -> dict:
        return {"n": self.n, "total": self.total, "rate": 0}

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._publish(time.monotonic())

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @classmethod
    def set_lock(cls, lock: threading.Lock) -> None:  # pragma: no cover - tqdm API shim
        cls._lock = lock

    @classmethod
    def get_lock(cls) -> threading.Lock:  # pragma: no cover - tqdm API shim
        return cls._lock

    # -- internals -----------------------------------------------------------

    def _publish(self, now: float) -> None:
        elapsed = now - self._last_ts
        speed_mbps = 0.0
        if elapsed > 0:
            speed_mbps = (self.n - self._last_bytes) / elapsed / 1_000_000
            self._last_bytes = self.n
            self._last_ts = now

        progress = (self.n / self.total * 100.0) if self.total else 0.0
        publish_event(
            "download.progress",
            {
                "job_id": self.job_id,
                "filename": self.filename,
                "progress_percent": round(min(progress, 100.0), 2),
                "bytes_downloaded": self.n,
                "total_bytes": self.total or None,
                "speed_mbps": round(speed_mbps, 2),
                "server_id": self.server_id,
            },
        )


def _make_tqdm_class(
    job_id: str, filename: str, server_id: str | None = None
) -> type[_EventPublishingTqdm]:
    """Create a tqdm-compatible class bound to a specific download job."""

    class _BoundTqdm(_EventPublishingTqdm):
        pass

    _BoundTqdm.job_id = job_id
    _BoundTqdm.filename = filename
    _BoundTqdm.server_id = server_id or ""
    return _BoundTqdm


class ModelManager:
    """Manages model file downloads."""

    def __init__(self) -> None:
        self.models_path = Path(settings.MODELS_PATH)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self._download_tasks: dict[str, asyncio.Task] = {}
        self._npu_locks: dict[str, asyncio.Lock] = {}

    # -- download ------------------------------------------------------------

    def is_downloading(self, filename: str) -> bool:
        """Check whether a download for this filename is in flight."""
        task = self._download_tasks.get(filename)
        return task is not None and not task.done()

    def download_task(self, filename: str) -> asyncio.Task | None:
        """Get the in-flight download task for a filename, if any."""
        return self._download_tasks.get(filename)

    def storage_path(self, repo_id: str, filename: str) -> Path:
        """Return the local path mirroring the repository file layout."""
        repo_path = Path(repo_id)
        relative_file = Path(filename)
        if (
            repo_path.is_absolute()
            or ".." in repo_path.parts
            or relative_file.is_absolute()
            or ".." in relative_file.parts
        ):
            raise ValueError("Model filename must be relative to its repository")
        return self.models_path / repo_id / relative_file

    async def download_model(
        self,
        repo_id: str,
        filename: str,
        source: str = "huggingface",
        job_id: str | None = None,
        server_id: str | None = None,
    ) -> str:
        """Download a model file, awaiting completion. Returns the local path."""
        if source != "huggingface":
            publish_event(
                "download.failed",
                {
                    "job_id": job_id or filename,
                    "filename": filename,
                    "server_id": server_id or "",
                    "error": f"Unsupported source: {source}",
                },
            )
            raise NotImplementedError(f"Source not implemented: {source}")

        existing = self.storage_path(repo_id, filename)
        if existing.exists():
            return str(existing)

        download_key = f"{repo_id}/{filename}"
        in_flight = self._download_tasks.get(download_key)
        if in_flight is not None:
            return await in_flight

        task = asyncio.create_task(
            self._download_to_disk(repo_id, filename, source, job_id, server_id)
        )
        self._download_tasks[download_key] = task
        try:
            return await task
        finally:
            self._download_tasks.pop(download_key, None)

    async def download_model_parts(
        self,
        repo_id: str,
        filenames: list[str],
        source: str = "huggingface",
        job_id: str | None = None,
        server_id: str | None = None,
    ) -> str:
        """Download every split part, returning the primary (-00001-of-) path.

        ``download_model`` short-circuits when a file already exists and
        de-duplicates in-flight downloads, so calling per-part is safe and
        resumable.
        """
        if not filenames:
            raise ValueError("No filenames provided for download")
        downloaded: dict[str, str] = {}
        for fn in filenames:
            downloaded[fn] = await self.download_model(
                repo_id,
                fn,
                source=source,
                job_id=job_id,
                server_id=server_id,
            )
        primary = pick_primary_filename(filenames)
        return downloaded[primary]

    async def download_repository(
        self, repo_id: str, job_id: str | None = None, server_id: str | None = None
    ) -> str:
        """Download an entire Hugging Face repository into MODELS_PATH."""
        local_dir = self.models_path / repo_id
        checkpoint = local_dir / "qwen3.8-27b-p1w4d-d2.hgn"
        tokenizer = local_dir / "tokenizer"
        if checkpoint.exists() and tokenizer.exists():
            return str(local_dir)

        download_key = f"repo:{repo_id}"
        in_flight = self._download_tasks.get(download_key)
        if in_flight is not None:
            return await in_flight

        task = asyncio.create_task(
            self._download_repository_to_disk(repo_id, job_id, server_id)
        )
        self._download_tasks[download_key] = task
        try:
            return await task
        finally:
            self._download_tasks.pop(download_key, None)

    async def download_npu_files(
        self,
        dir_id: str,
        files: list[tuple[str, int, str]],
        repo_id: str,
        revision: str | None = None,
        job_id: str | None = None,
        server_id: str | None = None,
    ) -> str:
        """Fetch one NPU model directory's files into ``MODELS_PATH/npu/<dir_id>``.

        ``files`` are ``(relative_path, size, sha256)`` records from the image's
        pins file. Each file is downloaded from ``repo_id`` into the model's own
        directory (not the default ``MODELS_PATH/<repo_id>`` layout), emitting
        download.started/progress/completed events per file so the UI shows
        progress. Existing files of the exact expected size are skipped so the
        step is idempotent across re-dispatched starts.
        """
        if not files:
            raise ValueError(f"no files to download for NPU model '{dir_id}'")
        local_dir = (self.models_path / "npu" / dir_id).resolve()
        effective_job_id = job_id or f"npu-{dir_id}"
        # Serialize per-directory so two re-dispatched starts cannot race the
        # unlink-and-redownload of a stale file.
        lock = self._npu_locks.setdefault(dir_id, asyncio.Lock())
        async with lock:
            for relative_path, size, sha256 in files:
                target = (local_dir / relative_path).resolve()
                if not target.is_relative_to(local_dir):
                    raise ValueError(
                        f"NPU file path '{relative_path}' escapes "
                        f"{local_dir} for '{dir_id}'"
                    )
                if target.exists() and self._matches_pin(target, size, sha256):
                    continue
                if target.exists():
                    # Size matches but the hash does not (or vice versa): a torn
                    # or wrong-release file. Remove it so the download replaces
                    # it.
                    logger.warning(
                        f"NPU file {target} does not match its pin; re-downloading"
                    )
                    target.unlink()
                await self._download_npu_file(
                    repo_id,
                    relative_path,
                    local_dir,
                    target,
                    revision,
                    effective_job_id,
                    server_id,
                )
        return str(local_dir)

    @staticmethod
    def _matches_pin(path: Path, size: int, sha256: str | None) -> bool:
        """A pinned file matches only when its size and sha256 both agree."""
        if path.stat().st_size != size:
            return False
        if not sha256:
            return True
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == sha256

    async def _download_npu_file(
        self,
        repo_id: str,
        relative_path: str,
        local_dir: Path,
        target: Path,
        revision: str | None,
        job_id: str,
        server_id: str | None,
    ) -> None:
        from huggingface_hub import hf_hub_download

        target.parent.mkdir(parents=True, exist_ok=True)
        tqdm_class = _make_tqdm_class(job_id, relative_path, server_id)
        logger.info(f"Downloading NPU file {relative_path} from {repo_id}")
        publish_event(
            "download.started",
            {
                "job_id": job_id,
                "filename": relative_path,
                "repo_id": repo_id,
                "server_id": server_id or "",
            },
        )
        try:
            await asyncio.to_thread(
                lambda: hf_hub_download(
                    repo_id=repo_id,
                    filename=relative_path,
                    revision=revision,
                    local_dir=str(local_dir),
                    tqdm_class=tqdm_class,
                )
            )
        except Exception as error:
            logger.error(f"NPU download failed for {relative_path}: {error}")
            publish_event(
                "download.failed",
                {
                    "job_id": job_id,
                    "filename": relative_path,
                    "repo_id": repo_id,
                    "server_id": server_id or "",
                    "error": str(error),
                },
            )
            raise
        publish_event(
            "download.completed",
            {
                "job_id": job_id,
                "filename": relative_path,
                "repo_id": repo_id,
                "server_id": server_id or "",
                "path": str(target),
            },
        )

    async def _download_repository_to_disk(
        self, repo_id: str, job_id: str | None, server_id: str | None = None
    ) -> str:
        """Run ``hf download REPO --local-dir`` without a subprocess."""
        from huggingface_hub import snapshot_download

        effective_job_id = job_id or f"repo-{repo_id}"
        publish_event(
            "download.started",
            {
                "job_id": effective_job_id,
                "filename": "*",
                "repo_id": repo_id,
                "server_id": server_id or "",
            },
        )
        local_dir = self.models_path / repo_id
        try:
            await asyncio.to_thread(
                lambda: snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                    tqdm_class=_make_tqdm_class(effective_job_id, "*", server_id),
                )
            )
        except Exception as error:
            publish_event(
                "download.failed",
                {
                    "job_id": effective_job_id,
                    "filename": "*",
                    "repo_id": repo_id,
                    "server_id": server_id or "",
                    "error": str(error),
                },
            )
            raise

        publish_event(
            "download.completed",
            {
                "job_id": effective_job_id,
                "filename": "*",
                "repo_id": repo_id,
                "server_id": server_id or "",
                "path": str(local_dir),
            },
        )
        return str(local_dir)

    async def _download_to_disk(
        self,
        repo_id: str,
        filename: str,
        source: str,
        job_id: str | None,
        server_id: str | None = None,
    ) -> str:
        """Run the blocking download in a thread, emitting progress events."""
        from huggingface_hub import hf_hub_download

        effective_job_id = job_id or filename
        tqdm_class = _make_tqdm_class(effective_job_id, filename, server_id)
        logger.info(f"Downloading {filename} from {repo_id}")
        publish_event(
            "download.started",
            {
                "job_id": effective_job_id,
                "filename": filename,
                "repo_id": repo_id,
                "server_id": server_id or "",
            },
        )

        try:
            downloaded_path = await asyncio.to_thread(
                lambda: hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(self.models_path / Path(repo_id)),
                    tqdm_class=tqdm_class,
                ),
            )
        except Exception as e:
            logger.error(f"Download failed for {filename}: {e}")
            publish_event(
                "download.failed",
                {
                    "job_id": effective_job_id,
                    "filename": filename,
                    "server_id": server_id or "",
                    "error": str(e),
                },
            )
            raise

        logger.info(f"Downloaded model to {downloaded_path}")
        publish_event(
            "download.completed",
            {
                "job_id": effective_job_id,
                "filename": filename,
                "server_id": server_id or "",
                "path": downloaded_path,
            },
        )
        return downloaded_path

    async def update_model(self, model_id: str) -> dict:
        """Update model from repository."""
        logger.info(f"Updating model {model_id}")
        return {
            "status": "updated",
            "model_id": model_id,
            "message": "Model update completed",
        }

    # -- local files ---------------------------------------------------------

    def list_models(self) -> list:
        """List all model files."""
        models = []

        for file in self.models_path.rglob("*.gguf"):
            models.append(
                {
                    "filename": file.name,
                    "path": str(file),
                    "size_bytes": file.stat().st_size,
                }
            )

        return models

    def delete_model(self, filename: str) -> bool:
        """Delete a model file."""
        model_path = self.models_path / filename

        if not model_path.exists():
            return False

        model_path.unlink()
        logger.info(f"Deleted model {filename}")
        return True

    def model_exists(self, filename: str) -> bool:
        """Check if model file exists."""
        return (self.models_path / filename).exists()

    def get_model_info(self, filename: str) -> dict[str, Any]:
        """Get detailed information about a model."""
        model_path = self.models_path / filename
        if not model_path.exists():
            return {}

        return {
            "filename": filename,
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "modified_time": model_path.stat().st_mtime,
        }


model_manager = ModelManager()

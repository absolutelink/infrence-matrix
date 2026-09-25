"""Manages model downloads from HuggingFace and ModelScope."""

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, ClassVar, Self

from app.core.config import settings
from app.core.logging import logger
from app.services.event_bus import publish_event


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
            },
        )


def _make_tqdm_class(job_id: str, filename: str) -> type[_EventPublishingTqdm]:
    """Create a tqdm-compatible class bound to a specific download job."""

    class _BoundTqdm(_EventPublishingTqdm):
        pass

    _BoundTqdm.job_id = job_id
    _BoundTqdm.filename = filename
    return _BoundTqdm


class ModelManager:
    """Manages model file downloads."""

    def __init__(self) -> None:
        self.models_path = Path(settings.MODELS_PATH)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self._download_tasks: dict[str, asyncio.Task] = {}

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
    ) -> str:
        """Download a model file, awaiting completion. Returns the local path."""
        if source != "huggingface":
            publish_event(
                "download.failed",
                {
                    "job_id": job_id or filename,
                    "filename": filename,
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
            self._download_to_disk(repo_id, filename, source, job_id)
        )
        self._download_tasks[download_key] = task
        try:
            return await task
        finally:
            self._download_tasks.pop(download_key, None)

    async def download_repository(self, repo_id: str, job_id: str | None = None) -> str:
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

        task = asyncio.create_task(self._download_repository_to_disk(repo_id, job_id))
        self._download_tasks[download_key] = task
        try:
            return await task
        finally:
            self._download_tasks.pop(download_key, None)

    async def _download_repository_to_disk(
        self, repo_id: str, job_id: str | None
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
            },
        )
        local_dir = self.models_path / repo_id
        try:
            await asyncio.to_thread(
                lambda: snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                    tqdm_class=_make_tqdm_class(effective_job_id, "*"),
                )
            )
        except Exception as error:
            publish_event(
                "download.failed",
                {
                    "job_id": effective_job_id,
                    "filename": "*",
                    "repo_id": repo_id,
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
    ) -> str:
        """Run the blocking download in a thread, emitting progress events."""
        from huggingface_hub import hf_hub_download

        effective_job_id = job_id or filename
        tqdm_class = _make_tqdm_class(effective_job_id, filename)
        logger.info(f"Downloading {filename} from {repo_id}")
        publish_event(
            "download.started",
            {
                "job_id": effective_job_id,
                "filename": filename,
                "repo_id": repo_id,
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

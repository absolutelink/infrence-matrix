"""Model file management endpoints."""

import asyncio
import os
import threading
import time
from typing import Any, ClassVar

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.services.event_bus import publish_event

router = APIRouter(prefix="/models", tags=["models"])


class DownloadModelRequest(BaseModel):
    repo_id: str
    filename: str
    source: str = "huggingface"
    job_id: str | None = None


class DownloadProgress(BaseModel):
    status: str
    progress_percent: float
    bytes_downloaded: int
    total_bytes: int | None = None


class _EventPublishingTqdm:
    """Stand-in for tqdm that publishes download.progress events.

    hf_hub_download calls `update(n)` per downloaded chunk and `close()` at
    the end. A class-level lock is exposed because hf_hub's thread pool
    initializer probes for `get_lock`/`set_lock` when a custom class is used.
    """

    _lock: ClassVar[threading.Lock] = threading.Lock()
    job_id: ClassVar[str] = ""
    filename: ClassVar[str] = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.total: int = int(kwargs.get("total") or 0)
        self.n: int = int(kwargs.get("initial") or 0)
        self.closed = False
        self._last_publish = 0.0

    def update(self, n: int = 1) -> None:
        if self.closed:
            return
        self.n += n
        now = time.monotonic()
        if now - self._last_publish >= settings.DOWNLOAD_PROGRESS_INTERVAL:
            self._last_publish = now
            self._publish()

    def _publish(self) -> None:
        progress = (self.n / self.total * 100.0) if self.total else 0.0
        publish_event(
            "download.progress",
            {
                "job_id": self.job_id,
                "filename": self.filename,
                "progress_percent": round(progress, 2),
                "bytes_downloaded": self.n,
                "total_bytes": self.total or None,
            },
        )

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._publish()

    def set_lock(
        self, lock: threading.Lock
    ) -> None:  # pragma: no cover - tqdm API shim
        type(self)._lock = lock

    @classmethod
    def get_lock(cls) -> threading.Lock:  # pragma: no cover - tqdm API shim
        return cls._lock


def _make_tqdm_class(job_id: str, filename: str) -> type[_EventPublishingTqdm]:
    """Create a tqdm subclass bound to a specific download job."""

    class _BoundTqdm(_EventPublishingTqdm):
        pass

    _BoundTqdm.job_id = job_id
    _BoundTqdm.filename = filename
    return _BoundTqdm


@router.post("/download")
async def download_model(request: DownloadModelRequest) -> dict:
    """Download a model from HuggingFace or ModelScope."""
    dest_path = os.path.join(settings.MODELS_PATH, request.filename)

    if os.path.exists(dest_path):
        return {"status": "already_exists", "path": dest_path}

    job_id = request.job_id or request.filename
    tqdm_class = _make_tqdm_class(job_id, request.filename)

    try:
        if request.source == "huggingface":
            from functools import partial

            from huggingface_hub.file_download import hf_hub_download as _hf_dl

            _download = partial(
                _hf_dl,
                repo_id=request.repo_id,
                filename=request.filename,
                local_dir=settings.MODELS_PATH,
                local_dir_use_symlinks=False,
                tqdm_class=tqdm_class,
            )
            downloaded_path = await asyncio.to_thread(_download)
            return {"status": "completed", "path": downloaded_path, "job_id": job_id}
        else:
            raise HTTPException(400, f"Unsupported source: {request.source}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Download failed: {e!s}") from e


@router.get("")
async def list_models() -> dict:
    """List all downloaded models."""
    models = []

    if os.path.exists(settings.MODELS_PATH):
        for file in os.listdir(settings.MODELS_PATH):
            if file.endswith(".gguf"):
                full_path = os.path.join(settings.MODELS_PATH, file)
                models.append(
                    {
                        "filename": file,
                        "path": full_path,
                        "size_bytes": os.path.getsize(full_path),
                    }
                )

    return {"models": models}


@router.delete("/{filename}")
async def delete_model(filename: str) -> dict:
    """Delete a model file."""
    file_path = os.path.join(settings.MODELS_PATH, filename)

    if not os.path.exists(file_path):
        raise HTTPException(404, "Model not found")

    os.remove(file_path)
    return {"status": "deleted"}

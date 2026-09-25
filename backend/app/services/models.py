import asyncio
import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download

from app.core.config import settings
from app.models import DownloadJob, Model

logger = logging.getLogger(__name__)

ModelType = Literal["llm", "mtp", "mmproj", "dflash"]


def guess_model_type(path: str | None, gguf_type: str | None = None) -> ModelType:
    """Guess the model type from the GGUF general.type and filename.

    mmproj / mtp / dflash files are auxiliary; everything else is a llm.
    """
    haystack_parts = []
    if gguf_type:
        haystack_parts.append(gguf_type.lower())
    if path:
        haystack_parts.append(Path(path).name.lower())
    haystack = " ".join(haystack_parts)

    for candidate in ("mmproj", "dflash", "mtp"):
        if candidate in haystack:
            return candidate  # type: ignore[return-value]
    return "llm"


class ModelManager:
    """Manages model downloads and registration."""

    def __init__(self) -> None:
        self.download_jobs: dict[str, DownloadJob] = {}
        self._download_tasks: dict[str, asyncio.Task[None]] = {}

    @staticmethod
    def _repository_path(repo_id: str, filename: str) -> Path:
        """Return the local path that mirrors a repository's file layout."""
        repo_path = Path(repo_id)
        relative_file = Path(filename)
        if (
            repo_path.is_absolute()
            or ".." in repo_path.parts
            or relative_file.is_absolute()
            or ".." in relative_file.parts
        ):
            raise ValueError("Model filename must be relative to its repository")
        return Path(settings.MODELS_PATH) / repo_id / relative_file

    @staticmethod
    def _repository_root(repo_id: str) -> Path:
        return ModelManager._repository_path(repo_id, "placeholder").parent

    async def download_from_huggingface(
        self,
        repo_id: str,
        filename: str,
        download_job: DownloadJob,
    ) -> bool:
        """Download a model from HuggingFace Hub."""
        download_job.source = "huggingface"
        download_job.repo_id = repo_id
        download_job.filename = filename
        download_job.status = "downloading"
        download_job.started_at = datetime.now(UTC)

        try:
            dest_dir = self._repository_root(repo_id)
            dest_dir.mkdir(parents=True, exist_ok=True)

            def _download() -> str:
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(dest_dir),
                    local_dir_use_symlinks=False,
                )

            loop = asyncio.get_event_loop()
            downloaded_path = await loop.run_in_executor(None, _download)

            downloaded_file = Path(downloaded_path)
            file_size = downloaded_file.stat().st_size

            download_job.bytes_downloaded = file_size
            download_job.total_bytes = file_size
            download_job.progress_percent = 100.0
            download_job.status = "completed"
            download_job.completed_at = datetime.now(UTC)
            download_job.destination_path = str(downloaded_file)

            logger.info(f"Downloaded {filename} from {repo_id} to {downloaded_file}")
            return True

        except Exception as e:
            logger.error(f"Download failed: {e}")
            download_job.status = "failed"
            download_job.last_error = str(e)
            download_job.last_error_at = datetime.now(UTC)
            return False

    async def download_from_modelscope(
        self,
        repo_id: str,
        filename: str,
        download_job: DownloadJob,
    ) -> bool:
        """Download a model from ModelScope."""
        download_job.source = "modelscope"
        download_job.repo_id = repo_id
        download_job.filename = filename
        download_job.status = "downloading"
        download_job.started_at = datetime.now(UTC)

        try:
            dest_dir = self._repository_root(repo_id)
            dest_dir.mkdir(parents=True, exist_ok=True)

            def _download() -> str:
                try:
                    from modelscope.hub.snapshot_download import (
                        snapshot_download as ms_snapshot_download,
                    )

                    return ms_snapshot_download(
                        repo_id=repo_id,
                        allow_patterns=[filename],
                        local_dir=str(dest_dir),
                    )
                except ImportError:
                    raise RuntimeError("modelscope package not installed")

            loop = asyncio.get_event_loop()
            downloaded_path = await loop.run_in_executor(None, _download)

            downloaded_file = Path(downloaded_path) / filename
            file_size = downloaded_file.stat().st_size

            download_job.bytes_downloaded = file_size
            download_job.total_bytes = file_size
            download_job.progress_percent = 100.0
            download_job.status = "completed"
            download_job.completed_at = datetime.now(UTC)
            download_job.destination_path = str(downloaded_file)

            logger.info(
                f"Downloaded {filename} from ModelScope {repo_id} to {downloaded_file}"
            )
            return True

        except Exception as e:
            logger.error(f"Download failed: {e}")
            download_job.status = "failed"
            download_job.last_error = str(e)
            download_job.last_error_at = datetime.now(UTC)
            return False

    def extract_model_metadata(
        self, model_path: str
    ) -> dict[str, str | int | bool | None]:
        """Extract metadata from a GGUF model file."""
        try:
            from gguf import GGUFReader

            reader = GGUFReader(model_path)
            metadata = {
                "architecture": "unknown",
                "model_type": "llm",
                "parameter_count": None,
                "quantization": "unknown",
                "supports_embeddings": False,
                "supports_vision": False,
            }

            gguf_type = None
            if reader.metadata:
                if "general.architecture" in reader.metadata:
                    metadata["architecture"] = str(
                        reader.metadata["general.architecture"]
                    )

                if "general.parameter_count" in reader.metadata:
                    metadata["parameter_count"] = int(
                        reader.metadata["general.parameter_count"]
                    )

                if "general.quantization_version" in reader.metadata:
                    metadata["quantization"] = str(
                        reader.metadata["general.quantization_version"]
                    )

                if "llama.context_length" in reader.metadata:
                    metadata["context_length"] = int(
                        reader.metadata["llama.context_length"]
                    )

                if "general.type" in reader.metadata:
                    gguf_type = str(reader.metadata["general.type"])
                    metadata["supports_embeddings"] = "embedding" in gguf_type.lower()

            metadata["model_type"] = guess_model_type(model_path, gguf_type)

            return metadata

        except ImportError:
            logger.warning("gguf package not installed, using default metadata")
            return {
                "architecture": "unknown",
                "model_type": "llm",
                "parameter_count": None,
                "quantization": "unknown",
                "context_length": 4096,
                "supports_embeddings": False,
                "supports_vision": False,
            }
        except Exception as e:
            logger.error(f"Failed to extract metadata from {model_path}: {e}")
            return {
                "architecture": "unknown",
                "model_type": "llm",
                "parameter_count": None,
                "quantization": "unknown",
                "context_length": 4096,
                "supports_embeddings": False,
                "supports_vision": False,
            }

    def compute_checksum(self, file_path: str) -> str:
        """Compute SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def register_model(
        self,
        name: str,
        path: str,
        source: Literal["huggingface", "modelscope", "manual"],
        source_repo_id: str | None = None,
    ) -> Model:
        """Register a model in the database."""
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        metadata = self.extract_model_metadata(path)
        file_size = model_path.stat().st_size

        model = Model(
            name=name,
            path=str(model_path.absolute()),
            size_bytes=file_size,
            architecture=metadata["architecture"],
            model_type=metadata["model_type"],
            parameter_count=metadata["parameter_count"],
            quantization=metadata["quantization"],
            supports_embeddings=metadata["supports_embeddings"],
            supports_vision=metadata["supports_vision"],
            source=source,
            source_repo_id=source_repo_id,
            source_url=None,
            source_file=name,
        )

        logger.info(f"Registered model: {name} ({file_size} bytes)")
        return model

    def list_local_models(self) -> list[Model]:
        """Scan local models directory and return list of models."""
        models_dir = Path(settings.MODELS_PATH)
        if not models_dir.exists():
            return []

        models = []
        for file in models_dir.rglob("*.gguf"):
            try:
                model = self.register_model(
                    name=file.name,
                    path=str(file),
                    source="manual",
                )
                models.append(model)
            except Exception as e:
                logger.error(f"Failed to register model {file.name}: {e}")

        return models

    async def pause_download(self, job_id: str) -> bool:
        """Pause a download job."""
        if job_id not in self.download_jobs:
            return False

        job = self.download_jobs[job_id]
        if job.status != "downloading":
            return False

        job.status = "paused"
        logger.info(f"Paused download job {job_id}")
        return True

    async def resume_download(self, job_id: str) -> bool:
        """Resume a paused download job."""
        if job_id not in self.download_jobs:
            return False

        job = self.download_jobs[job_id]
        if job.status != "paused":
            return False

        job.status = "downloading"
        logger.info(f"Resumed download job {job_id}")

        if job.source == "huggingface":
            asyncio.create_task(
                self.download_from_huggingface(
                    repo_id=job.repo_id,
                    filename=job.filename,
                    download_job=job,
                )
            )
        elif job.source == "modelscope":
            asyncio.create_task(
                self.download_from_modelscope(
                    repo_id=job.repo_id,
                    filename=job.filename,
                    download_job=job,
                )
            )

        return True

    async def cancel_download(self, job_id: str) -> bool:
        """Cancel a download job."""
        if job_id not in self.download_jobs:
            return False

        job = self.download_jobs[job_id]
        if job_id in self._download_tasks:
            self._download_tasks[job_id].cancel()

        job.status = "cancelled"
        logger.info(f"Cancelled download job {job_id}")
        return True


model_manager = ModelManager()

"""Model file management endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio
import os

from app.core.config import settings

router = APIRouter(prefix="/models", tags=["models"])


class DownloadModelRequest(BaseModel):
    repo_id: str
    filename: str
    source: str = "huggingface"


class DownloadProgress(BaseModel):
    status: str
    progress_percent: float
    bytes_downloaded: int
    total_bytes: int | None = None


@router.post("/download")
async def download_model(request: DownloadModelRequest) -> dict:
    """Download a model from HuggingFace or ModelScope."""
    from huggingface_hub import hf_hub_download

    dest_path = os.path.join(settings.MODELS_PATH, request.filename)

    if os.path.exists(dest_path):
        return {"status": "already_exists", "path": dest_path}

    try:
        if request.source == "huggingface":
            downloaded_path = await asyncio.to_thread(
                hf_hub_download,
                repo_id=request.repo_id,
                filename=request.filename,
                local_dir=settings.MODELS_PATH,
                local_dir_use_symlinks=False,
            )
            return {"status": "completed", "path": downloaded_path}
        else:
            raise HTTPException(400, f"Unsupported source: {request.source}")
    except Exception as e:
        raise HTTPException(500, f"Download failed: {str(e)}")


@router.get("")
async def list_models() -> dict:
    """List all downloaded models."""
    models = []

    if os.path.exists(settings.MODELS_PATH):
        for file in os.listdir(settings.MODELS_PATH):
            if file.endswith(".gguf"):
                full_path = os.path.join(settings.MODELS_PATH, file)
                models.append({
                    "filename": file,
                    "path": full_path,
                    "size_bytes": os.path.getsize(full_path),
                })

    return {"models": models}


@router.delete("/{filename}")
async def delete_model(filename: str) -> dict:
    """Delete a model file."""
    file_path = os.path.join(settings.MODELS_PATH, filename)

    if not os.path.exists(file_path):
        raise HTTPException(404, "Model not found")

    os.remove(file_path)
    return {"status": "deleted"}

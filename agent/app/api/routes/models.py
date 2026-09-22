"""Model file management endpoints."""

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.services.model_manager import model_manager

router = APIRouter(prefix="/models", tags=["models"])


class DownloadModelRequest(BaseModel):
    repo_id: str
    filename: str
    source: str = "huggingface"
    job_id: str | None = None


@router.post("/download")
async def download_model(request: DownloadModelRequest) -> dict:
    """Download a model from HuggingFace or ModelScope."""
    dest_path = os.path.join(settings.MODELS_PATH, request.filename)

    if model_manager.model_exists(request.filename):
        return {"status": "already_exists", "path": dest_path, "job_id": request.job_id}

    if model_manager.is_downloading(request.filename):
        return {"status": "in_progress", "path": dest_path, "job_id": request.job_id}

    try:
        downloaded_path = await model_manager.download_model(
            repo_id=request.repo_id,
            filename=request.filename,
            source=request.source,
            job_id=request.job_id,
        )
        return {
            "status": "completed",
            "path": downloaded_path,
            "job_id": request.job_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Download failed: {e!s}") from e


@router.get("/status/{filename}")
async def download_status(filename: str) -> dict:
    """Check download/local status of a model file."""
    if model_manager.model_exists(filename):
        info = model_manager.get_model_info(filename)
        return {"status": "available", "filename": filename, **info}

    if model_manager.is_downloading(filename):
        return {"status": "downloading", "filename": filename}

    return {"status": "missing", "filename": filename}


@router.get("")
async def list_models() -> dict:
    """List all downloaded models."""
    return {"models": model_manager.list_models()}


@router.delete("/{filename}")
async def delete_model(filename: str) -> dict:
    """Delete a model file."""
    if not model_manager.delete_model(filename):
        raise HTTPException(404, "Model not found")
    return {"status": "deleted"}

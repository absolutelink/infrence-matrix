from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from pathlib import Path

from app.core.config import settings

router = APIRouter(prefix="/models", tags=["models"])


class ModelInfo(BaseModel):
    filename: str
    path: str
    size_bytes: int


@router.get("")
async def list_models() -> dict:
    """List all model files."""
    models_path = Path(settings.MODELS_PATH)
    
    if not models_path.exists():
        return {"models": []}
    
    models = []
    for file in models_path.glob("*.gguf"):
        models.append({
            "filename": file.name,
            "path": str(file),
            "size_bytes": file.stat().st_size,
        })
    
    return {"models": models}


@router.post("/download")
async def download_model(repo_id: str, filename: str) -> dict:
    """Download a model from HuggingFace."""
    # TODO: Implement huggingface_hub download
    return {"job_id": "download-uuid", "status": "downloading"}


@router.delete("/{filename}")
async def delete_model(filename: str) -> dict:
    """Delete a model file."""
    model_path = Path(settings.MODELS_PATH) / filename
    
    if not model_path.exists():
        raise HTTPException(404, "Model not found")
    
    model_path.unlink()
    
    return {"deleted": True}

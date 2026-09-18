"""Manages model downloads from HuggingFace and ModelScope."""

import asyncio
from pathlib import Path
from typing import Optional, Callable

from app.core.config import settings
from app.core.logging import logger


class ModelManager:
    """Manages model file downloads."""
    
    def __init__(self) -> None:
        self.models_path = Path(settings.MODELS_PATH)
        self.models_path.mkdir(parents=True, exist_ok=True)
    
    async def download_model(
        self,
        repo_id: str,
        filename: str,
        source: str = "huggingface",
        on_progress: Optional[Callable[[float, int, int], None]] = None,
    ) -> str:
        """Download a model from HuggingFace or ModelScope."""
        from huggingface_hub import hf_hub_download
        
        logger.info(f"Downloading {filename} from {repo_id}")
        
        try:
            if source == "huggingface":
                local_path = await asyncio.to_thread(
                    hf_hub_download,
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(self.models_path),
                )
            else:
                # TODO: Implement ModelScope download
                raise NotImplementedError("ModelScope not implemented")
            
            logger.info(f"Downloaded model to {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise
    
    def list_models(self) -> list:
        """List all model files."""
        models = []
        
        for file in self.models_path.glob("*.gguf"):
            models.append({
                "filename": file.name,
                "path": str(file),
                "size_bytes": file.stat().st_size,
            })
        
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

"""V1 API routes."""

from app.api.routes.v1.responses import responses_router, responses_ws_router
from app.api.routes.v1.v1_audio import router as audio_router
from app.api.routes.v1.v1_batches import router as batches_router
from app.api.routes.v1.v1_chat_completions import router as chat_completions_router
from app.api.routes.v1.v1_completions import router as completions_router
from app.api.routes.v1.v1_embeddings import router as embeddings_router
from app.api.routes.v1.v1_files import router as files_router
from app.api.routes.v1.v1_models import router as models_router

__all__ = [
    "audio_router",
    "batches_router",
    "chat_completions_router",
    "completions_router",
    "embeddings_router",
    "files_router",
    "models_router",
    "responses_router",
    "responses_ws_router",
]

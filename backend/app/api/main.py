from fastapi import APIRouter

from app.api.routes import items, login, models, huggingface, private, users, utils, agents, metrics
from app.api.routes.v1 import (
    audio_router,
    batches_router,
    chat_completions_router,
    completions_router,
    embeddings_router,
    files_router,
    models_router,
    responses_router,
)
from app.core.config import settings

api_router = APIRouter()
api_router.include_router(login.router)
api_router.include_router(users.router)
api_router.include_router(utils.router)
api_router.include_router(items.router)
api_router.include_router(models.router, tags=["models"])
api_router.include_router(huggingface.router, tags=["huggingface"])
api_router.include_router(agents.router, prefix="/api", tags=["agents"])
api_router.include_router(metrics.router, tags=["metrics"])

api_router.include_router(models_router, prefix="/v1", tags=["v1/models"])
api_router.include_router(chat_completions_router, prefix="/v1", tags=["v1/chat/completions"])
api_router.include_router(completions_router, prefix="/v1", tags=["v1/completions"])
api_router.include_router(embeddings_router, prefix="/v1", tags=["v1/embeddings"])
api_router.include_router(responses_router, prefix="/v1", tags=["v1/responses"])
api_router.include_router(files_router, prefix="/v1", tags=["v1/files"])
api_router.include_router(batches_router, prefix="/v1", tags=["v1/batches"])
api_router.include_router(audio_router, prefix="/v1", tags=["v1/audio"])

if settings.FASTAPI_ENV == "development":
    api_router.include_router(private.router)

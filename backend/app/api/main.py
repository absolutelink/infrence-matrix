from fastapi import APIRouter

from app.api.routes import items, login, models, huggingface, private, users, utils, agents, metrics
from app.core.config import settings

api_router = APIRouter()
api_router.include_router(login.router)
api_router.include_router(users.router)
api_router.include_router(utils.router)
api_router.include_router(items.router)
api_router.include_router(models.router, tags=["models"])
api_router.include_router(huggingface.router, tags=["huggingface"])
api_router.include_router(agents.router, tags=["agents"])
api_router.include_router(metrics.router, tags=["metrics"])

if settings.FASTAPI_ENV == "development":
    api_router.include_router(private.router)

from fastapi import APIRouter

from app.api.routes import (
    agents,
    benchmarks,
    huggingface,
    metrics,
    models,
    queue,
    server_instances,
    utils,
)

api_router = APIRouter()
api_router.include_router(agents.router, tags=["agents"])
api_router.include_router(metrics.router, tags=["metrics"])
api_router.include_router(server_instances.router, tags=["server-instances"])
api_router.include_router(benchmarks.router, tags=["benchmarks"])
api_router.include_router(models.router, tags=["models"])
api_router.include_router(queue.router, tags=["queue"])
api_router.include_router(huggingface.router, tags=["huggingface"])
api_router.include_router(utils.router, tags=["utils"])

"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import servers, models, gpu, websocket, proxy
from app.core.config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Inference Matrix Agent",
        description="Agent service for distributed inference",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(servers.router)
    app.include_router(models.router)
    app.include_router(gpu.router)
    app.include_router(websocket.router)
    app.include_router(proxy.router)

    @app.get("/health")
    async def health_check() -> dict:
        return {"status": "healthy"}

    return app


app = create_app()

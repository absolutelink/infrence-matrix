"""FastAPI application factory."""

import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import servers, models, gpu, websocket, proxy
from app.core.config import settings
from app.core.logging import logger
from app.services.frontend_client import frontend_client


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

    @app.on_event("startup")
    async def startup_event():
        """Startup event to initialize services."""
        logger.info("Starting Inference Matrix Agent services...")
        
        # Start background tasks for frontend connection
        try:
            await frontend_client.start_background_tasks()
            logger.info("Started frontend client background tasks")
        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown event to cleanup services."""
        logger.info("Shutting down Inference Matrix Agent services...")
        await frontend_client.close()
        logger.info("Agent shutdown complete")

    return app


app = create_app()

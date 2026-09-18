from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import logger
from app.api.routes import servers, models, gpu, websocket


def create_application() -> FastAPI:
    """Create and configure FastAPI application."""

    application = FastAPI(
        title="Inference Matrix Agent",
        description="Manages llama.cpp servers for Inference Matrix",
        version="0.1.0",
    )

    # CORS (for local development)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    application.include_router(servers.router, prefix="/api")
    application.include_router(models.router, prefix="/api")
    application.include_router(gpu.router, prefix="/api")
    application.include_router(websocket.router, prefix="/api")

    # Health check
    @application.get("/api/health")
    async def health_check() -> dict:
        return {
            "status": "healthy",
            "agent_id": settings.AGENT_ID,
        }

    @application.on_event("startup")
    async def startup_event() -> None:
        logger.info(f"Starting Agent {settings.AGENT_ID}")
        # Register with Frontend
        # from app.services.frontend_client import frontend_client
        # await frontend_client.register()

    @application.on_event("shutdown")
    async def shutdown_event() -> None:
        logger.info("Shutting down Agent")
        # Cleanup: stop all servers

    return application


app = create_application()

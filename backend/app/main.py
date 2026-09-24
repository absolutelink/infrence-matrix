from contextlib import asynccontextmanager
from pathlib import Path

import sentry_sdk
from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from app.api.main import api_router
from app.api.routes.v1 import (
    audio_router,
    batches_router,
    chat_completions_router,
    completions_router,
    embeddings_router,
    files_router,
    models_router,
    responses_router,
    responses_ws_router,
)
from app.api.routes.websocket import router as agent_ws_router
from app.core.config import settings
from app.services.agent_manager import agent_manager

FRONTEND_DIR = Path(__file__).parent / "frontend"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    agent_manager.start_cleanup_loop()
    yield


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


if settings.SENTRY_DSN and settings.FASTAPI_ENV != "development":
    sentry_sdk.init(dsn=str(settings.SENTRY_DSN), enable_tracing=True)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
    if settings.CORS_ALLOW_ALL_ORIGINS
    else settings.FRONTEND_HOSTS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

# Agent WebSocket - mounted at /api (not /api/v1) so agents connect to
# /api/ws/agents/{agent_id}
app.include_router(agent_ws_router, prefix="/api")

# OpenAI-compatible endpoints - these need to be at /v1/... not /api/v1/v1/...
app.include_router(models_router, prefix="/v1", tags=["v1/models"])
app.include_router(chat_completions_router, prefix="/v1", tags=["v1/chat"])
app.include_router(completions_router, prefix="/v1", tags=["v1/completions"])
app.include_router(embeddings_router, prefix="/v1", tags=["v1/embeddings"])
app.include_router(responses_router, prefix="/v1", tags=["v1/responses"])
# Responses API WebSocket (spec transport): mounted at /v1 directly so the
# handshake is not wrapped by the APIRouter include indirection.
app.include_router(responses_ws_router, prefix="/v1", tags=["v1/responses-ws"])
app.include_router(files_router, prefix="/v1", tags=["v1/files"])
app.include_router(batches_router, prefix="/v1", tags=["v1/batches"])
app.include_router(audio_router, prefix="/v1", tags=["v1/audio"])

app.frontend("/", directory=FRONTEND_DIR)

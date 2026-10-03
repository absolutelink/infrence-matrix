import asyncio
import logging
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
    decisions_router,
    embeddings_router,
    files_router,
    models_router,
    moderations_router,
    rerank_router,
    responses_router,
    responses_ws_router,
)
from app.api.routes.websocket import router as agent_ws_router
from app.core.config import settings
from app.services.agent_manager import agent_manager
from app.services.benchmark import start_queue_worker, stop_queue_worker
from app.services.http_client import aclose_http_client
from app.services.inference_scheduler import inference_scheduler
from app.services.request_activity import (
    request_finished,
    request_started,
)
from app.services.token_stats import token_stats_prune_loop

FRONTEND_DIR = Path(__file__).parent / "frontend"
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    agent_manager.start_cleanup_loop()
    reconciled = await inference_scheduler.reconcile_persisted_leases()
    if reconciled:
        logger.warning(
            "Invalidated %d persisted inference lease(s) at startup", reconciled
        )
    inference_scheduler.start_reconciliation()
    start_queue_worker()
    prune_task = asyncio.create_task(token_stats_prune_loop())
    from app.api.routes.metrics import metrics_snapshot_loop

    metrics_task = asyncio.create_task(metrics_snapshot_loop())
    try:
        yield
    finally:
        metrics_task.cancel()
        prune_task.cancel()
        await asyncio.gather(prune_task, metrics_task, return_exceptions=True)
        await inference_scheduler.stop_reconciliation()
        await stop_queue_worker()
        await aclose_http_client()


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


@app.middleware("http")
async def track_inference_activity(request, call_next):
    """Keep streaming inference requests active until their body completes."""
    if not request.url.path.startswith("/v1/"):
        return await call_next(request)
    await request_started()
    response = None
    try:
        response = await call_next(request)
        body_iterator = getattr(response, "body_iterator", None)
        if body_iterator is None:
            await request_finished()
            return response

        original_iterator = body_iterator

        async def tracked_iterator():
            try:
                async for chunk in original_iterator:
                    yield chunk
            finally:
                await request_finished()

        response.body_iterator = tracked_iterator()
        return response
    except Exception:
        await request_finished()
        raise


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
app.include_router(rerank_router, prefix="/v1", tags=["v1/rerank"])
app.include_router(decisions_router, prefix="/v1", tags=["v1/decisions"])
app.include_router(moderations_router, prefix="/v1", tags=["v1/moderations"])
app.include_router(responses_router, prefix="/v1", tags=["v1/responses"])
# Responses API WebSocket (spec transport): mounted at /v1 directly so the
# handshake is not wrapped by the APIRouter include indirection.
app.include_router(responses_ws_router, prefix="/v1", tags=["v1/responses-ws"])
app.include_router(files_router, prefix="/v1", tags=["v1/files"])
app.include_router(batches_router, prefix="/v1", tags=["v1/batches"])
app.include_router(audio_router, prefix="/v1", tags=["v1/audio"])

app.frontend("/", directory=FRONTEND_DIR)

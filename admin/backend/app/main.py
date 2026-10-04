"""Inference Matrix admin application.

URL layout:
  /                 Swagger UI
  /openapi.json     OpenAPI schema
  /admin/*          React UI (static build)
  /admin/api/*      Admin management API
  /v1/*             Public OpenAI-compatible inference API
  /provider/ws      Provider instance WebSocket dial-in
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.redis import create_redis_client

FRONTEND_DIR = Path(__file__).parent / "frontend"


def custom_generate_unique_id(route: APIRoute) -> str:
    tag = route.tags[0] if route.tags else "default"
    return f"{tag}-{route.name}"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    redis_client = create_redis_client()
    _app.state.redis = redis_client

    sweep_task: asyncio.Task | None = None  # type: ignore[type-arg]
    sweep_stop = asyncio.Event()
    if settings.INSTANCE_SWEEP_ENABLED:
        from app.services.presence_sweep import presence_sweep_loop

        sweep_task = asyncio.create_task(presence_sweep_loop(redis_client, sweep_stop))

    try:
        yield
    finally:
        if sweep_task is not None:
            sweep_stop.set()
            sweep_task.cancel()
            try:
                await sweep_task
            except asyncio.CancelledError:
                pass
        await redis_client.aclose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url="/openapi.json",
    docs_url="/",
    redoc_url=None,
    generate_unique_id_function=custom_generate_unique_id,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.CORS_ALLOW_ALL_ORIGINS else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Route registration happens in app/api/main.py as phases land. The frontend
# SPA is mounted last so /admin/* falls through to index.html.
from app.api.main import api_router  # noqa: E402

app.include_router(api_router)

app.frontend("/admin", directory=FRONTEND_DIR, fallback="index.html", check_dir="auto")

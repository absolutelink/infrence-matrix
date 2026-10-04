from fastapi import APIRouter

from app.api.admin.health import router as admin_health_router
from app.api.admin.providers import router as admin_providers_router
from app.api.v1.responses import router as v1_responses_router
from app.api.ws import router as provider_ws_router

api_router = APIRouter()

api_router.include_router(admin_health_router)
api_router.include_router(admin_providers_router)
# Public OpenAI-compatible inference API lives at /v1 (NOT under /admin/api).
api_router.include_router(v1_responses_router)
# Provider WebSocket lives at the app root (/provider/ws), not under /admin/api.
api_router.include_router(provider_ws_router)

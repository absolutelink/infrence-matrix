from fastapi import APIRouter

from app.api.admin.definitions import router as admin_definitions_router
from app.api.admin.health import router as admin_health_router
from app.api.admin.instances import router as admin_instances_router
from app.api.admin.machines import router as admin_machines_router
from app.api.admin.providers import router as admin_providers_router
from app.api.v1.chat_completions import router as v1_chat_router
from app.api.v1.models import router as v1_models_router
from app.api.v1.responses import router as v1_responses_router
from app.api.v1.stubs import router as v1_stubs_router
from app.api.ws import router as provider_ws_router

api_router = APIRouter()

api_router.include_router(admin_health_router)
api_router.include_router(admin_providers_router)
# Phase 9 operator CRUD + instance actions (trusted LAN, like the rest of
# /admin/api).
api_router.include_router(admin_machines_router)
api_router.include_router(admin_definitions_router)
api_router.include_router(admin_instances_router)
# Public OpenAI-compatible inference API lives at /v1 (NOT under /admin/api).
api_router.include_router(v1_responses_router)
api_router.include_router(v1_chat_router)
api_router.include_router(v1_models_router)
# 501 stubs for dropped endpoints (registered last; explicit paths only).
api_router.include_router(v1_stubs_router)
# Provider WebSocket lives at the app root (/provider/ws), not under /admin/api.
api_router.include_router(provider_ws_router)

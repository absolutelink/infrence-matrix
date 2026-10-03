from fastapi import APIRouter

from app.api.admin.health import router as admin_health_router

api_router = APIRouter()

api_router.include_router(admin_health_router)

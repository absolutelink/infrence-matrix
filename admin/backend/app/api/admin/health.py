from fastapi import APIRouter

router = APIRouter(prefix="/admin/api", tags=["admin"])


@router.get("/health")
def admin_health() -> dict[str, str]:
    """Liveness and version probe for the admin service."""
    from app.core.config import settings

    return {"status": "ok", "version": settings.VERSION}

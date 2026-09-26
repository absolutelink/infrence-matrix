"""V1 Models endpoint - OpenAI-compatible model listing (server aliases)."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, col, select

from app.api.deps import get_db
from app.models import ServerInstance

logger = logging.getLogger(__name__)

router = APIRouter()


class ModelData(BaseModel):
    """Model data for OpenAI API response."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "inference-matrix"


class ModelsList(BaseModel):
    """List of models response."""

    object: str = "list"
    data: list[ModelData]


@router.get("/models", response_model=ModelsList)
def list_models(db: Session = Depends(get_db)) -> ModelsList:
    """
    List all configured server models.

    Server aliases remain discoverable even when their server is stopped.
    Requests using a stopped model can trigger the normal server startup flow.
    """
    statement = select(ServerInstance).where(
        col(ServerInstance.status).in_(["stopped", "starting", "running", "error"])
    )
    instances = db.exec(statement).all()

    model_data = []
    for instance in instances:
        if not instance.alias:
            continue
        metadata_created = (instance.model_metadata or {}).get("created")
        created_timestamp = (
            int(metadata_created)
            if isinstance(metadata_created, int)
            else int(instance.started_at.timestamp())
            if instance.started_at
            else 0
        )
        owned_by = (instance.model_metadata or {}).get("owned_by")
        model_data.append(
            ModelData(
                id=instance.alias,
                created=created_timestamp,
                owned_by=owned_by if isinstance(owned_by, str) else "inference-matrix",
            )
        )

    return ModelsList(data=model_data)


@router.get("/models/{model_id}")
def retrieve_model(model_id: str, db: Session = Depends(get_db)) -> ModelData:
    """
    Retrieve a specific model by alias.

    Args:
        model_id: The server alias to retrieve
    """
    statement = select(ServerInstance).where(ServerInstance.alias == model_id)
    instance = db.exec(statement).first()

    if not instance or instance.status not in {
        "stopped",
        "starting",
        "running",
        "error",
    }:
        raise HTTPException(status_code=404, detail="Model not found")

    metadata_created = (instance.model_metadata or {}).get("created")
    created_timestamp = (
        int(metadata_created)
        if isinstance(metadata_created, int)
        else int(instance.started_at.timestamp())
        if instance.started_at
        else 0
    )
    owned_by = (instance.model_metadata or {}).get("owned_by")
    return ModelData(
        id=instance.alias,
        created=created_timestamp,
        owned_by=owned_by if isinstance(owned_by, str) else "inference-matrix",
    )

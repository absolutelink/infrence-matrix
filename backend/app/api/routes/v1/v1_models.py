"""V1 Models endpoint - OpenAI-compatible model listing (server aliases)."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, col, select

from app.api.deps import get_db
from app.models import Model, ServerInstance

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
    List available models.

    Returns the aliases of server instances that are starting or running —
    these are the names clients can use in the `model` field of
    /v1/chat/completions, /v1/completions, and /v1/embeddings.
    """
    statement = (
        select(ServerInstance)
        .where(col(ServerInstance.status).in_(["starting", "running"]))
        .join(Model, col(ServerInstance.model_id) == col(Model.id))
    )
    instances = db.exec(statement).all()

    model_data = []
    for instance in instances:
        if not instance.alias:
            continue
        created_timestamp = (
            int(instance.started_at.timestamp()) if instance.started_at else 0
        )
        model_data.append(
            ModelData(
                id=instance.alias,
                created=created_timestamp,
                owned_by=instance.model.name if instance.model else "inference-matrix",
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

    if not instance:
        raise HTTPException(status_code=404, detail="Model not found")

    created_timestamp = (
        int(instance.started_at.timestamp()) if instance.started_at else 0
    )
    return ModelData(
        id=instance.alias,
        created=created_timestamp,
        owned_by=instance.model.name if instance.model else "inference-matrix",
    )

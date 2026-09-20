"""V1 Models endpoint - OpenAI-compatible model listing."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import get_db
from app.models import Model

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
def list_models(
    db: Session = Depends(get_db),
) -> ModelsList:
    """
    List available models.

    Returns a list of models that are available for inference.
    """
    statement = select(Model)
    models = db.exec(statement).all()

    model_data = []
    for model in models:
        created_timestamp = int(model.downloaded_at.timestamp())
        model_data.append(ModelData(
            id=model.name,
            created=created_timestamp,
        ))

    return ModelsList(data=model_data)


@router.get("/models/{model_id}")
def retrieve_model(
    model_id: str,
    db: Session = Depends(get_db),
) -> ModelData:
    """
    Retrieve a specific model.

    Args:
        model_id: The ID of the model to retrieve
    """
    statement = select(Model).where(Model.name == model_id)
    model = db.exec(statement).first()

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    created_timestamp = int(model.downloaded_at.timestamp())
    return ModelData(
        id=model.name,
        created=created_timestamp,
    )

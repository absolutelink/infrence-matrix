import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlmodel import col, func, select

from app.api.deps import CurrentUser, SessionDep
from app.models import Message, Model, ModelCreate, ModelUpdate

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/", response_model=list[Model])
def read_models(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve models.
    """
    count_statement = select(func.count()).select_from(Model)
    count = session.exec(count_statement).one()
    statement = (
        select(Model).order_by(col(Model.name).asc()).offset(skip).limit(limit)
    )
    models = session.exec(statement).all()
    return models


@router.get("/{id}", response_model=Model)
def read_model(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> Any:
    """
    Get model by ID.
    """
    model = session.get(Model, id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.post("/", response_model=Model)
def create_model(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    model_in: ModelCreate,
) -> Any:
    """
    Create new model.
    """
    # Check if model with this name already exists
    existing = session.exec(select(Model).where(Model.name == model_in.name)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model with this name already exists")
    
    model = Model.model_validate(model_in)
    session.add(model)
    session.commit()
    session.refresh(model)
    return model


@router.put("/{id}", response_model=Model)
def update_model(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    model_in: ModelUpdate,
) -> Any:
    """
    Update a model.
    """
    model = session.get(Model, id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    update_dict = model_in.model_dump(exclude_unset=True)
    model.sqlmodel_update(update_dict)
    session.add(model)
    session.commit()
    session.refresh(model)
    return model


@router.delete("/{id}")
def delete_model(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> Message:
    """
    Delete a model.
    """
    model = session.get(Model, id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    session.delete(model)
    session.commit()
    return Message(message="Model deleted successfully")

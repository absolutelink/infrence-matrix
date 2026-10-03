"""V1 Models endpoint - OpenAI-compatible model listing (server aliases)."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, col, select

from app.api.deps import get_db
from app.models import ServerInstance
from app.services.npu_aliases import NPU_SUFFIXES, client_name

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
    listed_ids = set()
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
        listed_ids.add(instance.alias)

    # NPU small-model virtual aliases (<alias>-embed, <alias>-rerank, ...)
    # are discoverable per flash instance, deduped across instances.
    for instance in instances:
        if not instance.alias or getattr(instance, "engine", None) != "halogen-flash":
            continue
        for upstream_id in (instance.engine_options or {}).get("npu_models") or []:
            suffix = NPU_SUFFIXES.get(upstream_id)
            if suffix is None:
                continue
            name = client_name(instance.alias, upstream_id)
            if name in listed_ids:
                continue
            listed_ids.add(name)
            model_data.append(
                ModelData(
                    id=name,
                    created=0,
                    owned_by="inference-matrix-npu",
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

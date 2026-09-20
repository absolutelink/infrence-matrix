"""V1 Batches endpoint - OpenAI-compatible batch processing API."""

import logging
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import get_db
from app.models import BatchJob, File

logger = logging.getLogger(__name__)

router = APIRouter()


class BatchRequest(BaseModel):
    """Batch creation request."""
    input_file_id: str
    endpoint: str
    completion_window: str = "24h"


class BatchData(BaseModel):
    """Batch data for OpenAI API response."""
    id: str
    object: str = "batch"
    created_at: int
    endpoint: str
    input_file_id: str
    output_file_id: str | None = None
    results_file_id: str | None = None
    status: str
    completion_window: str
    total_requests: int = 0
    processed_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    expires_at: int | None = None
    error_message: str | None = None


class BatchesList(BaseModel):
    """List of batches response."""
    object: str = "list"
    data: list[BatchData]


def _parse_completion_window(completion_window: str) -> int:
    """Parse completion window string to hours."""
    if completion_window == "24h":
        return 24
    elif completion_window == "7d":
        return 168
    else:
        try:
            hours = int(completion_window.replace("h", ""))
            return min(max(hours, 1), 168)
        except ValueError:
            return 24


@router.post("/batches", response_model=BatchData)
def create_batch(
    request: BatchRequest,
    db: Session = Depends(get_db),
) -> BatchData:
    """
    Create a batch job for async processing.

    Args:
        request: Batch creation request
    """
    if request.endpoint not in ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid endpoint: {request.endpoint}. Must be one of: /v1/chat/completions, /v1/completions, /v1/embeddings"
        )

    try:
        file_uuid = uuid.UUID(request.input_file_id.replace("file-", ""))
    except ValueError:
        raise HTTPException(status_code=404, detail="Input file not found")

    statement = select(File).where(File.id == file_uuid)
    input_file = db.exec(statement).first()

    if not input_file:
        raise HTTPException(status_code=404, detail="Input file not found")

    if input_file.status != "uploaded":
        raise HTTPException(
            status_code=400,
            detail=f"Input file status is '{input_file.status}', expected 'uploaded'"
        )

    completion_hours = _parse_completion_window(request.completion_window)
    expires_at = datetime.now(UTC) + timedelta(hours=completion_hours)

    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    batch_uuid = uuid.UUID(batch_id.replace("batch_", ""))

    batch_job = BatchJob(
        id=batch_uuid,
        input_file_id=input_file.id,
        endpoint=request.endpoint,
        completion_window_hours=completion_hours,
        status="validating",
        expires_at=expires_at,
    )

    db.add(batch_job)
    db.commit()
    db.refresh(batch_job)

    return BatchData(
        id=str(batch_job.id),
        created_at=int(batch_job.created_at.timestamp()),
        endpoint=batch_job.endpoint,
        input_file_id=str(batch_job.input_file_id),
        status=batch_job.status,
        completion_window=request.completion_window,
        expires_at=int(batch_job.expires_at.timestamp()),
    )


@router.get("/batches", response_model=BatchesList)
def list_batches(
    status: str | None = None,
    db: Session = Depends(get_db),
) -> BatchesList:
    """
    List batch jobs.

    Args:
        status: Optional filter by status
    """
    statement = select(BatchJob)
    if status:
        statement = statement.where(BatchJob.status == status)

    batches = db.exec(statement).all()

    batch_data = []
    for batch in batches:
        batch_data.append(BatchData(
            id=str(batch.id),
            created_at=int(batch.created_at.timestamp()),
            endpoint=batch.endpoint,
            input_file_id=str(batch.input_file_id),
            output_file_id=str(batch.output_file_id) if batch.output_file_id else None,
            results_file_id=str(batch.results_file_id) if batch.results_file_id else None,
            status=batch.status,
            completion_window=f"{batch.completion_window_hours}h",
            total_requests=batch.total_requests,
            processed_requests=batch.processed_requests,
            successful_requests=batch.successful_requests,
            failed_requests=batch.failed_requests,
            expires_at=int(batch.expires_at.timestamp()) if batch.expires_at else None,
            error_message=batch.error_message,
        ))

    return BatchesList(data=batch_data)


@router.get("/batches/{batch_id}", response_model=BatchData)
def retrieve_batch(
    batch_id: str,
    db: Session = Depends(get_db),
) -> BatchData:
    """
    Retrieve information about a specific batch job.

    Args:
        batch_id: The ID of the batch to retrieve
    """
    try:
        batch_uuid = uuid.UUID(batch_id.replace("batch_", ""))
    except ValueError:
        raise HTTPException(status_code=404, detail="Batch not found")

    statement = select(BatchJob).where(BatchJob.id == batch_uuid)
    batch = db.exec(statement).first()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    return BatchData(
        id=str(batch.id),
        created_at=int(batch.created_at.timestamp()),
        endpoint=batch.endpoint,
        input_file_id=str(batch.input_file_id),
        output_file_id=str(batch.output_file_id) if batch.output_file_id else None,
        results_file_id=str(batch.results_file_id) if batch.results_file_id else None,
        status=batch.status,
        completion_window=f"{batch.completion_window_hours}h",
        total_requests=batch.total_requests,
        processed_requests=batch.processed_requests,
        successful_requests=batch.successful_requests,
        failed_requests=batch.failed_requests,
        expires_at=int(batch.expires_at.timestamp()) if batch.expires_at else None,
        error_message=batch.error_message,
    )


@router.post("/batches/{batch_id}/cancel", response_model=BatchData)
def cancel_batch(
    batch_id: str,
    db: Session = Depends(get_db),
) -> BatchData:
    """
    Cancel a batch job.

    Args:
        batch_id: The ID of the batch to cancel
    """
    try:
        batch_uuid = uuid.UUID(batch_id.replace("batch_", ""))
    except ValueError:
        raise HTTPException(status_code=404, detail="Batch not found")

    statement = select(BatchJob).where(BatchJob.id == batch_uuid)
    batch = db.exec(statement).first()

    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    if batch.status in ["completed", "cancelled", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel batch with status '{batch.status}'"
        )

    batch.status = "cancelling"
    db.add(batch)
    db.commit()
    db.refresh(batch)

    return BatchData(
        id=str(batch.id),
        created_at=int(batch.created_at.timestamp()),
        endpoint=batch.endpoint,
        input_file_id=str(batch.input_file_id),
        output_file_id=str(batch.output_file_id) if batch.output_file_id else None,
        results_file_id=str(batch.results_file_id) if batch.results_file_id else None,
        status=batch.status,
        completion_window=f"{batch.completion_window_hours}h",
        total_requests=batch.total_requests,
        processed_requests=batch.processed_requests,
        successful_requests=batch.successful_requests,
        failed_requests=batch.failed_requests,
        expires_at=int(batch.expires_at.timestamp()) if batch.expires_at else None,
        error_message=batch.error_message,
    )

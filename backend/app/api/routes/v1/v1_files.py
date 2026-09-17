"""V1 Files endpoint - OpenAI-compatible file management API."""

import hashlib
import logging
import mimetypes
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from app.api.deps import get_db
from app.core.config import settings
from app.models import File as FileModel

logger = logging.getLogger(__name__)

router = APIRouter()


class FileData(BaseModel):
    """File data for OpenAI API response."""
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: str


class FilesList(BaseModel):
    """List of files response."""
    object: str = "list"
    data: list[FileData]


class DeleteFileResponse(BaseModel):
    """Delete file response."""
    id: str
    object: str = "file"
    deleted: bool


def _compute_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def _get_mime_type(filename: str) -> str:
    """Get MIME type for a file."""
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "application/octet-stream"


@router.get("/v1/files", response_model=FilesList)
def list_files(
    purpose: str | None = None,
    db: Session = Depends(get_db),
) -> FilesList:
    """
    List uploaded files.

    Args:
        purpose: Optional filter by file purpose
    """
    statement = select(FileModel)
    if purpose:
        statement = statement.where(FileModel.purpose == purpose)

    files = db.exec(statement).all()

    file_data = []
    for file_model in files:
        file_data.append(FileData(
            id=str(file_model.id),
            bytes=file_model.size_bytes,
            created_at=int(file_model.created_at.timestamp()),
            filename=file_model.filename,
            purpose=file_model.purpose,
            status=file_model.status,
        ))

    return FilesList(data=file_data)


@router.post("/v1/files", response_model=FileData)
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = Form(...),
    db: Session = Depends(get_db),
) -> FileData:
    """
    Upload a file.

    Args:
        file: File to upload
        purpose: File purpose (batch, retrieval, assistants)
    """
    if purpose not in ["batch", "retrieval", "assistants"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid purpose: {purpose}. Must be one of: batch, retrieval, assistants"
        )

    try:
        files_dir = Path(settings.FILES_PATH if hasattr(settings, "FILES_PATH") else "/tmp/inference-matrix/files")
        files_dir.mkdir(parents=True, exist_ok=True)

        file_id = f"file-{uuid.uuid4().hex[:12]}"
        file_extension = Path(file.filename or "").suffix
        file_path = files_dir / f"{file_id}{file_extension}"

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        file_size = file_path.stat().st_size
        checksum = _compute_checksum(file_path)
        mime_type = _get_mime_type(file.filename or "")

        file_model = FileModel(
            id=uuid.UUID(file_id.replace("file-", "")),
            filename=file.filename or "unknown",
            path=str(file_path.absolute()),
            size_bytes=file_size,
            mime_type=mime_type,
            checksum_sha256=checksum,
            purpose=purpose,
            status="uploaded",
        )

        db.add(file_model)
        db.commit()
        db.refresh(file_model)

        return FileData(
            id=str(file_model.id),
            bytes=file_model.size_bytes,
            created_at=int(file_model.created_at.timestamp()),
            filename=file_model.filename,
            purpose=file_model.purpose,
            status=file_model.status,
        )

    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {e}")


@router.get("/v1/files/{file_id}", response_model=FileData)
def retrieve_file(
    file_id: str,
    db: Session = Depends(get_db),
) -> FileData:
    """
    Retrieve information about a specific file.

    Args:
        file_id: The ID of the file to retrieve
    """
    try:
        file_uuid = uuid.UUID(file_id.replace("file-", ""))
    except ValueError:
        raise HTTPException(status_code=404, detail="File not found")

    statement = select(FileModel).where(FileModel.id == file_uuid)
    file_model = db.exec(statement).first()

    if not file_model:
        raise HTTPException(status_code=404, detail="File not found")

    return FileData(
        id=str(file_model.id),
        bytes=file_model.size_bytes,
        created_at=int(file_model.created_at.timestamp()),
        filename=file_model.filename,
        purpose=file_model.purpose,
        status=file_model.status,
    )


@router.delete("/v1/files/{file_id}", response_model=DeleteFileResponse)
def delete_file(
    file_id: str,
    db: Session = Depends(get_db),
) -> DeleteFileResponse:
    """
    Delete a file.

    Args:
        file_id: The ID of the file to delete
    """
    try:
        file_uuid = uuid.UUID(file_id.replace("file-", ""))
    except ValueError:
        raise HTTPException(status_code=404, detail="File not found")

    statement = select(FileModel).where(FileModel.id == file_uuid)
    file_model = db.exec(statement).first()

    if not file_model:
        raise HTTPException(status_code=404, detail="File not found")

    try:
        file_path = Path(file_model.path)
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.error(f"Failed to delete file {file_model.path}: {e}")

    db.delete(file_model)
    db.commit()

    return DeleteFileResponse(
        id=str(file_model.id),
        deleted=True,
    )


@router.get("/v1/files/{file_id}/content")
def retrieve_file_content(
    file_id: str,
    db: Session = Depends(get_db),
):
    """
    Retrieve the content of a file.

    Args:
        file_id: The ID of the file to retrieve content for
    """
    try:
        file_uuid = uuid.UUID(file_id.replace("file-", ""))
    except ValueError:
        raise HTTPException(status_code=404, detail="File not found")

    statement = select(FileModel).where(FileModel.id == file_uuid)
    file_model = db.exec(statement).first()

    if not file_model:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = Path(file_model.path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File content not found")

    from fastapi.responses import FileResponse

    return FileResponse(
        path=str(file_path),
        filename=file_model.filename,
        media_type=file_model.mime_type,
    )

import shutil
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi import Path as PathParam

from src.api.schemas.data import ProcessResponse, ResetResponse, UploadResponse
from src.core.chunking.fixed import FixedSizeChunker
from src.core.ingestion.processing import process_project
from src.core.ingestion.storage import (
    FileTooLargeError,
    FileTypeNotSupportedError,
    save_upload,
)
from src.models.enums import ResponseStatus
from src.config import Settings, get_settings

data_router = APIRouter(prefix="/api/data", tags=["data"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
ProjectId = Annotated[str, PathParam(pattern=r"^[A-Za-z0-9_-]{1,64}$")]

SCOPES = {
    "raw": ["raw"],
    "processed": ["processed"],
    "all": ["raw", "processed", "indexes"],
}


@data_router.post("/upload/{project_id}", response_model=UploadResponse)
async def upload(project_id: ProjectId, file: UploadFile, settings: SettingsDep):
    try:
        path = await save_upload(
            file,
            Path(settings.DATA_DIR) / "raw" / project_id,
            max_bytes=settings.FILE_MAX_SIZE * 1024 * 1024,
            allowed_types=set(settings.FILE_ALLOWED_TYPES),
        )
    except FileTypeNotSupportedError:
        raise HTTPException(
            status_code=415,
            detail={"status": ResponseStatus.FILE_TYPE_NOT_SUPPORTED},
        )
    except FileTooLargeError:
        raise HTTPException(
            status_code=413,
            detail={"status": ResponseStatus.FILE_SIZE_EXCEEDED},
        )
    return UploadResponse(status=ResponseStatus.SUCCESS, stored_filename=path.name)


@data_router.post("/process/{project_id}", response_model=ProcessResponse)
def process(project_id: ProjectId, settings: SettingsDep, force: bool = False):
    raw_dir = Path(settings.DATA_DIR) / "raw" / project_id
    if not raw_dir.is_dir():
        raise HTTPException(status_code=404, detail="Project has no uploaded files")

    chunker = FixedSizeChunker(
        settings.FILE_DEFAULT_CHUNK_SIZE, settings.FILE_DEFAULT_OVERLAP_SIZE
    )
    result = process_project(
        raw_dir,
        Path(settings.DATA_DIR) / "processed" / project_id,
        project_id,
        chunker,
        force=force,
    )
    return ProcessResponse(
        status=ResponseStatus.SUCCESS,
        files_processed=result.files_processed,
        files_skipped=result.files_skipped,
        files_removed=result.files_removed,
        chunks_created=result.chunks_created,
        failed_files=result.failed_files,
    )


@data_router.delete("/{project_id}", response_model=ResetResponse)
def reset_project(
    project_id: ProjectId,
    settings: SettingsDep,
    scope: Literal["raw", "processed", "all"] = "all",
):
    deleted = []
    for folder in SCOPES[scope]:
        target = Path(settings.DATA_DIR) / folder / project_id
        if target.is_dir():
            shutil.rmtree(target)
            deleted.append(folder)
    return ResetResponse(status=ResponseStatus.SUCCESS, deleted=deleted)


@data_router.delete("", response_model=ResetResponse)
def reset_all(settings: SettingsDep, confirm: bool = False):
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="This deletes ALL data. Call again with ?confirm=true",
        )
    deleted = []
    for folder in ("raw", "processed", "indexes"):
        target = Path(settings.DATA_DIR) / folder
        if target.is_dir():
            shutil.rmtree(target)
            deleted.append(folder)
    return ResetResponse(status=ResponseStatus.SUCCESS, deleted=deleted)


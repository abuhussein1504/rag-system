from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi import Path as PathParam

from src.api.schemas.data import UploadResponse
from src.config import Settings, get_settings
from src.core.ingestion.storage import (
    FileTooLargeError,
    FileTypeNotSupportedError,
    save_upload,
)
from src.enums import ResponseStatus

data_router = APIRouter(prefix="/api/data", tags=["data"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
ProjectId = Annotated[str, PathParam(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


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


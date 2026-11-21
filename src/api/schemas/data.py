from pydantic import BaseModel

from src.models.enums import ResponseStatus


class UploadResponse(BaseModel):
    status: ResponseStatus
    stored_filename: str

class ProcessResponse(BaseModel):
    status: ResponseStatus
    files_processed: int
    files_skipped: int
    files_removed: int
    chunks_created: int
    failed_files: list[str]

class ResetResponse(BaseModel):
    status: ResponseStatus
    deleted: list[str]

class IndexResponse(BaseModel):
    status: ResponseStatus
    files_indexed: int
    chunks_indexed: int


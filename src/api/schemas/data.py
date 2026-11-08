from pydantic import BaseModel

from src.enums import ResponseStatus


class UploadResponse(BaseModel):
    status: ResponseStatus
    stored_filename: str


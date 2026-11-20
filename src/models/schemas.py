from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models.enums import RetrievalSource


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    project_id: str
    source: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    doc_id: str
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_offsets(self) -> "Chunk":
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    source: RetrievalSource


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    snippet: str


class Answer(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False


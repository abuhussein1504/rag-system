import re
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from src.models.schemas import Chunk

_UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{12}_")
MIN_OVERLAP_RATIO = 0.5


def original_name(stored_name: str) -> str:
    return _UPLOAD_PREFIX.sub("", stored_name)


class GoldItem(BaseModel):
    id: str
    question: str = Field(min_length=5)
    doc: str
    page: int | None = None
    start_char: int = Field(ge=0)
    end_char: int
    preview: str = ""

    @model_validator(mode="after")
    def _check_span(self) -> "GoldItem":
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self


def load_golden(path: Path) -> list[GoldItem]:
    lines = path.read_text(encoding="utf-8").splitlines()
    items = [GoldItem.model_validate_json(line) for line in lines if line.strip()]
    ids = [item.id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question ids in the golden set")
    return items


def is_relevant(chunk: Chunk, gold: GoldItem) -> bool:
    if original_name(chunk.metadata.get("source", "")) != gold.doc:
        return False
    if chunk.metadata.get("page") != gold.page:
        return False
    overlap = min(chunk.end_char, gold.end_char) - max(chunk.start_char, gold.start_char)
    if overlap <= 0:
        return False
    smaller = min(chunk.end_char - chunk.start_char, gold.end_char - gold.start_char)
    return overlap >= MIN_OVERLAP_RATIO * smaller


import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from src.models.schemas import Chunk
from src.core.ingestion.storage import original_name


_WORDS = re.compile(r"\w+")
SHINGLE_SIZE = 5
MIN_COVERAGE = 0.6


def _shingles(text: str) -> set[tuple[str, ...]]:
    words = _WORDS.findall(text.lower())
    if len(words) < SHINGLE_SIZE:
        return set()
    return {tuple(words[i : i + SHINGLE_SIZE]) for i in range(len(words) - SHINGLE_SIZE + 1)}


def coverage(answer: str, chunk_text: str) -> float:
    wanted = _shingles(answer)
    if not wanted:
        return 0.0
    return len(wanted & _shingles(chunk_text)) / len(wanted)


class GoldItem(BaseModel):
    id: str
    question: str = Field(min_length=5)
    doc: str
    page: int | None = None
    answers: list[str] = Field(min_length=1)

    @field_validator("answers")
    @classmethod
    def _answers_long_enough(cls, answers: list[str]) -> list[str]:
        for answer in answers:
            if len(_WORDS.findall(answer)) <= SHINGLE_SIZE:
                raise ValueError(f"Answer needs at least {SHINGLE_SIZE + 1} words: {answer!r}")
        return answers


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
    return any(coverage(answer, chunk.text) >= MIN_COVERAGE for answer in gold.answers)


def is_right_page(chunk: Chunk, gold: GoldItem) -> bool:
    return (
        gold.page is not None
        and original_name(chunk.metadata.get("source", "")) == gold.doc
        and chunk.metadata.get("page") == gold.page
    )


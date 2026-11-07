import pytest
from pydantic import ValidationError
from typing import Any

from src.utils.enums import RetrievalSource
from src.utils.schemas import Answer, Chunk, RetrievedChunk


def make_chunk(**overrides: Any) -> Chunk:
    data: dict[str, Any] = dict(id="c1", doc_id="d1", text="hello", start_char=0, end_char=5)
    data.update(overrides)
    return Chunk(**data)


def test_chunk_valid():
    chunk = make_chunk()
    assert chunk.metadata == {}


def test_chunk_is_immutable():
    chunk = make_chunk()
    with pytest.raises(ValidationError):
        chunk.text = "changed"


@pytest.mark.parametrize("text", [""])
def test_chunk_rejects_empty_text(text):
    with pytest.raises(ValidationError):
        make_chunk(text=text)


def test_chunk_rejects_bad_offsets():
    with pytest.raises(ValidationError):
        make_chunk(start_char=10, end_char=5)


def test_retrieved_chunk_source_is_enum():
    rc = RetrievedChunk(chunk=make_chunk(), score=0.9, source=RetrievalSource.DENSE)
    assert rc.source is RetrievalSource.DENSE


def test_answer_defaults():
    answer = Answer(text="hi")
    assert answer.citations == []
    assert answer.refused is False


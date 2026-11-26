from pydantic import BaseModel, Field

from src.models.enums import RetrieverName


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    retriever: RetrieverName | None = Field(
        default=None, description="Defaults to the DEFAULT_RETRIEVER setting."
    )


class AskCitation(BaseModel):
    number: int
    chunk_id: str
    doc_id: str
    source: str
    page: int | None = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[AskCitation]


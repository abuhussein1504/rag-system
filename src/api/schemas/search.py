from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str
    page: int | None = None


class SearchResponse(BaseModel):
    hits: list[SearchHit]


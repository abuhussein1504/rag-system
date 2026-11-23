from typing import Protocol

from src.models.schemas import RetrievedChunk


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]: ...


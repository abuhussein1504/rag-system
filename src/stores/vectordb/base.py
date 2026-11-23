from typing import Protocol

from src.models.schemas import Chunk, RetrievedChunk


class VectorStore(Protocol):
    def ensure_collection(self, name: str, dimension: int) -> None: ...

    def upsert(
        self, collection: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None: ...

    def search(
        self, collection: str, vector: list[float], k: int
    ) -> list[RetrievedChunk]: ...

    def count(self, collection: str) -> int: ...

    def delete_by_source(self, collection: str, source: str) -> None: ...

    def delete_collection(self, name: str) -> None: ...

    def delete_all(self) -> None: ...


from typing import Protocol

from src.models.schemas import Chunk, Document


class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]: ...

from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


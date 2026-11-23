from src.core.retrieval.base import Retriever
from src.models.enums import RetrievalSource
from src.models.schemas import RetrievedChunk
from src.stores.reranking.base import Reranker


class RerankingRetriever:
    def __init__(self, base: Retriever, reranker: Reranker, *, candidates: int = 30) -> None:
        self.base = base
        self.reranker = reranker
        self.candidates = candidates

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        found = self.base.retrieve(query, self.candidates)
        if not found:
            return []
        scores = self.reranker.score(query, [hit.chunk.text for hit in found])
        ranked = sorted(zip(found, scores, strict=True), key=lambda pair: -pair[1])
        return [
            RetrievedChunk(chunk=hit.chunk, score=score, source=RetrievalSource.RERANKED)
            for hit, score in ranked[:k]
        ]


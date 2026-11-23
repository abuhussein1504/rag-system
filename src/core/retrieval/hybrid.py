from src.core.retrieval.base import Retriever
from src.core.retrieval.fusion import reciprocal_rank_fusion
from src.models.schemas import RetrievedChunk


class HybridRetriever:
    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        *,
        dense_k: int = 30,
        sparse_k: int = 30,
        rrf_k: int = 60,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        dense_hits = self.dense.retrieve(query, self.dense_k)
        sparse_hits = self.sparse.retrieve(query, self.sparse_k)
        return reciprocal_rank_fusion([dense_hits, sparse_hits], k=self.rrf_k, top_k=k)


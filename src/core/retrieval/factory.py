from pathlib import Path

from src.core.indexing.indexer import collection_name
from src.core.retrieval.base import Retriever
from src.core.retrieval.bm25 import BM25Retriever
from src.core.retrieval.corpus import load_chunks
from src.core.retrieval.dense import DenseRetriever
from src.core.retrieval.hybrid import HybridRetriever
from src.core.retrieval.rerank import RerankingRetriever
from src.stores.embedding.base import Embedder
from src.stores.reranking.base import Reranker
from src.stores.vectordb.base import VectorStore

RETRIEVER_NAMES = ("dense", "bm25", "hybrid", "hybrid_rerank")


def build_retriever(
    name: str,
    *,
    project: str,
    processed_dir: Path,
    embedder: Embedder,
    store: VectorStore,
    reranker: Reranker | None = None,
) -> Retriever:
    dense = DenseRetriever(embedder, store, collection_name(project))
    if name == "dense":
        return dense
    sparse = BM25Retriever(load_chunks(processed_dir))
    if name == "bm25":
        return sparse
    hybrid = HybridRetriever(dense, sparse)
    if name == "hybrid":
        return hybrid
    if name == "hybrid_rerank":
        if reranker is None:
            raise ValueError("The 'hybrid_rerank' retriever needs a reranker")
        return RerankingRetriever(hybrid, reranker, candidates=30)
    raise ValueError(f"Unknown retriever '{name}'. Choose from: {', '.join(RETRIEVER_NAMES)}")


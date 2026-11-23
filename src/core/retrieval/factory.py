from pathlib import Path

from src.core.indexing.indexer import collection_name
from src.core.retrieval.base import Retriever
from src.core.retrieval.bm25 import BM25Retriever
from src.core.retrieval.corpus import load_chunks
from src.core.retrieval.dense import DenseRetriever
from src.core.retrieval.hybrid import HybridRetriever
from src.stores.embedding.base import Embedder
from src.stores.vectordb.base import VectorStore

RETRIEVER_NAMES = ("dense", "bm25", "hybrid")


def build_retriever(
    name: str,
    *,
    project: str,
    processed_dir: Path,
    embedder: Embedder,
    store: VectorStore,
) -> Retriever:
    dense = DenseRetriever(embedder, store, collection_name(project))
    if name == "dense":
        return dense
    sparse = BM25Retriever(load_chunks(processed_dir))
    if name == "bm25":
        return sparse
    if name == "hybrid":
        return HybridRetriever(dense, sparse)
    raise ValueError(f"Unknown retriever '{name}'. Choose from: {', '.join(RETRIEVER_NAMES)}")


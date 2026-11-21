from src.models.schemas import RetrievedChunk
from src.stores.embedding.base import Embedder
from src.stores.vectordb.base import VectorStore


class DenseRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore, collection: str) -> None:
        self.embedder = embedder
        self.store = store
        self.collection = collection

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        vector = self.embedder.embed_query(query)
        return self.store.search(self.collection, vector, k)


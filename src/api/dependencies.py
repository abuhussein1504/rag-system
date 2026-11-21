from functools import lru_cache

from src.config import Settings, get_settings
from src.models.enums import EmbeddingBackend

from src.stores.embedding.base import Embedder
from src.stores.embedding.gemini_embedder import GeminiEmbedder

from src.stores.vectordb.base import VectorStore
from src.stores.vectordb.qdrant import QdrantStore

from typing import Annotated

from fastapi import Depends
from fastapi import Path as PathParam


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    if s.EMBEDDING_BACKEND == EmbeddingBackend.GEMINI:
        if s.GOOGLE_API_KEY is None:
            raise RuntimeError("GOOGLE_API_KEY is required for the gemini backend")
        return GeminiEmbedder(
            s.GOOGLE_API_KEY.get_secret_value(),
            s.EMBEDDING_MODEL_ID or "gemini-embedding-001",
            s.EMBEDDING_MODEL_SIZE,
        )
    raise ValueError(f"Unsupported embedding backend: {s.EMBEDDING_BACKEND}")


@lru_cache
def get_vector_store() -> VectorStore:
    s = get_settings()
    return QdrantStore(
        s.VECTOR_DB_PATH, s.VECTOR_DB_DISTANCE_METHOD
    )


SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
StoreDep = Annotated[VectorStore, Depends(get_vector_store)]
ProjectId = Annotated[str, PathParam(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


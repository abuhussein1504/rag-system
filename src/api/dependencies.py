from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from fastapi import Path as PathParam

from src.config import Settings, get_settings
from src.core.llm.base import LLMClient
from src.core.llm.gemini import GeminiLLM
from src.models.enums import EmbeddingBackend
from src.stores.embedding.base import Embedder
from src.stores.embedding.gemini_embedder import GeminiEmbedder
from src.stores.embedding.remote_embedder import RemoteEmbedder
from src.stores.reranking.base import Reranker
from src.stores.reranking.remote_reranker import RemoteReranker
from src.stores.vectordb.base import VectorStore
from src.stores.vectordb.qdrant import QdrantStore


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
    if s.EMBEDDING_BACKEND == EmbeddingBackend.REMOTE:
        if not s.REMOTE_EMBEDDER_URL:
            raise RuntimeError("REMOTE_EMBEDDER_URL is required for the remote backend")
        return RemoteEmbedder(
            base_url=s.REMOTE_EMBEDDER_URL,
            dimension=s.EMBEDDING_MODEL_SIZE,
        )
    raise ValueError(f"Unsupported embedding backend: {s.EMBEDDING_BACKEND}")


@lru_cache
def get_vector_store() -> VectorStore:
    s = get_settings()
    return QdrantStore(
        s.VECTOR_DB_PATH, s.VECTOR_DB_DISTANCE_METHOD
    )


@lru_cache
def get_reranker() -> Reranker:
    s = get_settings()
    if not s.REMOTE_EMBEDDER_URL:
        raise RuntimeError("REMOTE_EMBEDDER_URL is required for the remote reranker")
    return RemoteReranker(s.REMOTE_EMBEDDER_URL)


@lru_cache
def get_llm() -> LLMClient:
    s = get_settings()
    if s.GOOGLE_API_KEY is None:
        raise RuntimeError("GOOGLE_API_KEY is required for answer generation")
    return GeminiLLM(
        s.GOOGLE_API_KEY.get_secret_value(),
        s.GENERATION_MODEL_ID,
        s.GENERATION_TEMPERATURE,
    )


SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
StoreDep = Annotated[VectorStore, Depends(get_vector_store)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]
ProjectId = Annotated[str, PathParam(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


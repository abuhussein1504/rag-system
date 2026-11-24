import hashlib
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from main import app
from src.api.dependencies import get_embedder, get_vector_store
from src.config import get_settings
from src.models.enums import DistanceMethod
from src.stores.vectordb.qdrant import QdrantStore


class FakeEmbedder:
    dimension = 8

    def _vec(self, text: str) -> list[float]:
        raw = [b / 255 for b in hashlib.sha256(text.encode()).digest()[:8]]
        norm = sum(x * x for x in raw) ** 0.5
        return [x / norm for x in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture(autouse=True)
def block_real_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("Tests must not use the network. Override get_embedder.")

    monkeypatch.setattr("urllib.request.urlopen", refuse)


@pytest.fixture
def client(tmp_path):
    fake_settings = SimpleNamespace(
        DATA_DIR=str(tmp_path),
        FILE_MAX_SIZE=1,  # 1 MB
        FILE_ALLOWED_TYPES=["txt", "md"],
        APP_NAME="test",
        APP_VERSION="0",
        FILE_DEFAULT_CHUNK_SIZE=200,
        FILE_DEFAULT_OVERLAP_SIZE=20,
        DEFAULT_RETRIEVER="dense",
    )
    store = QdrantStore(str(tmp_path / "qdrant"), DistanceMethod.COSINE)

    app.dependency_overrides[get_settings] = lambda: fake_settings
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_vector_store] = lambda: store
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


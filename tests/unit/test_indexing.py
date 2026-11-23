import json

from src.core.chunking.fixed import FixedSizeChunker
from src.core.indexing.indexer import collection_name, index_project
from src.core.ingestion.processing import process_project
from src.core.retrieval.dense import DenseRetriever
from src.models.enums import DistanceMethod
from src.stores.vectordb.qdrant import QdrantStore
from tests.conftest import FakeEmbedder


def prepare(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text("cats are small animals. " * 20, encoding="utf-8")
    processed = tmp_path / "processed"
    process_project(raw, processed, "p1", FixedSizeChunker(100, 10))
    store = QdrantStore(str(tmp_path / "qdrant"), DistanceMethod.COSINE)
    return processed, store, FakeEmbedder()


def chunk_lines(processed):
    return next(processed.glob("*.jsonl")).read_text(encoding="utf-8").splitlines()


def test_index_then_retrieve_finds_exact_chunk(tmp_path):
    processed, store, emb = prepare(tmp_path)
    result = index_project(processed, collection_name("p1"), emb, store)
    assert result.chunks_indexed > 0

    text = json.loads(chunk_lines(processed)[0])["text"]
    hits = DenseRetriever(emb, store, collection_name("p1")).retrieve(text, 3)
    assert hits[0].chunk.text == text
    assert hits[0].score > 0.99


def test_reindex_does_not_duplicate(tmp_path):
    processed, store, emb = prepare(tmp_path)
    index_project(processed, collection_name("p1"), emb, store)
    index_project(processed, collection_name("p1"), emb, store)
    assert store.count(collection_name("p1")) == len(chunk_lines(processed))


def test_delete_collection(tmp_path):
    processed, store, emb = prepare(tmp_path)
    index_project(processed, collection_name("p1"), emb, store)
    store.delete_collection(collection_name("p1"))
    assert store.count(collection_name("p1")) == 0
    assert store.search(collection_name("p1"), emb.embed_query("x"), 3) == []


def test_count_of_missing_collection_is_zero(tmp_path):
    store = QdrantStore(str(tmp_path / "qdrant"), DistanceMethod.COSINE)
    assert store.count("nope") == 0


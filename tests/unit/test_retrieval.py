import pytest

from scripts.diff_reports import compare, paired_bootstrap
from src.core.retrieval.bm25 import BM25Index, BM25Retriever, tokenize
from src.core.retrieval.factory import build_retriever
from src.core.retrieval.fusion import reciprocal_rank_fusion
from src.core.retrieval.hybrid import HybridRetriever
from src.models.enums import DistanceMethod, RetrievalSource
from src.models.schemas import Chunk, RetrievedChunk
from src.stores.vectordb.qdrant import QdrantStore
from tests.conftest import FakeEmbedder


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(id=chunk_id, doc_id="d", text=text, start_char=0, end_char=len(text))


def hit(chunk_id: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=make_chunk(chunk_id, f"text {chunk_id}"), score=score, source=RetrievalSource.DENSE
    )


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Special Case Pattern, in 2026!") == ["special", "case", "pattern", "2026"]


def test_bm25_ranks_the_document_with_the_rare_term_first():
    texts = ["budget app saves money", "the special case pattern removes null checks",
             "money money money"]
    ranked = BM25Index(texts).search("special case pattern", 3)
    assert ranked[0][0] == 1
    assert len(ranked) == 1


def test_bm25_prefers_the_shorter_document_for_the_same_term():
    texts = ["budget " + "filler " * 40, "budget filler"]
    ranked = BM25Index(texts).search("budget", 2)
    assert [doc for doc, _ in ranked] == [1, 0]


def test_bm25_empty_query_and_empty_corpus():
    assert BM25Index(["some text"]).search("", 5) == []
    assert BM25Index([]).search("anything", 5) == []


def test_bm25_retriever_returns_sparse_hits():
    chunks = [make_chunk("a", "cats purr"), make_chunk("b", "dogs bark")]
    hits = BM25Retriever(chunks).retrieve("dogs", 5)
    assert [h.chunk.id for h in hits] == ["b"]
    assert hits[0].source is RetrievalSource.SPARSE


def test_rrf_rewards_chunks_found_by_both_lists():
    fused = reciprocal_rank_fusion([[hit("a"), hit("b")], [hit("c"), hit("b")]], k=60)
    assert fused[0].chunk.id == "b"
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 62)
    assert all(h.source is RetrievalSource.HYBRID for h in fused)


def test_rrf_ties_keep_first_seen_order_and_top_k_works():
    fused = reciprocal_rank_fusion([[hit("a"), hit("b")], [hit("c")]], k=60, top_k=2)
    assert [h.chunk.id for h in fused] == ["a", "c"]


def test_hybrid_merges_dense_and_sparse():
    class Fixed:
        def __init__(self, ids):
            self.ids = ids

        def retrieve(self, query, k):
            return [hit(i) for i in self.ids][:k]

    hybrid = HybridRetriever(Fixed(["a", "b"]), Fixed(["b", "c"]), dense_k=5, sparse_k=5)
    assert [h.chunk.id for h in hybrid.retrieve("q", 3)][0] == "b"


def test_factory_builds_each_retriever(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    chunk = make_chunk("a", "cats purr")
    (processed / "f.jsonl").write_text(chunk.model_dump_json() + "\n", encoding="utf-8")
    store = QdrantStore(str(tmp_path / "qdrant"), DistanceMethod.COSINE)

    for name in ("dense", "bm25", "hybrid"):
        retriever = build_retriever(
            name, project="p1", processed_dir=processed, embedder=FakeEmbedder(), store=store
        )
        assert hasattr(retriever, "retrieve")
    with pytest.raises(ValueError):
        build_retriever(
            "magic", project="p1", processed_dir=processed, embedder=FakeEmbedder(), store=store
        )


def _report(name, ranks):
    return {"name": name, "questions": [
        {"id": f"q{i}", "question": f"question {i}", "rank": r} for i, r in enumerate(ranks)
    ]}


def test_compare_counts_better_and_worse():
    result = compare(_report("a", [1, 2, None, 3]), _report("b", [1, 1, 4, 5]))
    assert len(result["improved"]) == 2
    assert len(result["worsened"]) == 1
    assert result["same"] == 1


def test_bootstrap_interval_contains_zero_for_identical_runs():
    values = [1.0, 0.5, 0.0, 0.25] * 10
    low, high = paired_bootstrap(values, values)
    assert low == high == 0.0


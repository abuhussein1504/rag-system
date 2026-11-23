import json
import urllib.error
from email.message import Message

import pytest

from src.core.retrieval.factory import build_retriever
from src.core.retrieval.rerank import RerankingRetriever
from src.models.enums import DistanceMethod, RetrievalSource
from src.models.schemas import Chunk, RetrievedChunk
from src.stores.reranking.remote_reranker import RemoteReranker
from src.stores.vectordb.qdrant import QdrantStore
from tests.conftest import FakeEmbedder


def hit(chunk_id: str, text: str) -> RetrievedChunk:
    chunk = Chunk(id=chunk_id, doc_id="d", text=text, start_char=0, end_char=len(text))
    return RetrievedChunk(chunk=chunk, score=0.1, source=RetrievalSource.HYBRID)


class FixedRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.asked_for = None

    def retrieve(self, query, k):
        self.asked_for = k
        return self.hits[:k]


class KeywordReranker:
    """Scores a text by how many times it contains the word 'answer'."""

    def score(self, query, texts):
        return [float(t.count("answer")) for t in texts]


def test_reranker_reorders_and_cuts_to_k():
    base = FixedRetriever([hit("a", "no"), hit("b", "answer answer"), hit("c", "answer")])
    retriever = RerankingRetriever(base, KeywordReranker(), candidates=3)
    result = retriever.retrieve("question", 2)
    assert [h.chunk.id for h in result] == ["b", "c"]
    assert all(h.source is RetrievalSource.RERANKED for h in result)
    assert result[0].score == 2.0


def test_reranker_asks_the_base_for_many_candidates():
    base = FixedRetriever([hit("a", "x")])
    RerankingRetriever(base, KeywordReranker(), candidates=30).retrieve("q", 5)
    assert base.asked_for == 30


def test_reranker_with_no_candidates_returns_nothing():
    assert RerankingRetriever(FixedRetriever([]), KeywordReranker()).retrieve("q", 5) == []


def test_equal_scores_keep_the_original_order():
    base = FixedRetriever([hit("a", "x"), hit("b", "y"), hit("c", "z")])
    result = RerankingRetriever(base, KeywordReranker()).retrieve("q", 3)
    assert [h.chunk.id for h in result] == ["a", "b", "c"]


class FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def install(monkeypatch, handler):
    calls = []

    def fake_urlopen(req, timeout=None):
        payload = json.loads(req.data)
        calls.append((req.full_url, payload))
        return handler(payload, len(calls))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return calls


def test_remote_reranker_sends_query_and_texts(monkeypatch):
    calls = install(
        monkeypatch, lambda p, n: FakeResponse({"scores": [0.5] * len(p["texts"])})
    )
    scores = RemoteReranker("http://x/").score("what?", ["a", "b"])
    assert scores == [0.5, 0.5]
    assert calls[0][0] == "http://x/rerank"
    assert calls[0][1] == {"query": "what?", "texts": ["a", "b"]}


def test_remote_reranker_makes_no_call_for_no_texts(monkeypatch):
    calls = install(monkeypatch, lambda p, n: FakeResponse({"scores": []}))
    assert RemoteReranker("http://x").score("q", []) == []
    assert calls == []


def test_remote_reranker_retries_on_503(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)

    def handler(payload, call_number):
        if call_number < 3:
            raise urllib.error.HTTPError("http://x", 503, "unavailable", Message(), None)
        return FakeResponse({"scores": [1.0]})

    calls = install(monkeypatch, handler)
    assert RemoteReranker("http://x").score("q", ["a"]) == [1.0]
    assert len(calls) == 3


def test_remote_reranker_rejects_a_wrong_number_of_scores(monkeypatch):
    install(monkeypatch, lambda p, n: FakeResponse({"scores": [1.0]}))
    with pytest.raises(RuntimeError):
        RemoteReranker("http://x").score("q", ["a", "b"])


def test_factory_builds_hybrid_rerank(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    chunk = Chunk(id="a", doc_id="d", text="cats purr", start_char=0, end_char=9)
    (processed / "f.jsonl").write_text(chunk.model_dump_json() + "\n", encoding="utf-8")
    store = QdrantStore(str(tmp_path / "qdrant"), DistanceMethod.COSINE)
    kwargs = {"project": "p1", "processed_dir": processed, "embedder": FakeEmbedder(),
              "store": store}

    assert hasattr(build_retriever("hybrid_rerank", reranker=KeywordReranker(), **kwargs),
                   "retrieve")
    with pytest.raises(ValueError):
        build_retriever("hybrid_rerank", **kwargs)


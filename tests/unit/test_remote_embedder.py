import json
import urllib.error
from email.message import Message

from src.stores.embedding.remote_embedder import RemoteEmbedder


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
        payload = json.loads(req.data) if req.data else None
        calls.append((req.full_url, payload))
        return handler(payload, len(calls))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return calls


def ok(payload, call_number):
    return FakeResponse({"embeddings": [[0.1, 0.2]] * len(payload["texts"])})


def test_documents_are_batched(monkeypatch):
    calls = install(monkeypatch, ok)
    vectors = RemoteEmbedder("http://x", batch_size=2).embed_documents(["a", "b", "c"])
    assert len(vectors) == 3
    assert len(calls) == 2


def test_query_uses_query_task_type(monkeypatch):
    calls = install(monkeypatch, ok)
    RemoteEmbedder("http://x").embed_query("hello")
    assert calls[0][1]["task_type"] == "RETRIEVAL_QUERY"


def test_retries_on_503(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)

    def handler(payload, call_number):
        if call_number < 3:
            raise urllib.error.HTTPError("http://x", 503, "unavailable", Message(), None)
        return ok(payload, call_number)

    calls = install(monkeypatch, handler)
    assert len(RemoteEmbedder("http://x").embed_query("x")) == 2
    assert len(calls) == 3


def test_does_not_retry_on_400(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)

    def handler(payload, call_number):
        raise urllib.error.HTTPError("http://x", 400, "bad request", Message(), None)

    calls = install(monkeypatch, handler)
    try:
        RemoteEmbedder("http://x").embed_query("x")
        raise AssertionError("should have failed")
    except urllib.error.HTTPError:
        pass
    assert len(calls) == 1


def test_wrong_number_of_vectors_is_an_error(monkeypatch):
    install(monkeypatch, lambda payload, n: FakeResponse({"embeddings": [[0.1]]}))
    try:
        RemoteEmbedder("http://x").embed_documents(["a", "b"])
        raise AssertionError("should have failed")
    except RuntimeError:
        pass


def test_info_reads_health(monkeypatch):
    install(monkeypatch, lambda payload, n: FakeResponse({"model": "m", "dimension": 768}))
    assert RemoteEmbedder("http://x").info()["model"] == "m"


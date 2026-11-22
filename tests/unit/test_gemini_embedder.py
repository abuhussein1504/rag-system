from types import SimpleNamespace

from src.models.retry import retry
from src.stores.embedding.gemini_embedder import GeminiEmbedder


class FakeClient:
    def __init__(self):
        self.calls = []
        self.models = self

    def embed_content(self, model, contents, config):
        self.calls.append((list(contents), config.task_type))
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1, 0.2]) for _ in contents]
        )


def test_documents_are_batched_and_use_document_task():
    client = FakeClient()
    emb = GeminiEmbedder("key", dimension=2, batch_size=2, client=client)
    vectors = emb.embed_documents(["a", "b", "c"])
    assert len(vectors) == 3
    assert [len(texts) for texts, _ in client.calls] == [2, 1]
    assert all(task == "RETRIEVAL_DOCUMENT" for _, task in client.calls)


def test_query_uses_query_task():
    client = FakeClient()
    emb = GeminiEmbedder("key", dimension=2, client=client)
    assert len(emb.embed_query("hello")) == 2
    assert client.calls[0][1] == "RETRIEVAL_QUERY"


def test_retry_succeeds_after_failures(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("temporary")
        return "ok"

    assert retry(flaky, should_retry=lambda e: True) == "ok"
    assert len(attempts) == 3


def test_retry_gives_up_on_non_retryable_error():
    def broken():
        raise ValueError("bad input")

    try:
        retry(broken, should_retry=lambda e: False)
        raise AssertionError("should have raised")
    except ValueError:
        pass


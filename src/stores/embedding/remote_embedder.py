import json
import urllib.error
import urllib.request
from typing import Any

from src.models.retry import retry


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, urllib.error.URLError | TimeoutError | ConnectionError)


class RemoteEmbedder:
    def __init__(
        self,
        base_url: str,
        dimension: int = 768,
        timeout: int = 120,
        batch_size: int = 64,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.dimension = dimension
        self.timeout = timeout
        self.batch_size = batch_size

    def _request(self, url: str, payload: dict[str, Any] | None = None) -> Any:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _post(self, texts: list[str], task_type: str) -> list[list[float]]:
        body = retry(
            lambda: self._request(
                f"{self.base_url}/embed", {"texts": texts, "task_type": task_type}
            ),
            should_retry=_is_retryable,
        )
        embeddings: list[list[float]] = body["embeddings"]
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Remote embedder returned {len(embeddings)} vectors for {len(texts)} texts"
            )
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._post(texts[i : i + self.batch_size], "RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._post([text], "RETRIEVAL_QUERY")[0]

    def info(self) -> dict[str, Any]:
        result: dict[str, Any] = self._request(f"{self.base_url}/health")
        return result


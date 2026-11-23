import json
import urllib.error
import urllib.request
from typing import Any

from src.models.retry import retry


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, urllib.error.URLError | TimeoutError | ConnectionError)


class RemoteReranker:
    def __init__(self, base_url: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, payload: dict[str, Any]) -> Any:
        req = urllib.request.Request(
            f"{self.base_url}/rerank",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        body = retry(
            lambda: self._post({"query": query, "texts": texts}),
            should_retry=_is_retryable,
        )
        scores: list[float] = body["scores"]
        if len(scores) != len(texts):
            raise RuntimeError(f"Reranker returned {len(scores)} scores for {len(texts)} texts")
        return scores


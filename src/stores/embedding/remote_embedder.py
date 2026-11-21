import urllib.request
import json
from typing import Any


class RemoteEmbedder:
    def __init__(
        self,
        base_url: str,
        dimension: int = 768,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.dimension = dimension
        self.timeout = timeout

    def _post(self, texts: list[str], task_type: str) -> list[list[float]]:
        payload = json.dumps({"texts": texts, "task_type": task_type}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body: Any = json.loads(resp.read())
        embeddings: list[list[float]] = body["embeddings"]
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Remote embedder returned {len(embeddings)} vectors for {len(texts)} texts"
            )
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._post(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._post([text], "RETRIEVAL_QUERY")[0]


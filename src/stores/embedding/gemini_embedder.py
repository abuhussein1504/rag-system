from typing import Any

from google import genai
from google.genai import errors, types

from src.models.retry import retry


def _is_retryable(exc: Exception) -> bool:
    return isinstance(exc, errors.APIError) and (
        exc.code == 429 or (exc.code or 0) >= 500
    )


class GeminiEmbedder:
    def __init__(
        self,
        api_key: str,
        model_id: str = "gemini-embedding-001",
        dimension: int = 768,
        batch_size: int = 32,
        client: Any = None,  # lets tests pass a fake client
    ) -> None:
        self.client = client or genai.Client(api_key=api_key)
        self.model_id = model_id
        self.dimension = dimension
        self.batch_size = batch_size

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        config = types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self.dimension
        )
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = retry(
                lambda: self.client.models.embed_content(
                    model=self.model_id, contents=batch, config=config
                ),
                should_retry=_is_retryable,
            )
            batch_vectors = [
                list(e.values) for e in (response.embeddings or []) if e.values
            ]
            if len(batch_vectors) != len(batch):
                raise RuntimeError("Embedding API returned a wrong number of vectors")
            vectors.extend(batch_vectors)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


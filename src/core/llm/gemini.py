from functools import partial
from typing import Any

from google import genai
from google.genai import errors, types

from src.core.llm.base import LLMError
from src.models.retry import retry


def _is_retryable(exc: Exception) -> bool:
    return isinstance(exc, errors.APIError) and (
        exc.code == 429 or (exc.code or 0) >= 500
    )


class GeminiLLM:
    def __init__(
        self,
        api_key: str,
        model_id: str,
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        self.client = client or genai.Client(api_key=api_key)
        self.model_id = model_id
        self.temperature = temperature

    def generate(self, system: str, prompt: str) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=self.temperature
        )
        try:
            response = retry(
                partial(
                    self.client.models.generate_content,
                    model=self.model_id,
                    contents=prompt,
                    config=config,
                ),
                should_retry=_is_retryable,
            )
        except errors.APIError as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc
        text = (response.text or "").strip()
        if not text:
            raise LLMError("Gemini returned an empty answer (it may have been blocked)")
        return text
    

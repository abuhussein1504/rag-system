import json
import re
import urllib.error
import urllib.request
from functools import partial
from typing import Any

from src.core.llm.base import LLMError
from src.models.retry import retry

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
MAX_WAIT_FOR_RETRY = 90  # seconds


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    value = exc.headers.get("Retry-After") if exc.headers else None
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 429:
            wait = _retry_after(exc)
            return wait is None or wait <= MAX_WAIT_FOR_RETRY
        return exc.code >= 500
    return isinstance(exc, urllib.error.URLError | TimeoutError | ConnectionError)


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")[:300]
    except Exception:
        return ""


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 120,
        reasoning_effort: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort

    def _post(self, payload: dict[str, Any]) -> Any:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "rag-system/0.1",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def generate(self, system: str, prompt: str) -> str:
        payload = {
            "model": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        try:
            body = retry(
                partial(self._post, payload),
                should_retry=_is_retryable,
                attempts=6,
                base_delay=3.0,
            )
        except urllib.error.HTTPError as exc:
            raise LLMError(f"LLM request failed ({exc.code}): {_error_detail(exc)}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        try:
            content = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response format: {str(body)[:200]}") from exc
        text = _THINK.sub("", content).strip()
        if not text:
            raise LLMError("The model returned an empty answer (try a larger max_tokens)")
        return text

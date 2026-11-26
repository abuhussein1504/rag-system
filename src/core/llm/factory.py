from typing import Any

from src.core.llm.base import LLMClient
from src.core.llm.gemini import GeminiLLM
from src.core.llm.openai_compat import OpenAICompatibleLLM
from src.models.enums import GenerationBackend


def build_llm(
    backend: GenerationBackend,
    *,
    model_id: str,
    api_key: str | None,
    base_url: str | None,
    temperature: float,
    reasoning_effort: str | None = None,
) -> LLMClient:
    if backend == GenerationBackend.GEMINI:
        if not api_key:
            raise RuntimeError("An API key is required for the gemini backend")
        return GeminiLLM(api_key, model_id, temperature)
    if backend == GenerationBackend.OPENAI_COMPATIBLE:
        if not api_key or not base_url:
            raise RuntimeError("openai_compatible needs both an API key and a base URL")
        return OpenAICompatibleLLM(
            base_url,
            api_key,
            model_id,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
    raise ValueError(f"Unsupported LLM backend: {backend}")


def llm_from_settings(s: Any, *, judge: bool = False) -> LLMClient:
    gen_backend = s.GENERATION_BACKEND
    backend = (s.JUDGE_BACKEND or gen_backend) if judge else gen_backend
    inherits = (not judge) or backend == gen_backend 

    secret = (s.JUDGE_API_KEY if judge else None) or (s.GENERATION_API_KEY if inherits else None)
    if secret is None and backend == GenerationBackend.GEMINI:
        secret = s.GOOGLE_API_KEY
    base_url = (s.JUDGE_BASE_URL if judge else None) or (
        s.GENERATION_BASE_URL if inherits else None
    )
    model = (s.JUDGE_MODEL_ID or s.GENERATION_MODEL_ID) if judge else s.GENERATION_MODEL_ID

    return build_llm(
        backend,
        model_id=model,
        api_key=secret.get_secret_value() if secret else None,
        base_url=base_url,
        temperature=0.0 if judge else s.GENERATION_TEMPERATURE,
        reasoning_effort=getattr(
            s, "JUDGE_REASONING_EFFORT" if judge else "GENERATION_REASONING_EFFORT", None
        ),
    )

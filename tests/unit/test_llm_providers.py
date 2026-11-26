import json
import urllib.error
from email.message import Message
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from src.core.llm.base import LLMError
from src.core.llm.factory import build_llm, llm_from_settings
from src.core.llm.gemini import GeminiLLM, _is_retryable
from src.core.llm.openai_compat import OpenAICompatibleLLM
from src.models.enums import GenerationBackend

# ---- the OpenAI-compatible client (fake network)


class FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def reply(text):
    return FakeResponse({"choices": [{"message": {"content": text}}]})


def install(monkeypatch, handler):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        return handler(req, len(calls))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    return calls


def http_error(code: int, headers: dict[str, str] | None = None) -> urllib.error.HTTPError:
    msg = Message()
    for key, value in (headers or {}).items():
        msg[key] = value
    return urllib.error.HTTPError("http://x", code, "error", msg, None)


def make(**kwargs):
    return OpenAICompatibleLLM("http://api.test/v1/", "secret", "some-model", **kwargs)


def test_request_has_the_openai_chat_shape(monkeypatch):
    calls = install(monkeypatch, lambda req, n: reply("  An answer.  "))
    assert make(temperature=0.2, max_tokens=99).generate("be brief", "question?") == "An answer."

    req = calls[0]
    assert req.full_url == "http://api.test/v1/chat/completions"
    assert req.get_header("Authorization") == "Bearer secret"
    assert req.get_header("User-agent") == "rag-system/0.1"
    body = json.loads(req.data)
    assert body["model"] == "some-model"
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 99
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "question?"},
    ]


def test_reasoning_effort_is_sent_only_when_set(monkeypatch):
    calls = install(monkeypatch, lambda req, n: reply("ok"))
    make().generate("s", "p")
    make(reasoning_effort="low").generate("s", "p")
    assert "reasoning_effort" not in json.loads(calls[0].data)
    assert json.loads(calls[1].data)["reasoning_effort"] == "low"


def test_the_token_budget_leaves_room_for_reasoning_models():
    assert make().max_tokens >= 4096


def test_reasoning_blocks_are_removed(monkeypatch):
    install(monkeypatch, lambda req, n: reply("<think>hmm\nlet me see</think>\nThe answer [1]."))
    assert make().generate("s", "p") == "The answer [1]."


def test_short_rate_limits_and_server_errors_are_retried(monkeypatch):
    def handler(req, n):
        if n == 1:
            raise http_error(429, {"Retry-After": "5"})
        if n == 2:
            raise http_error(503)
        return reply("ok")

    calls = install(monkeypatch, handler)
    assert make().generate("s", "p") == "ok"
    assert len(calls) == 3


def test_a_daily_quota_is_not_retried(monkeypatch):
    calls = install(monkeypatch, lambda req, n: (_ for _ in ()).throw(
        http_error(429, {"Retry-After": "3600"})))
    with pytest.raises(LLMError, match="429"):
        make().generate("s", "p")
    assert len(calls) == 1


def test_client_errors_become_llm_errors_without_retry(monkeypatch):
    calls = install(monkeypatch, lambda req, n: (_ for _ in ()).throw(http_error(401)))
    with pytest.raises(LLMError, match="401"):
        make().generate("s", "p")
    assert len(calls) == 1


@pytest.mark.parametrize("body", [{"choices": []}, {"error": "x"}, {"choices": [{"message": {}}]}])
def test_unexpected_or_empty_responses_are_llm_errors(monkeypatch, body):
    install(monkeypatch, lambda req, n: FakeResponse(body))
    with pytest.raises(LLMError):
        make().generate("s", "p")


# ---- Gemini: daily quota fails fast

def test_gemini_daily_quota_is_not_retried_but_short_limits_are():
    daily = {"error": {"message": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}}
    from google.genai import errors

    assert not _is_retryable(errors.APIError(429, daily))
    assert _is_retryable(errors.APIError(429, {"error": {"message": "slow down"}}))
    assert _is_retryable(errors.APIError(503, {}))


# ---- choosing the provider from settings

def settings(**overrides):
    base = {
        "GENERATION_BACKEND": GenerationBackend.GEMINI,
        "GENERATION_MODEL_ID": "gen-model",
        "GENERATION_TEMPERATURE": 0.3,
        "GENERATION_BASE_URL": None,
        "GENERATION_API_KEY": None,
        "GOOGLE_API_KEY": SecretStr("google-key"),
        "JUDGE_BACKEND": None,
        "JUDGE_MODEL_ID": None,
        "JUDGE_BASE_URL": None,
        "JUDGE_API_KEY": None,
    }
    return SimpleNamespace(**{**base, **overrides})


def test_gemini_is_the_default_and_uses_the_google_key():
    llm = llm_from_settings(settings())
    assert isinstance(llm, GeminiLLM)
    assert llm.model_id == "gen-model"


def test_reasoning_effort_comes_from_the_role_specific_setting():
    s = settings(
        GENERATION_BACKEND=GenerationBackend.OPENAI_COMPATIBLE,
        GENERATION_BASE_URL="http://api.test/v1",
        GENERATION_API_KEY=SecretStr("k"),
        GENERATION_REASONING_EFFORT="low",
        JUDGE_REASONING_EFFORT="high",
    )
    gen = llm_from_settings(s)
    judge = llm_from_settings(s, judge=True)
    assert isinstance(gen, OpenAICompatibleLLM)
    assert isinstance(judge, OpenAICompatibleLLM)
    assert gen.reasoning_effort == "low"
    assert judge.reasoning_effort == "high"


def test_openai_compatible_generator():
    s = settings(
        GENERATION_BACKEND=GenerationBackend.OPENAI_COMPATIBLE,
        GENERATION_BASE_URL="http://api.test/v1",
        GENERATION_API_KEY=SecretStr("k"),
    )
    llm = llm_from_settings(s)
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.temperature == 0.3


def test_judge_inherits_everything_when_nothing_is_set():
    llm = llm_from_settings(settings(), judge=True)
    assert isinstance(llm, GeminiLLM)
    assert llm.model_id == "gen-model"
    assert llm.temperature == 0.0


def test_judge_can_use_a_different_provider():
    s = settings(
        JUDGE_BACKEND=GenerationBackend.OPENAI_COMPATIBLE,
        JUDGE_MODEL_ID="judge-model",
        JUDGE_BASE_URL="http://judge.test/v1",
        JUDGE_API_KEY=SecretStr("judge-key"),
    )
    judge = llm_from_settings(s, judge=True)
    assert isinstance(judge, OpenAICompatibleLLM)
    assert (judge.model_id, judge.base_url, judge.api_key) == (
        "judge-model", "http://judge.test/v1", "judge-key")
    assert isinstance(llm_from_settings(s), GeminiLLM)


def test_a_judge_on_another_provider_does_not_reuse_the_generators_key():
    s = settings(
        GENERATION_BACKEND=GenerationBackend.OPENAI_COMPATIBLE,
        GENERATION_BASE_URL="http://gen.test/v1",
        GENERATION_API_KEY=SecretStr("gen-key"),
        JUDGE_BACKEND=GenerationBackend.OPENAI_COMPATIBLE,
        JUDGE_BASE_URL="http://judge.test/v1",
    )
    inherited = llm_from_settings(s, judge=True)
    assert isinstance(inherited, OpenAICompatibleLLM)
    assert inherited.api_key == "gen-key"


@pytest.mark.parametrize("kwargs", [
    {"api_key": None, "base_url": "http://x"},
    {"api_key": "k", "base_url": None},
])
def test_openai_compatible_needs_key_and_url(kwargs):
    with pytest.raises(RuntimeError):
        build_llm(GenerationBackend.OPENAI_COMPATIBLE, model_id="m", temperature=0.0, **kwargs)


def test_gemini_needs_a_key():
    with pytest.raises(RuntimeError):
        build_llm(GenerationBackend.GEMINI, model_id="m", api_key=None, base_url=None,
                  temperature=0.0)


def test_unsupported_backend():
    with pytest.raises(ValueError):
        build_llm(GenerationBackend.OLLAMA, model_id="m", api_key="k", base_url="u",
                  temperature=0.0)
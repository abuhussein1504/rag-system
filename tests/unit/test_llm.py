from types import SimpleNamespace

import pytest

from src.core.llm.base import LLMError
from src.core.llm.gemini import GeminiLLM


class FakeGeminiClient:
    def __init__(self, text):
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate)
        self._text = text

    def _generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self._text)


def test_gemini_sends_prompt_and_system_instruction():
    client = FakeGeminiClient("  hello  ")
    llm = GeminiLLM("key", "gemini-test", temperature=0.2, client=client)
    assert llm.generate("be brief", "question?") == "hello"
    call = client.calls[0]
    assert call["model"] == "gemini-test"
    assert call["contents"] == "question?"
    assert call["config"].system_instruction == "be brief"
    assert call["config"].temperature == 0.2


@pytest.mark.parametrize("text", [None, "", "   "])
def test_gemini_empty_answer_is_an_llm_error(text):
    with pytest.raises(LLMError):
        GeminiLLM("key", "m", client=FakeGeminiClient(text)).generate("s", "p")
    

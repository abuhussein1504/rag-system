from typing import Protocol


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    def generate(self, system: str, prompt: str) -> str: ...


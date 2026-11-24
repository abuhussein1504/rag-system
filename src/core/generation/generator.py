import logging
import re

from src.core.generation.prompt import REFUSAL_TEXT, SYSTEM_PROMPT, build_prompt
from src.core.llm.base import LLMClient
from src.core.retrieval.base import Retriever
from src.models.schemas import Answer, Citation, RetrievedChunk

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(
    text: str, hits: list[RetrievedChunk], *, snippet_chars: int = 200
) -> list[Citation]:
    numbers: list[int] = []
    for match in _CITATION.finditer(text):
        for raw in match.group(1).split(","):
            n = int(raw)
            if 1 <= n <= len(hits) and n not in numbers:
                numbers.append(n)
    citations = []
    for n in numbers:
        chunk = hits[n - 1].chunk
        citations.append(
            Citation(
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                snippet=chunk.text[:snippet_chars],
                source=str(chunk.metadata.get("source", "")),
                page=chunk.metadata.get("page"),
            )
        )
    return citations


def _is_refusal(text: str) -> bool:
    return text.casefold().startswith(REFUSAL_TEXT.casefold().rstrip("."))


class AnswerGenerator:
    def __init__(self, retriever: Retriever, llm: LLMClient, *, snippet_chars: int = 200) -> None:
        self.retriever = retriever
        self.llm = llm
        self.snippet_chars = snippet_chars

    def answer(self, question: str, k: int = 5) -> Answer:
        hits = self.retriever.retrieve(question, k)
        if not hits:
            return Answer(text=REFUSAL_TEXT, refused=True)

        text = self.llm.generate(SYSTEM_PROMPT, build_prompt(question, hits)).strip()
        if _is_refusal(text):
            return Answer(text=REFUSAL_TEXT, refused=True)

        citations = extract_citations(text, hits, snippet_chars=self.snippet_chars)
        if not citations:
            logger.warning("The model answered without citing any passage")
        return Answer(text=text, citations=citations)


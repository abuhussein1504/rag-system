import pytest

from src.core.generation.generator import AnswerGenerator, extract_citations
from src.core.generation.prompt import REFUSAL_TEXT, SYSTEM_PROMPT, build_prompt
from src.core.retrieval.factory import RETRIEVER_NAMES
from src.models.enums import RetrievalSource, RetrieverName
from src.models.schemas import Chunk, RetrievedChunk


def hit(chunk_id: str, text: str, **metadata) -> RetrievedChunk:
    chunk = Chunk(
        id=chunk_id, doc_id="doc", text=text, start_char=0, end_char=len(text), metadata=metadata
    )
    return RetrievedChunk(chunk=chunk, score=1.0, source=RetrievalSource.DENSE)


class FixedRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, query, k):
        return self.hits[:k]


class ScriptedLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, system, prompt):
        self.calls.append((system, prompt))
        return self.reply


HITS = [
    hit("a", "Cats purr.", source="cats.pdf", page=3),
    hit("b", "Dogs bark.", source="dogs.txt"),
    hit("c", "Birds sing.", source="birds.md"),
]


def test_prompt_numbers_passages_and_shows_source_and_page():
    prompt = build_prompt("What do cats do?", HITS)
    assert "[1] (source: cats.pdf, page 3)\nCats purr." in prompt
    assert "[2] (source: dogs.txt)\nDogs bark." in prompt
    assert prompt.endswith("Question: What do cats do?")


def test_system_prompt_contains_the_refusal_sentence():
    assert REFUSAL_TEXT in SYSTEM_PROMPT


def test_answer_has_citations_with_source_and_page():
    llm = ScriptedLLM("Cats purr [1].")
    answer = AnswerGenerator(FixedRetriever(HITS), llm).answer("q", 3)
    assert answer.text == "Cats purr [1]."
    assert not answer.refused
    assert [(c.chunk_id, c.source, c.page) for c in answer.citations] == [("a", "cats.pdf", 3)]
    assert len(llm.calls) == 1


def test_citations_are_ordered_by_first_mention_and_deduplicated():
    cites = extract_citations("Dogs [2]. Cats [1][2]. Again [2, 3].", HITS)
    assert [c.chunk_id for c in cites] == ["b", "a", "c"]


def test_out_of_range_citations_are_ignored():
    assert extract_citations("Nothing [0] [4] [99]", HITS) == []


def test_snippet_is_truncated():
    long_hit = hit("x", "word " * 100)
    [cite] = extract_citations("[1]", [long_hit], snippet_chars=20)
    assert len(cite.snippet) == 20


def test_no_retrieved_chunks_refuses_without_calling_the_llm():
    llm = ScriptedLLM("should not be used")
    answer = AnswerGenerator(FixedRetriever([]), llm).answer("q")
    assert answer.refused
    assert answer.text == REFUSAL_TEXT
    assert llm.calls == []


@pytest.mark.parametrize("reply", [REFUSAL_TEXT, REFUSAL_TEXT.rstrip("."), REFUSAL_TEXT.upper()])
def test_model_refusal_is_detected(reply):
    answer = AnswerGenerator(FixedRetriever(HITS), ScriptedLLM(reply)).answer("q")
    assert answer.refused
    assert answer.text == REFUSAL_TEXT
    assert answer.citations == []


def test_answer_without_citations_is_kept_but_has_none():
    answer = AnswerGenerator(FixedRetriever(HITS), ScriptedLLM("Cats purr.")).answer("q")
    assert answer.text == "Cats purr."
    assert answer.citations == []
    assert not answer.refused


def test_top_k_limits_the_passages_sent_to_the_llm():
    llm = ScriptedLLM("ok [1]")
    AnswerGenerator(FixedRetriever(HITS), llm).answer("q", 1)
    prompt = llm.calls[0][1]
    assert "Cats purr." in prompt
    assert "Dogs bark." not in prompt


def test_retriever_enum_matches_the_factory():
    assert {r.value for r in RetrieverName} == set(RETRIEVER_NAMES)


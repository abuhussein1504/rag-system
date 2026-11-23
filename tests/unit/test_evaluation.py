import pytest
from pydantic import ValidationError

from src.core.evaluation.golden import (
    GoldItem,
    coverage,
    is_relevant,
    is_right_page,
    load_golden,
    original_name,
)
from src.core.evaluation.metrics import aggregate, first_relevant_rank
from src.core.evaluation.runner import evaluate
from src.models.enums import RetrievalSource
from src.models.schemas import Chunk, RetrievedChunk

ANSWER = "The first rule of functions is that they should be small."


def make_chunk(text, source="abc123def456_a.pdf", page=None) -> Chunk:
    meta = {"source": source}
    if page is not None:
        meta["page"] = page
    return Chunk(
        id=text[:20], doc_id="d", text=text, start_char=0, end_char=len(text), metadata=meta
    )


GOLD = GoldItem(id="q001", question="What is the first rule?", doc="a.pdf", page=65, answers=[ANSWER])


def test_original_name_removes_upload_prefix():
    assert original_name("afb14667855d_clean-code.pdf") == "clean-code.pdf"
    assert original_name("plain.txt") == "plain.txt"


def test_coverage_full_and_zero():
    assert coverage(ANSWER, "intro text. " + ANSWER + " more text") == 1.0
    assert coverage(ANSWER, "something completely different about cooking pasta today") == 0.0


def test_coverage_ignores_case_punctuation_and_line_breaks():
    messy = "the FIRST rule of\nfunctions is that, they should be small"
    assert coverage(ANSWER, messy) == 1.0


def test_chunk_with_answer_is_relevant():
    assert is_relevant(make_chunk("before " + ANSWER + " after"), GOLD)


def test_chunk_without_answer_is_not_relevant():
    assert not is_relevant(make_chunk("nothing useful here at all, sorry friend"), GOLD)


def test_other_file_is_not_relevant():
    assert not is_relevant(make_chunk(ANSWER, source="abc123def456_b.pdf"), GOLD)


def test_any_of_several_answers_counts():
    other = "Functions should do one thing and they should do it well only."
    gold = GoldItem(id="q2", question="What about functions?", doc="a.pdf", answers=[ANSWER, other])
    assert is_relevant(make_chunk(other), gold)


def test_right_page():
    assert is_right_page(make_chunk("x", page=65), GOLD)
    assert not is_right_page(make_chunk("x", page=66), GOLD)


def test_gold_needs_a_long_enough_answer():
    with pytest.raises(ValidationError):
        GoldItem(id="q", question="What is it?", doc="a.pdf", answers=["too short"])


def test_gold_needs_at_least_one_answer():
    with pytest.raises(ValidationError):
        GoldItem(id="q", question="What is it?", doc="a.pdf", answers=[])


def test_duplicate_ids_are_rejected(tmp_path):
    line = GOLD.model_dump_json()
    f = tmp_path / "g.jsonl"
    f.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_golden(f)


def test_first_relevant_rank():
    assert first_relevant_rank([False, True, True]) == 2
    assert first_relevant_rank([False, False]) is None


def test_aggregate_and_prefix():
    m = aggregate([1, 2, None, 6])
    assert m["hit@1"] == 0.25
    assert m["hit@3"] == 0.5
    assert m["hit@10"] == 0.75
    assert m["mrr"] == 0.4167
    assert "page_mrr" in aggregate([1], prefix="page_")


def test_evaluate_ranks_and_pages():
    wrong = RetrievedChunk(
        chunk=make_chunk("nothing useful here at all, sorry friend", page=1),
        score=0.9, source=RetrievalSource.DENSE,
    )
    right = RetrievedChunk(
        chunk=make_chunk("intro. " + ANSWER + " more", page=65),
        score=0.8, source=RetrievalSource.DENSE,
    )
    report = evaluate([GOLD], lambda q, k: [wrong, right], k=5)
    q = report["questions"][0]
    assert q["rank"] == 2
    assert q["page_rank"] == 2
    assert report["metrics"]["hit@1"] == 0.0
    assert report["metrics"]["hit@3"] == 1.0
    assert report["metrics"]["page_hit@3"] == 1.0



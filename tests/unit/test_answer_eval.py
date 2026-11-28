import json

import pytest

from src.core.evaluation.answers import (
    RecordingRetriever,
    UnansweredQuestion,
    build_judge_prompt,
    evaluate_answerable,
    load_unanswerable,
    parse_judgement,
    reusable_rows,
    run_answer_eval,
    summarize,
)
from src.core.evaluation.golden import GoldItem
from src.core.llm.base import LLMError
from src.models.enums import RetrievalSource
from src.models.schemas import Answer, Chunk, Citation, RetrievedChunk

ANSWER_TEXT = "The first rule of functions is that they should be small."
GOLD = GoldItem(id="q1", question="What is the first rule?", doc="a.pdf", page=1,
                answers=[ANSWER_TEXT])


def hit(chunk_id: str, text: str) -> RetrievedChunk:
    chunk = Chunk(id=chunk_id, doc_id="d", text=text, start_char=0, end_char=len(text),
                  metadata={"source": "abc123def456_a.pdf", "page": 1})
    return RetrievedChunk(chunk=chunk, score=1.0, source=RetrievalSource.HYBRID)


GOOD = hit("good", "intro. " + ANSWER_TEXT + " more text")
BAD = hit("bad", "something else entirely about cooking pasta in a big pot")


def cited(number: int, hit_: RetrievedChunk) -> Citation:
    return Citation(number=number, chunk_id=hit_.chunk.id, doc_id="d", snippet="s")


class ScriptedJudge:
    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[str] = []

    def generate(self, system: str, prompt: str) -> str:
        self.calls.append(prompt)
        if self.error:
            raise self.error
        return self.reply


FAITHFUL = '{"claims": [{"claim": "a", "supported": true}], "correct": true}'


# ---- judge parsing

def test_parse_judgement_reads_plain_json():
    verdict = parse_judgement(
        '{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}],'
        ' "correct": false}'
    )
    assert (verdict.claims_total, verdict.claims_supported, verdict.correct) == (2, 1, False)
    assert verdict.faithfulness == 0.5


def test_parse_judgement_tolerates_fences_and_extra_words():
    text = "Sure!\n```json\n" + FAITHFUL + "\n```\nHope that helps."
    assert parse_judgement(text).correct is True


@pytest.mark.parametrize(
    "text",
    ["no json here", '{"claims": [], "correct": true}', '{"correct": true}',
     '{"claims": [{"claim": "a", "supported": true}], "correct": "yes"}', "{broken"],
)
def test_parse_judgement_rejects_bad_answers(text):
    with pytest.raises(ValueError):
        parse_judgement(text)


def test_judge_prompt_contains_passages_reference_and_answer():
    prompt = build_judge_prompt("Q?", [GOOD], ["the reference"], "the answer")
    assert "[1] (source: abc123def456_a.pdf, page 1)" in prompt
    assert "- the reference" in prompt
    assert "the answer" in prompt


# ---- one question

def answer_fn_for(answer, hits):
    return lambda question: (answer, hits)


def test_faithful_correct_answer_with_a_relevant_citation():
    answer = Answer(text="Small [1].", citations=[cited(1, GOOD)])
    row = evaluate_answerable(GOLD, answer_fn_for(answer, [GOOD, BAD]), ScriptedJudge(FAITHFUL))
    assert row["correct"] is True
    assert row["faithfulness"] == 1.0
    assert row["fully_supported"] is True
    assert row["context_has_answer"] is True
    assert row["citation_hit"] is True
    assert row["cited_numbers"] == [1]


def test_citing_the_wrong_chunk_is_not_a_citation_hit():
    answer = Answer(text="Small [2].", citations=[cited(2, BAD)])
    row = evaluate_answerable(GOLD, answer_fn_for(answer, [GOOD, BAD]), ScriptedJudge(FAITHFUL))
    assert row["context_has_answer"] is True
    assert row["citation_hit"] is False


def test_a_refusal_counts_as_not_correct_and_skips_the_judge():
    judge = ScriptedJudge(FAITHFUL)
    row = evaluate_answerable(
        GOLD, answer_fn_for(Answer(text="I don't know", refused=True), [BAD]), judge
    )
    assert row["refused"] is True
    assert row["correct"] is False
    assert judge.calls == []


def test_an_unreadable_verdict_is_recorded_not_crashed():
    answer = Answer(text="Small [1].", citations=[cited(1, GOOD)])
    row = evaluate_answerable(GOLD, answer_fn_for(answer, [GOOD]), ScriptedJudge("garbage"))
    assert row["judge_error"] is True
    assert row["correct"] is False


def test_an_unavailable_judge_stops_the_run_instead_of_faking_a_verdict():
    answer = Answer(text="Small [1].", citations=[cited(1, GOOD)])
    with pytest.raises(LLMError):
        evaluate_answerable(
            GOLD, answer_fn_for(answer, [GOOD]), ScriptedJudge(error=LLMError("quota"))
        )


def test_reusable_rows_drop_only_judge_failures():
    report = {
        "answerable": [row(id="q1"), row(id="q2", judge_error=True)],
        "unanswerable": [{"id": "u1", "refused": True}],
    }
    assert set(reusable_rows(report)) == {"a:q1", "u:u1"}


def test_recording_retriever_remembers_the_last_hits():
    class Inner:
        def retrieve(self, query, k):
            return [GOOD]

    recorder = RecordingRetriever(Inner())
    assert recorder.retrieve("q", 5) == [GOOD]
    assert recorder.last == [GOOD]


# ---- many questions

def row(**overrides):
    base = {"id": "x", "question": "q", "answer": "a", "refused": False, "cited_numbers": [1],
            "context_has_answer": True, "citation_hit": True, "latency_ms": 10.0,
            "faithfulness": 1.0, "fully_supported": True, "correct": True, "judge_error": False}
    return {**base, **overrides}


def test_summarize_computes_each_rate():
    answerable = [
        row(),
        row(faithfulness=0.5, fully_supported=False, correct=False, cited_numbers=[]),
        row(refused=True, correct=False, faithfulness=None, fully_supported=None,
            citation_hit=False, cited_numbers=[]),
        row(judge_error=True, faithfulness=None, fully_supported=None, correct=False),
    ]
    unanswerable = [{"refused": True}, {"refused": True}, {"refused": False}]
    m = summarize(answerable, unanswerable)

    assert m["answer_rate"] == 0.75
    assert m["correct_rate"] == 0.25
    assert m["faithfulness"] == 0.75           # only the two judged answers
    assert m["fully_supported_rate"] == 0.5
    assert m["uncited_rate"] == round(1 / 3, 4)  # one of three answered has no citation
    assert m["refusal_rate_unanswerable"] == round(2 / 3, 4)
    assert m["judge_errors"] == 1


def test_run_answer_eval_skips_finished_questions_and_waits_between_calls():
    answer = Answer(text="Small [1].", citations=[cited(1, GOOD)])
    refusal = Answer(text="I don't know", refused=True)
    calls, saved, pauses = [], [], []

    def answer_fn(question):
        calls.append(question)
        return (refusal if "weather" in question else answer), [GOOD]

    previous = row(id="q1")
    report = run_answer_eval(
        [GOLD, GOLD.model_copy(update={"id": "q2"})],
        [UnansweredQuestion(id="u1", question="What is the weather?")],
        answer_fn,
        ScriptedJudge(FAITHFUL),
        existing={"a:q1": previous},
        on_row=lambda key, r: saved.append(key),
        delay=2.0,
        sleep=pauses.append,
    )
    assert saved == ["a:q2", "u:u1"]          # q1 was not repeated
    assert len(calls) == 2
    assert pauses == [2.0, 2.0]
    assert report["metrics"]["n_answerable"] == 2
    assert report["metrics"]["refusal_rate_unanswerable"] == 1.0


def test_load_unanswerable_checks_ids(tmp_path):
    f = tmp_path / "u.jsonl"
    line = json.dumps({"id": "u1", "question": "Who is Ronaldo?"})
    f.write_text(line + "\n", encoding="utf-8")
    assert load_unanswerable(f)[0].id == "u1"
    f.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_unanswerable(f)


def test_show_rows_formats_a_row_for_hand_checking():
    from scripts.show_rows import format_row

    text = format_row(row(id="q9", question="Why?", answer="Because [1]."),
                      {"page": 7, "answers": ["the reference"]})
    assert "== q9" in text
    assert "QUESTION: Why?" in text
    assert "ANSWER:   Because [1]." in text
    assert "GOLD (page 7): the reference" in text


def test_answer_report_comparison_table_and_flips():
    from scripts.compare_answer_reports import build_table, flips

    old = {"name": "a", "metrics": {"correct_rate": 0.5}, "answerable": [
        row(id="q1", correct=True), row(id="q2", correct=False), row(id="q3", correct=False)]}
    new = {"name": "b", "metrics": {"correct_rate": 0.6}, "answerable": [
        row(id="q1", correct=False), row(id="q2", correct=True), row(id="q3", correct=False)]}
    assert flips(old, new) == (["q2"], ["q1"])
    table = build_table([old, new])
    assert "| a | 0.5 |" in table
    assert "| b | 0.6 |" in table

import json
import re
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from src.core.evaluation.golden import GoldItem, is_relevant
from src.core.generation.prompt import format_passage
from src.core.llm.base import LLMClient
from src.models.schemas import Answer, RetrievedChunk

JUDGE_SYSTEM = (
    "You are a strict evaluator of answers produced by a retrieval system. "
    "Judge ONLY from the texts you are given, never from your own knowledge. "
    "Respond with a single JSON object and nothing else."
)

AnswerFn = Callable[[str], tuple[Answer, list[RetrievedChunk]]]
Row = dict[str, Any]


class RecordingRetriever:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.last: list[RetrievedChunk] = []

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        self.last = self.inner.retrieve(query, k)
        return self.last


class UnansweredQuestion(BaseModel):
    id: str
    question: str = Field(min_length=5)


def load_unanswerable(path: Path) -> list[UnansweredQuestion]:
    lines = path.read_text(encoding="utf-8").splitlines()
    items = [UnansweredQuestion.model_validate_json(line) for line in lines if line.strip()]
    if len({item.id for item in items}) != len(items):
        raise ValueError("Duplicate ids in the unanswerable set")
    return items


# ---- the judge

def build_judge_prompt(
    question: str, hits: list[RetrievedChunk], reference: list[str], answer: str
) -> str:
    passages = "\n\n".join(format_passage(i, hit) for i, hit in enumerate(hits, start=1))
    references = "\n".join(f"- {text}" for text in reference)
    return f"""Question:
{question}

Passages that were given to the system (numbered):
{passages}

Reference passage(s) that contain the correct information:
{references}

Answer to evaluate:
{answer}

Tasks:
1. Split the answer into its factual claims. For each claim, say whether the numbered passages \
support it (supported: true or false). Ignore citation markers like [1].
2. Say whether the answer correctly and sufficiently answers the question according to the \
reference passage(s) (correct: true or false).

Respond as JSON only: {{"claims": [{{"claim": "...", "supported": true}}], "correct": true}}"""


class Judgement(BaseModel):
    claims_total: int
    claims_supported: int
    correct: bool

    @property
    def faithfulness(self) -> float:
        return self.claims_supported / self.claims_total


def parse_judgement(text: str) -> Judgement:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise ValueError("The judge did not return JSON")
    try:
        data = json.loads(match.group(0))
        claims = data["claims"]
        supported = [bool(claim["supported"]) for claim in claims]
        correct = data["correct"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Unreadable judge answer: {exc}") from exc
    if not supported:
        raise ValueError("The judge listed no claims")
    if not isinstance(correct, bool):
        raise ValueError("'correct' must be true or false")
    return Judgement(
        claims_total=len(supported), claims_supported=sum(supported), correct=correct
    )


# ---- one question at a time

def evaluate_answerable(item: GoldItem, answer_fn: AnswerFn, judge: LLMClient) -> Row:
    started = time.perf_counter()
    answer, hits = answer_fn(item.question)
    latency_ms = round(1000 * (time.perf_counter() - started), 1)

    by_id = {hit.chunk.id: hit.chunk for hit in hits}
    cited = [by_id[c.chunk_id] for c in answer.citations if c.chunk_id in by_id]
    row: Row = {
        "id": item.id,
        "question": item.question,
        "answer": answer.text,
        "refused": answer.refused,
        "cited_numbers": [c.number for c in answer.citations],
        "context_has_answer": any(is_relevant(hit.chunk, item) for hit in hits),
        "citation_hit": any(is_relevant(chunk, item) for chunk in cited),
        "latency_ms": latency_ms,
        "faithfulness": None,
        "fully_supported": None,
        "correct": False,
        "judge_error": False,
    }
    if answer.refused:
        return row

    try:
        prompt = build_judge_prompt(item.question, hits, item.answers, answer.text)
        verdict = parse_judgement(judge.generate(JUDGE_SYSTEM, prompt))
    except ValueError:  # unreadable verdict; an unavailable judge (LLMError) stops the run
        row["judge_error"] = True
        return row
    row["faithfulness"] = round(verdict.faithfulness, 4)
    row["fully_supported"] = verdict.claims_supported == verdict.claims_total
    row["correct"] = verdict.correct
    return row


def evaluate_unanswerable(item: UnansweredQuestion, answer_fn: AnswerFn) -> Row:
    answer, _ = answer_fn(item.question)
    return {"id": item.id, "question": item.question, "answer": answer.text,
            "refused": answer.refused}


# ---- all questions

def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarize(answerable: list[Row], unanswerable: list[Row]) -> dict[str, Any]:
    answered = [r for r in answerable if not r["refused"]]
    judged = [r for r in answered if not r["judge_error"]]
    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "answer_rate": _rate([not r["refused"] for r in answerable]),
        "correct_rate": _rate([r["correct"] for r in answerable]),
        "faithfulness": round(mean(r["faithfulness"] for r in judged), 4) if judged else None,
        "fully_supported_rate": _rate([bool(r["fully_supported"]) for r in judged]),
        "context_hit_rate": _rate([r["context_has_answer"] for r in answerable]),
        "citation_hit_rate": _rate([r["citation_hit"] for r in answered]),
        "uncited_rate": _rate([not r["cited_numbers"] for r in answered]),
        "refusal_rate_unanswerable": _rate([r["refused"] for r in unanswerable]),
        "judge_errors": sum(r["judge_error"] for r in answered),
        "avg_latency_ms": round(mean(r["latency_ms"] for r in answerable), 1)
        if answerable else None,
    }


def reusable_rows(report: dict[str, Any]) -> dict[str, Row]:
    """Rows of a finished report that need no redo: everything except judge failures."""
    keep: dict[str, Row] = {}
    for row in report["answerable"]:
        if not row["judge_error"]:
            keep[f"a:{row['id']}"] = row
    for row in report["unanswerable"]:
        keep[f"u:{row['id']}"] = row
    return keep


def run_answer_eval(
    golden: list[GoldItem],
    unanswerable: list[UnansweredQuestion],
    answer_fn: AnswerFn,
    judge: LLMClient,
    *,
    existing: dict[str, Row] | None = None,
    on_row: Callable[[str, Row], None] | None = None,
    delay: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    done = existing or {}
    answerable_rows: list[Row] = []
    unanswerable_rows: list[Row] = []

    def handle(key: str, compute: Callable[[], Row]) -> Row:
        if key in done:
            return done[key]
        row = compute()
        if on_row:
            on_row(key, row)
        sleep(delay)
        return row

    for item in golden:
        compute = partial(evaluate_answerable, item, answer_fn, judge)
        answerable_rows.append(handle(f"a:{item.id}", compute))
    for question in unanswerable:
        compute = partial(evaluate_unanswerable, question, answer_fn)
        unanswerable_rows.append(handle(f"u:{question.id}", compute))
    return {
        "metrics": summarize(answerable_rows, unanswerable_rows),
        "answerable": answerable_rows,
        "unanswerable": unanswerable_rows,
    }


import time
from collections.abc import Callable
from typing import Any

from src.core.evaluation.golden import GoldItem, is_relevant, original_name
from src.core.evaluation.metrics import aggregate, first_relevant_rank
from src.models.schemas import RetrievedChunk

Retrieve = Callable[[str, int], list[RetrievedChunk]]


def evaluate(golden: list[GoldItem], retrieve: Retrieve, *, k: int = 10) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    latencies: list[float] = []

    for gold in golden:
        started = time.perf_counter()
        hits = retrieve(gold.question, k)
        latencies.append(time.perf_counter() - started)

        flags = [is_relevant(hit.chunk, gold) for hit in hits]
        questions.append(
            {
                "id": gold.id,
                "question": gold.question,
                "rank": first_relevant_rank(flags),
                "top3": [
                    {
                        "source": original_name(hit.chunk.metadata.get("source", "")),
                        "page": hit.chunk.metadata.get("page"),
                        "score": round(hit.score, 4),
                        "relevant": flag,
                        "preview": hit.chunk.text[:120],
                    }
                    for hit, flag in list(zip(hits, flags, strict=True))[:3]
                ],
            }
        )

    metrics = aggregate([q["rank"] for q in questions])
    metrics["avg_latency_ms"] = round(1000 * sum(latencies) / len(latencies), 1)
    return {"metrics": metrics, "questions": questions}


import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src.api.dependencies import (
    get_embedder,
    get_judge_llm,
    get_llm,
    get_reranker,
    get_vector_store,
)
from src.config import get_settings
from src.core.evaluation.answers import (
    RecordingRetriever,
    Row,
    load_unanswerable,
    reusable_rows,
    run_answer_eval,
)
from src.core.evaluation.golden import load_golden
from src.core.generation.generator import AnswerGenerator
from src.core.indexing.indexer import collection_name
from src.core.llm.base import LLMError
from src.core.retrieval.factory import RETRIEVER_NAMES, build_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated answers.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True, help="e.g. 06_answers_hybrid")
    parser.add_argument("--retriever", choices=RETRIEVER_NAMES, default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, help="only the first N answerable questions")
    parser.add_argument("--delay", type=float, default=4.0, help="seconds between questions")
    parser.add_argument(
        "--redo-errors",
        metavar="REPORT",
        help="start from a finished report and redo only the rows the judge failed on",
    )
    parser.add_argument("--golden", default="eval/golden_set.jsonl")
    parser.add_argument("--unanswerable", default="eval/unanswerable.jsonl")
    args = parser.parse_args()

    settings = get_settings()
    store = get_vector_store()
    if store.count(collection_name(args.project)) == 0:
        raise SystemExit(f"Project '{args.project}' is not indexed.")

    golden = load_golden(Path(args.golden))
    if args.limit:
        golden = golden[: args.limit]
    unanswerable = load_unanswerable(Path(args.unanswerable))

    recorder = RecordingRetriever(
        build_retriever(
            args.retriever,
            project=args.project,
            processed_dir=Path(settings.DATA_DIR) / "processed" / args.project,
            embedder=get_embedder(),
            store=store,
            reranker=get_reranker() if "rerank" in args.retriever else None,
        )
    )
    generator = AnswerGenerator(recorder, get_llm())

    judge = get_judge_llm()
    judge_backend = settings.JUDGE_BACKEND or settings.GENERATION_BACKEND
    judge_model = settings.JUDGE_MODEL_ID or settings.GENERATION_MODEL_ID

    out_dir = Path("eval/answer_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / f"{args.name}.partial.jsonl"
    existing: dict[str, Row] = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            existing[entry["key"]] = entry["row"]
        print(f"Resuming: {len(existing)} questions already done.")
    elif args.redo_errors:
        finished = json.loads((out_dir / f"{args.redo_errors}.json").read_text(encoding="utf-8"))
        existing = reusable_rows(finished)
        print(f"Keeping {len(existing)} good rows of {args.redo_errors}; redoing the rest.")

    def save_row(key: str, row: Row) -> None:
        with partial.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "row": row}, ensure_ascii=False) + "\n")
        print(f"  {key}")

    try:
        report = run_answer_eval(
            golden,
            unanswerable,
            lambda question: (generator.answer(question, args.top_k), recorder.last),
            judge,
            existing=existing,
            on_row=save_row,
            delay=args.delay,
        )
    except LLMError as exc:
        raise SystemExit(
            f"\nStopped: {exc}\nProgress is saved. Run the same command again to continue."
        ) from None
    
    report["name"] = args.name
    report["created_at"] = datetime.now(UTC).isoformat()
    report["config"] = {
        "retriever": args.retriever,
        "top_k": args.top_k,
        "generation_backend": settings.GENERATION_BACKEND,
        "generation_model": settings.GENERATION_MODEL_ID,
        "judge_backend": judge_backend,
        "judge_model": judge_model,
        "same_model_judges_itself": (
            judge_backend == settings.GENERATION_BACKEND
            and judge_model == settings.GENERATION_MODEL_ID
        ),
    }

    out = out_dir / f"{args.name}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    partial.unlink(missing_ok=True)

    print(f"\n{args.name}")
    for key, value in report["metrics"].items():
        print(f"  {key:<28}{value}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()


import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src.api.dependencies import get_embedder, get_reranker, get_vector_store
from src.config import get_settings
from src.core.evaluation.golden import load_golden
from src.core.evaluation.runner import evaluate
from src.core.indexing.indexer import collection_name
from src.core.retrieval.factory import RETRIEVER_NAMES, build_retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True, help="e.g. 01_dense_baseline")
    parser.add_argument("--golden", default="eval/golden_set.jsonl")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--retriever", choices=RETRIEVER_NAMES, default="dense")
    args = parser.parse_args()

    settings = get_settings()
    golden = load_golden(Path(args.golden))
    store = get_vector_store()

    if store.count(collection_name(args.project)) == 0:
        raise SystemExit(
            f"Collection for project '{args.project}' is empty. "
            f"Index it first: POST /api/data/index/{args.project}"
        )
    
    retriever = build_retriever(
        args.retriever,
        project=args.project,
        processed_dir=Path(settings.DATA_DIR) / "processed" / args.project,
        embedder=get_embedder(),
        reranker=get_reranker() if "rerank" in args.retriever else None,
        store=store,
    )

    report = evaluate(golden, retriever.retrieve, k=args.k)
    report["name"] = args.name
    report["created_at"] = datetime.now(UTC).isoformat()
    report["config"] = {
        "retriever": args.retriever,
        "chunk_size": settings.FILE_DEFAULT_CHUNK_SIZE,
        "overlap": settings.FILE_DEFAULT_OVERLAP_SIZE,
        "embedding_backend": settings.EMBEDDING_BACKEND,
        "n_questions": len(golden),
    }

    embedder = get_embedder()
    info = getattr(embedder, "info", None)
    try:
        report["config"]["embedder"] = info() if callable(info) else {}
    except Exception:
        report["config"]["embedder"] = {"model": settings.EMBEDDING_MODEL_ID}

    out = Path("eval/reports") / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{args.name}  (questions: {len(golden)})")
    for key, value in report["metrics"].items():
        print(f"  {key:<16}{value}")
    missed = sum(1 for q in report["questions"] if q["rank"] is None)
    print(f"  not found in top {args.k}: {missed}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()


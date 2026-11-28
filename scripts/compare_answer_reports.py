import argparse
import json
from pathlib import Path
from typing import Any

COLUMNS = [
    "correct_rate", "faithfulness", "fully_supported_rate", "answer_rate",
    "citation_hit_rate", "uncited_rate", "refusal_rate_unanswerable", "judge_errors",
]


def build_table(reports: list[dict[str, Any]]) -> str:
    header = "| report | " + " | ".join(COLUMNS) + " |"
    separator = "|" + "---|" * (len(COLUMNS) + 1)
    rows = []
    for report in reports:
        metrics = report["metrics"]
        cells = [str(metrics.get(column, "-")) for column in COLUMNS]
        rows.append(f"| {report['name']} | " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def flips(old: dict[str, Any], new: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Questions that became correct, and questions that stopped being correct."""
    was_correct = {row["id"]: row["correct"] for row in old["answerable"]}
    fixed = [r["id"] for r in new["answerable"] if r["correct"] and not was_correct.get(r["id"])]
    broken = [r["id"] for r in new["answerable"] if not r["correct"] and was_correct.get(r["id"])]
    return fixed, broken


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare answer-evaluation reports.")
    parser.add_argument("old", nargs="?", help="e.g. 06_answers_hybrid")
    parser.add_argument("new", nargs="?", help="e.g. 07_answers_prompt_v2")
    args = parser.parse_args()

    folder = Path("eval/answer_reports")
    paths = sorted(p for p in folder.glob("*.json") if not p.name.startswith("trial"))
    print(build_table([json.loads(p.read_text(encoding="utf-8")) for p in paths]))

    if args.old and args.new:
        old = json.loads((folder / f"{args.old}.json").read_text(encoding="utf-8"))
        new = json.loads((folder / f"{args.new}.json").read_text(encoding="utf-8"))
        fixed, broken = flips(old, new)
        print(f"\n{args.old} -> {args.new}")
        print(f"  became correct ({len(fixed)}): {', '.join(fixed) or '-'}")
        print(f"  stopped being correct ({len(broken)}): {', '.join(broken) or '-'}")


if __name__ == "__main__":
    main()

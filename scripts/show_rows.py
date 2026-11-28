import argparse
import json
from pathlib import Path
from typing import Any


def format_row(row: dict[str, Any], gold: dict[str, Any] | None) -> str:
    lines = [
        f"== {row['id']}  correct={row['correct']}  faithfulness={row['faithfulness']}  "
        f"refused={row['refused']}  judge_error={row['judge_error']}",
        f"context had the answer: {row['context_has_answer']}   cited: {row['cited_numbers']}",
        f"QUESTION: {row['question']}",
        f"ANSWER:   {row['answer']}",
    ]
    if gold:
        lines.append(f"GOLD (page {gold.get('page')}): " + " | ".join(gold["answers"]))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print answer-evaluation rows for hand checking.")
    parser.add_argument("report", help="e.g. 06_answers_hybrid")
    parser.add_argument("ids", nargs="+", help="e.g. q001 q075")
    parser.add_argument("--golden", default="eval/golden_set.jsonl")
    args = parser.parse_args()

    report = json.loads(Path(f"eval/answer_reports/{args.report}.json").read_text("utf-8"))
    rows = {r["id"]: r for r in report["answerable"]}
    lines = Path(args.golden).read_text(encoding="utf-8").splitlines()
    gold = {g["id"]: g for g in (json.loads(line) for line in lines if line.strip())}
    for question_id in args.ids:
        if question_id not in rows:
            print(f"== {question_id}: not in the report\n")
            continue
        print(format_row(rows[question_id], gold.get(question_id)) + "\n")


if __name__ == "__main__":
    main()

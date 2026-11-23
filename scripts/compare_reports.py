import json
from pathlib import Path
from typing import Any

COLUMNS = ["hit@1", "hit@5", "hit@10", "mrr", "page_hit@5", "page_mrr", "avg_latency_ms"]


def build_table(reports: list[dict[str, Any]]) -> str:
    header = "| experiment | " + " | ".join(COLUMNS) + " |"
    separator = "|" + "---|" * (len(COLUMNS) + 1)
    rows = []
    for report in reports:
        metrics = report["metrics"]
        cells = [
            f"{metrics[c]:.3f}" if c != "avg_latency_ms" and c in metrics
            else str(metrics.get(c, "-"))
            for c in COLUMNS
        ]
        rows.append(f"| {report['name']} | " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def main() -> None:
    paths = sorted(Path("eval/reports").glob("*.json"))
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    print(build_table(reports))


if __name__ == "__main__":
    main()


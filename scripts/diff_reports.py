import argparse
import json
import random
from pathlib import Path
from typing import Any


def _reciprocal(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def _order(rank: int | None) -> float:
    return float("inf") if rank is None else float(rank)


def _change(pair: tuple[dict[str, Any], dict[str, Any]]) -> float:
    old, new = pair
    return _order(new["rank"]) - _order(old["rank"])


def paired_bootstrap(
    old: list[float], new: list[float], rounds: int = 2000, seed: int = 0
) -> tuple[float, float]:
    rng = random.Random(seed)
    diffs = [b - a for a, b in zip(old, new, strict=True)]
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def compare(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_by_id = {q["id"]: q for q in old["questions"]}
    pairs = [(old_by_id[q["id"]], q) for q in new["questions"] if q["id"] in old_by_id]

    improved = [(a, b) for a, b in pairs if _order(b["rank"]) < _order(a["rank"])]
    worsened = [(a, b) for a, b in pairs if _order(b["rank"]) > _order(a["rank"])]
    low, high = paired_bootstrap(
        [_reciprocal(a["rank"]) for a, _ in pairs], [_reciprocal(b["rank"]) for _, b in pairs]
    )
    return {
        "n": len(pairs),
        "improved": improved,
        "worsened": worsened,
        "same": len(pairs) - len(improved) - len(worsened),
        "mrr_change_low": round(low, 4),
        "mrr_change_high": round(high, 4),
    }


def _load(name: str) -> dict[str, Any]:
    path = Path("eval/reports") / f"{name}.json"
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluation reports.")
    parser.add_argument("old")
    parser.add_argument("new")
    args = parser.parse_args()

    result = compare(_load(args.old), _load(args.new))
    print(f"{args.old} -> {args.new}  ({result['n']} questions)")
    print(f"  better: {len(result['improved'])}   worse: {len(result['worsened'])}   "
          f"same: {result['same']}")
    low, high = result["mrr_change_low"], result["mrr_change_high"]
    print(f"  change in MRR, 95% interval: {low:+.3f} to {high:+.3f}")
    verdict = "a real change" if low > 0 or high < 0 else "within noise (not a proven change)"
    print(f"  verdict: {verdict}")

    print("\nBiggest drops:")
    for a, b in sorted(result["worsened"], key=_change, reverse=True)[:5]:
        print(f"  rank {a['rank']} -> {b['rank']}: {a['question'][:90]}")
    print("\nBiggest gains:")
    for a, b in sorted(result["improved"], key=_change)[:5]:
        print(f"  rank {a['rank']} -> {b['rank']}: {a['question'][:90]}")


if __name__ == "__main__":
    main()


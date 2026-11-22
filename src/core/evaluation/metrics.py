from collections.abc import Sequence


def first_relevant_rank(flags: Sequence[bool]) -> int | None:
    for position, is_correct in enumerate(flags, start=1):
        if is_correct:
            return position
    return None


def aggregate(
    ranks: Sequence[int | None], ks: Sequence[int] = (1, 3, 5, 10), prefix: str = ""
) -> dict[str, float]:
    n = len(ranks)
    if n == 0:
        raise ValueError("No results to aggregate")
    metrics = {
        f"{prefix}hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in ks
    }
    metrics[f"{prefix}mrr"] = sum(1 / r for r in ranks if r is not None) / n
    return {name: round(value, 4) for name, value in metrics.items()}


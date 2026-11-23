from collections.abc import Sequence

from src.models.enums import RetrievalSource
from src.models.schemas import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], *, k: int = 60, top_k: int | None = None
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    first_hit: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            first_hit.setdefault(chunk_id, hit)

    ordered = sorted(scores, key=lambda chunk_id: -scores[chunk_id])  # ties keep first-seen order
    fused = [
        RetrievedChunk(
            chunk=first_hit[chunk_id].chunk,
            score=scores[chunk_id],
            source=RetrievalSource.HYBRID,
        )
        for chunk_id in ordered
    ]
    return fused if top_k is None else fused[:top_k]


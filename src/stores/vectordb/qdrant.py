import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from src.models.enums import DistanceMethod, RetrievalSource
from src.models.schemas import Chunk, RetrievedChunk

_DISTANCES = {
    DistanceMethod.COSINE: models.Distance.COSINE,
    DistanceMethod.DOT: models.Distance.DOT,
}


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    def __init__(
        self, path: str, distance: DistanceMethod = DistanceMethod.COSINE
    ) -> None:
        abs_path = Path(path).resolve()
        abs_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(abs_path))
        self.distance = _DISTANCES[distance]

    def ensure_collection(self, name: str, dimension: int) -> None:
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=dimension, distance=self.distance
                ),
            )

    def upsert(
        self, collection: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        points = [
            models.PointStruct(
                id=_point_id(c.id),
                vector=v,
                payload={
                    "chunk_id": c.id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "metadata": c.metadata,
                },
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=collection, points=points)

    def search(
        self, collection: str, vector: list[float], k: int
    ) -> list[RetrievedChunk]:
        if not self.client.collection_exists(collection):
            return []
        result = self.client.query_points(
            collection_name=collection, query=vector, limit=k, with_payload=True
        )
        hits = []
        for p in result.points:
            pl = p.payload or {}
            chunk = Chunk(
                id=pl["chunk_id"],
                doc_id=pl["doc_id"],
                text=pl["text"],
                start_char=pl["start_char"],
                end_char=pl["end_char"],
                metadata=pl["metadata"],
            )
            hits.append(
                RetrievedChunk(chunk=chunk, score=p.score, source=RetrievalSource.DENSE)
            )
        return hits

    def delete_by_source(self, collection: str, source: str) -> None:
        if not self.client.collection_exists(collection):
            return
        self.client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=source),
                        )
                    ]
                )
            ),
        )

    def delete_collection(self, name: str) -> None:
        if self.client.collection_exists(name):
            self.client.delete_collection(name)

    def delete_all(self) -> None:
        for c in self.client.get_collections().collections:
            self.client.delete_collection(c.name)


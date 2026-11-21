from dataclasses import dataclass
from pathlib import Path

from src.models.schemas import Chunk
from src.stores.embedding.base import Embedder
from src.stores.vectordb.base import VectorStore


def collection_name(project_id: str) -> str:
    return f"project_{project_id}"


@dataclass
class IndexResult:
    files_indexed: int = 0
    chunks_indexed: int = 0


def index_project(
    processed_dir: Path,
    collection: str,
    embedder: Embedder,
    store: VectorStore,
    *,
    batch_size: int = 64,
) -> IndexResult:
    store.ensure_collection(collection, embedder.dimension)
    result = IndexResult()

    for path in sorted(processed_dir.glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        chunks = [Chunk.model_validate_json(line) for line in lines if line.strip()]
        if not chunks:
            continue

        store.delete_by_source(collection, chunks[0].metadata["source"])
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors = embedder.embed_documents([c.text for c in batch])
            store.upsert(collection, batch, vectors)

        result.files_indexed += 1
        result.chunks_indexed += len(chunks)

    return result


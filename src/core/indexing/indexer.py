import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src.models.schemas import Chunk
from src.stores.embedding.base import Embedder
from src.stores.vectordb.base import VectorStore

logger = logging.getLogger(__name__)


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
    embed_call_delay: float = 0.7,
) -> IndexResult:
    store.ensure_collection(collection, embedder.dimension)
    result = IndexResult()

    for path in sorted(processed_dir.glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        chunks = [Chunk.model_validate_json(line) for line in lines if line.strip()]
        if not chunks:
            continue

        total_batches = -(-len(chunks) // batch_size)
        logger.info(
            "Indexing %s — %d chunks in %d batch(es)",
            path.name,
            len(chunks),
            total_batches,
        )

        store.delete_by_source(collection, chunks[0].metadata["source"])
        for batch_idx, i in enumerate(range(0, len(chunks), batch_size)):
            if batch_idx > 0 and embed_call_delay > 0:
                time.sleep(embed_call_delay)
            batch = chunks[i : i + batch_size]
            logger.info(
                "  Embedding batch %d/%d (%d chunks)…",
                batch_idx + 1,
                total_batches,
                len(batch),
            )
            vectors = embedder.embed_documents([c.text for c in batch])
            store.upsert(collection, batch, vectors)
            logger.info("  Batch %d/%d stored.", batch_idx + 1, total_batches)

        result.files_indexed += 1
        result.chunks_indexed += len(chunks)
        logger.info("Done indexing %s.", path.name)

    return result


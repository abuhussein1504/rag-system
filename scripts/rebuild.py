import argparse
import logging
from pathlib import Path

from src.api.dependencies import get_embedder, get_vector_store
from src.config import get_settings
from src.core.chunking.fixed import FixedSizeChunker
from src.core.indexing.indexer import collection_name, index_project
from src.core.ingestion.processing import process_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-process and re-index one project.")
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    settings = get_settings()
    data = Path(settings.DATA_DIR)
    chunker = FixedSizeChunker(
        settings.FILE_DEFAULT_CHUNK_SIZE, settings.FILE_DEFAULT_OVERLAP_SIZE
    )

    processed = process_project(
        data / "raw" / args.project,
        data / "processed" / args.project,
        args.project,
        chunker,
        force=True,
    )
    print(processed)

    indexed = index_project(
        data / "processed" / args.project,
        collection_name(args.project),
        get_embedder(),
        get_vector_store(),
    )
    print(indexed)


if __name__ == "__main__":
    main()


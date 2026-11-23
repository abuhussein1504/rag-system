from pathlib import Path

from src.models.schemas import Chunk


def load_chunks(processed_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(processed_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunks.append(Chunk.model_validate_json(line))
    return chunks


import os
from dataclasses import dataclass, field
from pathlib import Path

from src.core.chunking.base import Chunker
from src.core.ingestion.registry import load_file


@dataclass
class ProcessResult:
    files_processed: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    chunks_created: int = 0
    failed_files: list[str] = field(default_factory=list)


def processed_path(raw_path: Path, out_dir: Path) -> Path:
    return out_dir / f"{raw_path.name}.jsonl"


def _is_up_to_date(raw_path: Path, out_path: Path) -> bool:
    return out_path.exists() and out_path.stat().st_mtime >= raw_path.stat().st_mtime


def process_project(
    raw_dir: Path,
    out_dir: Path,
    project_id: str,
    chunker: Chunker,
    *,
    force: bool = False,
) -> ProcessResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = ProcessResult()
    raw_files = [p for p in sorted(raw_dir.iterdir()) if p.is_file()]

    for path in raw_files:
        out_path = processed_path(path, out_dir)
        if not force and _is_up_to_date(path, out_path):
            result.files_skipped += 1
            continue

        tmp_path = out_path.with_suffix(".tmp")
        try:
            count = 0
            with open(tmp_path, "w", encoding="utf-8") as out:
                for doc in load_file(path, project_id):
                    for chunk in chunker.chunk(doc):
                        out.write(chunk.model_dump_json() + "\n")
                        count += 1
            os.replace(tmp_path, out_path)
            result.files_processed += 1
            result.chunks_created += count
        except Exception:
            tmp_path.unlink(missing_ok=True)
            result.failed_files.append(path.name)

    raw_names = {p.name for p in raw_files}
    for jsonl in out_dir.glob("*.jsonl"):
        if jsonl.name.removesuffix(".jsonl") not in raw_names:
            jsonl.unlink()
            result.files_removed += 1

    return result


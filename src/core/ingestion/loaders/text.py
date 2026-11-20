from pathlib import Path

from src.core.ingestion.cleaner import clean_text
from src.core.ingestion.registry import make_document, register
from src.schemas import Document


@register("txt", "md", "csv")
def load_text(path: Path, project_id: str) -> list[Document]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")

    # Markdown and CSV depend on line breaks, so keep them.
    keep_lines = path.suffix.lower() in {".md", ".csv"}
    text = clean_text(raw, join_lines=not keep_lines)

    if not text:
        return []
    return [make_document(path, project_id, text)]


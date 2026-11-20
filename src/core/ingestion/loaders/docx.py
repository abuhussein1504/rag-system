from pathlib import Path

from docx import Document as DocxDocument

from src.core.ingestion.cleaner import clean_text
from src.core.ingestion.registry import make_document, register
from src.models.schemas import Document


@register("docx")
def load_docx(path: Path, project_id: str) -> list[Document]:
    doc = DocxDocument(str(path))

    parts = [p.text for p in doc.paragraphs if p.text.strip()]

    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))

    text = clean_text("\n\n".join(parts), join_lines=False)
    if not text:
        return []
    return [make_document(path, project_id, text)]


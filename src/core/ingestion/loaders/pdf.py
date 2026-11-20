from pathlib import Path
from typing import cast, Iterable

import pymupdf

from src.core.ingestion.cleaner import clean_text
from src.core.ingestion.registry import make_document, register
from src.models.schemas import Document


@register("pdf")
def load_pdf(path: Path, project_id: str) -> list[Document]:
    docs: list[Document] = []
    with pymupdf.open(path) as pdf:
        total = len(pdf)
        for number, page in enumerate(cast(Iterable[pymupdf.Page], pdf), start=1):
            text = clean_text(cast(str, page.get_text()))
            if not text:
                continue
            docs.append(
                make_document(
                    path, project_id, text,
                    key=number, page=number, total_pages=total,
                )
            )
    return docs


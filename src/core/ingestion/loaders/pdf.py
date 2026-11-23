import logging
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pymupdf

from src.core.ingestion.cleaner import clean_text
from src.core.ingestion.pdf_cleaning import clean_pdf_pages
from src.core.ingestion.registry import make_document, register
from src.models.schemas import Document

logger = logging.getLogger(__name__)


@register("pdf")
def load_pdf(path: Path, project_id: str) -> list[Document]:
    with pymupdf.open(path) as pdf:
        raw_pages = [cast(str, page.get_text()) for page in cast(Iterable[pymupdf.Page], pdf)]
    total = len(raw_pages)

    cleaning = clean_pdf_pages(raw_pages)
    logger.info(
        "%s: %d pages, %d page numbers removed, %d contents pages skipped, running headers: %s",
        path.name, total, cleaning.page_numbers_removed,
        cleaning.contents_pages_skipped, cleaning.running_headers or "none",
    )

    docs: list[Document] = []
    for number, raw in enumerate(cleaning.pages, start=1):
        text = clean_text(raw)
        if not text:
            continue
        docs.append(
            make_document(path, project_id, text, key=number, page=number, total_pages=total)
        )
    return docs


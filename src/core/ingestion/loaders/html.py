from pathlib import Path

from bs4 import BeautifulSoup

from src.core.ingestion.cleaner import clean_text
from src.core.ingestion.registry import make_document, register
from src.schemas import Document

NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]


@register("html", "htm")
def load_html(path: Path, project_id: str) -> list[Document]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else None

    for tag in soup(NOISE_TAGS):
        tag.decompose()

    text = clean_text(soup.get_text(separator="\n"), join_lines=False)
    if not text:
        return []
    return [make_document(path, project_id, text, title=title)]


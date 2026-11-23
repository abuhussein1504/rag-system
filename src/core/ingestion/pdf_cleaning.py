import re
from collections import Counter
from dataclasses import dataclass, field

EDGE_LINES = 2
MIN_DOTTED_LINES = 5
MAX_HEADER_CHARS = 100

_ROMAN = re.compile(r"m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})", re.IGNORECASE)
_PAGE_WORD = re.compile(r"^(page\s+)?(?P<num>\d{1,4})(\s+of\s+\d{1,4})?$", re.IGNORECASE)
_DOTTED = re.compile(r"\.{5,}")


def is_page_number(line: str) -> bool:
    text = line.strip().strip("-–—").strip()
    if not text:
        return False
    if _PAGE_WORD.match(text):
        return True
    return len(text) <= 6 and _ROMAN.fullmatch(text) is not None


def _edge_indexes(lines: list[str]) -> set[int]:
    filled = [i for i, line in enumerate(lines) if line.strip()]
    return set(filled[:EDGE_LINES] + filled[-EDGE_LINES:])


def strip_page_numbers(lines: list[str]) -> list[str]:
    edge = _edge_indexes(lines)
    return [line for i, line in enumerate(lines) if not (i in edge and is_page_number(line))]


def _normalize(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())


def find_running_headers(pages: list[list[str]], min_pages: int = 5) -> set[str]:
    counts: Counter[str] = Counter()
    for lines in pages:
        found = {_normalize(lines[i]) for i in _edge_indexes(lines)}
        counts.update(f for f in found if f and len(f) <= MAX_HEADER_CHARS)
    return {line for line, n in counts.items() if n >= min_pages}


def drop_running_headers(lines: list[str], headers: set[str]) -> list[str]:
    edge = _edge_indexes(lines)
    return [
        line for i, line in enumerate(lines) if not (i in edge and _normalize(line) in headers)
    ]


def is_contents_page(text: str) -> bool:
    return len(_DOTTED.findall(text)) >= MIN_DOTTED_LINES


@dataclass
class PdfCleaning:
    pages: list[str] = field(default_factory=list)
    page_numbers_removed: int = 0
    contents_pages_skipped: int = 0
    running_headers: list[str] = field(default_factory=list)


def clean_pdf_pages(raw_pages: list[str], *, min_header_pages: int = 5) -> PdfCleaning:
    result = PdfCleaning()

    contents = [is_contents_page(text) for text in raw_pages]
    result.contents_pages_skipped = sum(contents)

    stripped: list[list[str]] = []
    for text in raw_pages:
        lines = text.splitlines()
        kept = strip_page_numbers(lines)
        result.page_numbers_removed += len(lines) - len(kept)
        stripped.append(kept)

    headers = find_running_headers(
        [lines for lines, skip in zip(stripped, contents, strict=True) if not skip],
        min_header_pages,
    )
    result.running_headers = sorted(headers)

    for lines, skip in zip(stripped, contents, strict=True):
        result.pages.append("" if skip else "\n".join(drop_running_headers(lines, headers)))
    return result
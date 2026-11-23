import pymupdf
import pytest

from src.core.ingestion.cleaner import clean_text
from src.core.ingestion.pdf_cleaning import (
    clean_pdf_pages,
    find_running_headers,
    is_contents_page,
    is_page_number,
    strip_page_numbers,
)
from src.core.ingestion.registry import load_file


@pytest.mark.parametrize("line", ["12", "xiv", "ii", "Page 3 of 70", "- 5 -", " 7 "])
def test_page_numbers_are_detected(line):
    assert is_page_number(line)


@pytest.mark.parametrize(
    "line", ["", "Chapter 3", "mild", "Introduction", "2024 results", "5 apples", "1.1 Overview"]
)
def test_normal_lines_are_not_page_numbers(line):
    assert not is_page_number(line)


def test_only_edge_lines_are_removed():
    lines = ["12", "Heading", "body", "7", "more body", "13"]
    assert strip_page_numbers(lines) == ["Heading", "body", "7", "more body"]


def test_running_header_is_found():
    pages = [["My  Thesis", f"unique body {n}"] for n in range(6)]
    assert find_running_headers(pages, min_pages=5) == {"my thesis"}


def test_lines_that_only_look_similar_are_not_headers():
    pages = [[f"Table 3.{n}", f"unique body {n}"] for n in range(8)]
    assert find_running_headers(pages, min_pages=5) == set()


def test_rare_repeats_are_not_headers():
    pages = [["Summary", f"unique body {n}"] for n in range(3)]
    assert find_running_headers(pages, min_pages=5) == set()


def test_contents_page_detection():
    toc = "\n".join(f"1.{n} Section {n} ........................ {n}" for n in range(6))
    assert is_contents_page(toc)
    assert not is_contents_page("A normal page with a sentence... and an ellipsis.")


def test_clean_pdf_pages_end_to_end():
    toc = "\n".join(f"Chapter {n} .................. {n}" for n in range(6))
    body = [f"{n}\nMy Thesis | Chapter 3\nBody text of page {n}." for n in range(1, 9)]
    result = clean_pdf_pages([toc, toc, *body])

    assert result.pages[0] == ""
    assert result.pages[1] == ""
    assert result.contents_pages_skipped == 2
    assert result.page_numbers_removed == 8
    assert result.running_headers == ["my thesis | chapter 3"]
    assert result.pages[2] == "Body text of page 1."


def test_pdf_loader_strips_page_numbers(tmp_path):
    f = tmp_path / "a.pdf"
    pdf = pymupdf.open()
    for n in range(1, 4):
        page = pdf.new_page()
        page.insert_text((72, 40), str(n))
        page.insert_text((72, 100), f"Real content of page {n}")
    pdf.save(f)
    pdf.close()

    docs = load_file(f, "p1")
    assert [d.text for d in docs] == [f"Real content of page {n}" for n in (1, 2, 3)]
    assert [d.metadata["page"] for d in docs] == [1, 2, 3]  # PDF page numbers stay stable


def test_word_bullets_become_dashes():
    assert clean_text("\uf0b7 Short answers") == "- Short answers"


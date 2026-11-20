import pymupdf
import pytest
from docx import Document as DocxDocument

from src.core.ingestion.registry import (
    UnsupportedFileError,
    load_file,
    supported_extensions,
)


def test_supported_extensions():
    assert {"pdf", "txt", "md", "csv", "docx", "html"} <= supported_extensions()


def test_unsupported_extension(tmp_path):
    f = tmp_path / "a.exe"
    f.write_bytes(b"x")
    with pytest.raises(UnsupportedFileError):
        load_file(f, "p1")


def test_txt(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world", encoding="utf-8")
    docs = load_file(f, "p1")
    assert len(docs) == 1
    assert docs[0].text == "hello world"
    assert docs[0].metadata["format"] == "txt"


def test_empty_txt_gives_no_documents(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("   \n  ", encoding="utf-8")
    assert load_file(f, "p1") == []


def test_csv_keeps_rows(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("name,age\nAli,30\n", encoding="utf-8")
    assert load_file(f, "p1")[0].text == "name,age\nAli,30"


def test_pdf_one_document_per_page(tmp_path):
    f = tmp_path / "a.pdf"
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "First page")
    pdf.new_page().insert_text((72, 72), "Second page")
    pdf.save(f)
    pdf.close()

    docs = load_file(f, "p1")
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert "First page" in docs[0].text


def test_docx_paragraphs_and_tables(tmp_path):
    f = tmp_path / "a.docx"
    d = DocxDocument()
    d.add_paragraph("Hello docx")
    table = d.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "a"
    table.cell(0, 1).text = "b"
    d.save(str(f))

    text = load_file(f, "p1")[0].text
    assert "Hello docx" in text
    assert "a | b" in text


def test_html_removes_scripts_and_keeps_title(tmp_path):
    f = tmp_path / "a.html"
    f.write_text(
        "<html><head><title>My Page</title><script>var x=1;</script></head>"
        "<body><nav>Menu</nav><p>Real content</p></body></html>",
        encoding="utf-8",
    )
    doc = load_file(f, "p1")[0]
    assert "Real content" in doc.text
    assert "var x" not in doc.text
    assert "Menu" not in doc.text
    assert doc.metadata["title"] == "My Page"


def test_ids_are_stable(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    assert load_file(f, "p1")[0].id == load_file(f, "p1")[0].id

from src.core.chunking.fixed import FixedSizeChunker
from src.core.ingestion.processing import process_project


def test_broken_file_is_reported_and_others_continue(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "bad.pdf").write_bytes(b"this is not a pdf")
    (raw / "ok.txt").write_text("hello world", encoding="utf-8")

    result = process_project(raw, tmp_path / "out", "p1", FixedSizeChunker(100, 10))

    assert result.failed_files == ["bad.pdf"]
    assert result.files_processed == 1

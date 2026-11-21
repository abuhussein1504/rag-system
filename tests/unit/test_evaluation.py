from src.core.evaluation.golden import GoldItem, is_relevant, original_name
from src.core.evaluation.metrics import aggregate, first_relevant_rank
from src.core.evaluation.runner import evaluate
from src.models.enums import RetrievalSource
from src.models.schemas import Chunk, RetrievedChunk


def make_chunk(start, end, source="abc123def456_a.txt", page=None) -> Chunk:
    meta = {"source": source}
    if page is not None:
        meta["page"] = page
    return Chunk(
        id=f"c{start}", doc_id="d", text="x" * (end - start),
        start_char=start, end_char=end, metadata=meta,
    )


GOLD = GoldItem(id="q001", question="What is this?", doc="a.txt", start_char=100, end_char=300)


def test_original_name_removes_upload_prefix():
    assert original_name("afb14667855d_clean-code.pdf") == "clean-code.pdf"
    assert original_name("plain.txt") == "plain.txt"


def test_same_range_is_relevant():
    assert is_relevant(make_chunk(100, 300), GOLD)


def test_other_file_is_not_relevant():
    assert not is_relevant(make_chunk(100, 300, source="abc123def456_b.txt"), GOLD)


def test_other_page_is_not_relevant():
    assert not is_relevant(make_chunk(100, 300, page=2), GOLD)


def test_tiny_overlap_is_not_relevant():
    assert not is_relevant(make_chunk(290, 600), GOLD)


def test_no_overlap_is_not_relevant():
    assert not is_relevant(make_chunk(400, 600), GOLD)


def test_small_chunk_inside_gold_range_is_relevant():
    assert is_relevant(make_chunk(150, 250), GOLD)


def test_first_relevant_rank():
    assert first_relevant_rank([False, True, True]) == 2
    assert first_relevant_rank([False, False]) is None


def test_aggregate():
    m = aggregate([1, 2, None, 6])
    assert m["hit@1"] == 0.25
    assert m["hit@3"] == 0.5
    assert m["hit@10"] == 0.75
    assert m["mrr"] == 0.4167


def test_evaluate_with_fake_retriever():
    def retrieve(query, k):
        good = RetrievedChunk(chunk=make_chunk(100, 300), score=0.9, source=RetrievalSource.DENSE)
        return [good]

    report = evaluate([GOLD], retrieve, k=5)
    assert report["questions"][0]["rank"] == 1
    assert report["metrics"]["hit@1"] == 1.0


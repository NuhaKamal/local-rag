import pytest

from local_rag.chunker import chunk_pages, chunk_text
from local_rag.loader import Page


def test_hand_traced_example():
    # the example we traced by hand in Phase 4: size 10, overlap 3 -> step 7
    text = "abcdefghijklmnopqrstuvwxy"
    assert chunk_text(text, chunk_size=10, overlap=3) == [
        "abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxy",
    ]


def test_neighbouring_chunks_share_the_overlap():
    chunks = chunk_text("abcdefghijklmnopqrstuvwxy", 10, 3)
    for before, after in zip(chunks, chunks[1:]):
        assert before[-3:] == after[:3]


def test_short_text_is_a_single_chunk():
    assert chunk_text("hello", chunk_size=800, overlap=100) == ["hello"]


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []


@pytest.mark.parametrize("size, overlap", [(0, 0), (-5, 0), (10, 10), (10, 11), (10, -1)])
def test_bad_settings_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", size, overlap)


def test_chunk_pages_keeps_metadata():
    pages = [Page(text="a" * 25, source="book.pdf", page_number=7)]
    chunks = chunk_pages(pages, chunk_size=10, overlap=3)
    assert len(chunks) == 4
    assert all(c.source == "book.pdf" and c.page_number == 7 for c in chunks)
    assert [c.chunk_id for c in chunks] == [
        "book.pdf:p7:c0", "book.pdf:p7:c1", "book.pdf:p7:c2", "book.pdf:p7:c3",
    ]
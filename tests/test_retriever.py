import pytest

from local_rag.chunker import Chunk
from local_rag.retrieval.retriever import retrieve


class FakeEmbedder:
    """Stands in for the real model: always returns the query vector we give it."""

    def __init__(self, query_vector):
        self.query_vector = query_vector

    def embed_query(self, question):
        return self.query_vector


def make_chunks(n):
    return [
        Chunk(text=f"chunk {i}", source="t.pdf", page_number=i + 1,
              chunk_index=0, chunk_id=f"t.pdf:p{i + 1}:c0")
        for i in range(n)
    ]


def test_ranks_by_cosine_similarity():
    vecs = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    results = retrieve("q", vecs, make_chunks(3), FakeEmbedder([1.0, 0.0]), k=3)
    assert [r["chunk_id"] for r in results] == ["t.pdf:p1:c0", "t.pdf:p3:c0", "t.pdf:p2:c0"]
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.7071, abs=1e-3)
    assert results[2]["score"] == pytest.approx(0.0, abs=1e-6)


def test_k_limits_the_results():
    vecs = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    assert len(retrieve("q", vecs, make_chunks(3), FakeEmbedder([1.0, 0.0]), k=2)) == 2


def test_k_larger_than_chunk_count_returns_all():
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    assert len(retrieve("q", vecs, make_chunks(2), FakeEmbedder([1.0, 0.0]), k=10)) == 2


def test_cosine_ignores_vector_length():
    # same direction, very different length -> same score
    vecs = [[1.0, 0.0], [5.0, 0.0]]
    results = retrieve("q", vecs, make_chunks(2), FakeEmbedder([1.0, 0.0]), k=2)
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(1.0)


def test_chunks_and_vectors_must_match_in_length():
    with pytest.raises(ValueError):
        retrieve("q", [[1.0, 0.0]], make_chunks(2), FakeEmbedder([1.0, 0.0]))


def test_empty_index_returns_nothing():
    assert retrieve("q", [], [], FakeEmbedder([1.0, 0.0])) == []
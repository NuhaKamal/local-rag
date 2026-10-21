import pytest

from rag_from_scratch.models import Chunk
from rag_from_scratch.retriever import retrieve


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


def ids(hits):
    return [chunk.chunk_id for chunk, _score in hits]


def test_ranks_by_cosine_similarity():
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    hits = retrieve("q", make_chunks(3), vectors, FakeEmbedder([1.0, 0.0]), top_k=3)
    assert ids(hits) == ["t.pdf:p1:c0", "t.pdf:p3:c0", "t.pdf:p2:c0"]
    assert hits[0][1] == pytest.approx(1.0)
    assert hits[1][1] == pytest.approx(0.7071, abs=1e-3)
    assert hits[2][1] == pytest.approx(0.0, abs=1e-6)


def test_top_k_limits_the_results():
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    assert len(retrieve("q", make_chunks(3), vectors, FakeEmbedder([1.0, 0.0]), top_k=2)) == 2


def test_top_k_larger_than_chunk_count_returns_all():
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    assert len(retrieve("q", make_chunks(2), vectors, FakeEmbedder([1.0, 0.0]), top_k=10)) == 2


def test_cosine_ignores_vector_length():
    vectors = [[1.0, 0.0], [5.0, 0.0]]
    hits = retrieve("q", make_chunks(2), vectors, FakeEmbedder([1.0, 0.0]), top_k=2)
    assert hits[0][1] == pytest.approx(1.0)
    assert hits[1][1] == pytest.approx(1.0)


def test_chunks_and_vectors_must_match_in_length():
    with pytest.raises(ValueError):
        retrieve("q", make_chunks(2), [[1.0, 0.0]], FakeEmbedder([1.0, 0.0]))


def test_empty_index_returns_nothing():
    assert retrieve("q", [], [], FakeEmbedder([1.0, 0.0])) == []
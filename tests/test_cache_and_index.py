import pytest

from local_rag.chunker import Chunk
from local_rag.embedder import EmbeddingCache
from local_rag.index import load_index, save_index


def test_cache_key_is_deterministic_and_sensitive():
    key = EmbeddingCache.make_key("model-a", "hello")
    assert key == EmbeddingCache.make_key("model-a", "hello")
    assert key != EmbeddingCache.make_key("model-b", "hello")   # different model
    assert key != EmbeddingCache.make_key("model-a", "hello!")  # different text


def test_cache_counts_hits_and_misses(tmp_path):
    cache = EmbeddingCache(tmp_path / "cache.json")
    assert cache.get("x") is None
    cache.put("x", [1.0, 2.0])
    assert cache.get("x") == [1.0, 2.0]
    assert (cache.hits, cache.misses) == (1, 1)


def test_cache_survives_save_and_reload(tmp_path):
    path = tmp_path / "cache.json"
    cache = EmbeddingCache(path)
    cache.put("x", [0.5])
    cache.save()
    assert EmbeddingCache(path).get("x") == [0.5]


def test_index_round_trip(tmp_path):
    chunks = [Chunk("some text", "a.pdf", 1, 0, "a.pdf:p1:c0")]
    save_index(chunks, [[0.1, 0.2]], "model.gguf", tmp_path / "index.json")
    loaded_chunks, vecs, model = load_index(tmp_path / "index.json")
    assert loaded_chunks == chunks
    assert vecs == [[0.1, 0.2]]
    assert model == "model.gguf"


def test_saving_mismatched_lengths_fails(tmp_path):
    chunks = [Chunk("t", "a.pdf", 1, 0, "a.pdf:p1:c0")]
    with pytest.raises(ValueError):
        save_index(chunks, [], "model.gguf", tmp_path / "index.json")


def test_loading_a_missing_index_fails_helpfully(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "nope.json")
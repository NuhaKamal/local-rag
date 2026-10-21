"""
embedder.py: Stage 3. Turn text into vectors (embeddings), with a disk cache.

CONTRACT:
  INPUT : text (a chunk, or a question)
  OUTPUT: list[float], 768 numbers; similar meanings give nearby vectors

HOW IT WORKS:
  1. nomic expects a task prefix: "search_document: " for chunks and
     "search_query: " for questions.
  2. Cache key = sha256(model name + text). Same text and same model: reuse the
     saved vector instead of recomputing it.
  3. embed_chunks() saves the cache to .cache/embeddings.json when it finishes.

WHY IT IS SEPARATE:
  Embedding is the slowest, most model-specific step. Swapping the embedding
  model, or the cache, only touches this file.

RUN IT ON ITS OWN:
  uv run python -m rag_from_scratch.embedder     (needs the nomic model in models/)
"""

import hashlib
import json
import logging
import tempfile
import time
from pathlib import Path

from rag_from_scratch.models import Chunk

logger = logging.getLogger(__name__)

EMBED_MODEL_PATH = Path("models/nomic-embed-text-v1.5.Q4_K_M.gguf")
CACHE_PATH = Path(".cache/embeddings.json")

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbeddingCache:
    """A dictionary saved to disk: key -> vector."""

    def __init__(self, path: Path = CACHE_PATH):
        self.path = Path(path)
        self.data: dict[str, list[float]] = {}
        self.hits = 0
        self.misses = 0
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            logger.debug("cache loaded: %d vectors from %s", len(self.data), self.path)

    @staticmethod
    def make_key(model_name: str, text: str) -> str:
        """Fingerprint of (model, text)."""
        raw = f"{model_name}\n{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, key: str) -> list[float] | None:
        vector = self.data.get(key)
        if vector is None:
            self.misses += 1
        else:
            self.hits += 1
        return vector

    def put(self, key: str, vector: list[float]) -> None:
        self.data[key] = vector

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data), encoding="utf-8")
        logger.debug("cache saved: %d vectors to %s", len(self.data), self.path)


class Embedder:
    """Wraps the nomic model. Always checks the cache before computing."""

    def __init__(self, model_path: Path = EMBED_MODEL_PATH, cache: EmbeddingCache | None = None):
        from llama_cpp import Llama  # imported here: heavy, and only needed when the model loads

        self.model_name = Path(model_path).name
        logger.info("loading embedding model %s", self.model_name)
        self.llm = Llama(
            model_path=str(model_path),
            embedding=True,
            n_ctx=2048,
            n_batch=2048,
            n_ubatch=2048,
            verbose=False,
        )
        self.cache = cache or EmbeddingCache()

    def embed(self, text: str) -> list[float]:
        """Embed any text (cache first)."""
        key = self.cache.make_key(self.model_name, text)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        vector = self.llm.create_embedding(text)["data"][0]["embedding"]
        self.cache.put(key, vector)
        logger.debug("cache miss: computed a vector for %d characters", len(text))
        return vector

    def embed_query(self, question: str) -> list[float]:
        """Embed a user question (with the query prefix)."""
        return self.embed(QUERY_PREFIX + question)

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed every chunk (with the document prefix), then save the cache."""
        vectors = []
        for i, chunk in enumerate(chunks, start=1):
            vectors.append(self.embed(DOC_PREFIX + chunk.text))
            if i % 25 == 0:
                logger.info("embedded %d/%d chunks", i, len(chunks))
        self.cache.save()
        logger.info("embedding done: %d chunks (cache hits %d, misses %d)",
                    len(chunks), self.cache.hits, self.cache.misses)
        return vectors


def _demo() -> None:
    print("== embedder.py demo ==")
    if not EMBED_MODEL_PATH.exists():
        print(f"Model not found: {EMBED_MODEL_PATH}")
        print("Download it first (see the README, Quick start), then run this again.")
        return

    text = "A functional movement is a pattern the body uses in daily life."
    print(f"INPUT : {text!r}")
    print(f"        (the prefix {DOC_PREFIX!r} is added before embedding)")

    with tempfile.TemporaryDirectory() as folder:  # temporary cache: your real cache is untouched
        embedder = Embedder(cache=EmbeddingCache(Path(folder) / "demo_cache.json"))
        for attempt in (1, 2):
            hits_before = embedder.cache.hits
            start = time.perf_counter()
            vector = embedder.embed(DOC_PREFIX + text)
            ms = (time.perf_counter() - start) * 1000
            source = "FROM CACHE" if embedder.cache.hits > hits_before else "COMPUTED"
            print(f"  attempt {attempt}: {source:<10} {ms:9.2f} ms")

    length = sum(x * x for x in vector) ** 0.5
    first = [round(x, 3) for x in vector[:5]]
    print(f"OUTPUT: a vector of {len(vector)} numbers; first 5 = {first}; length = {length:.2f}")


if __name__ == "__main__":
    _demo()
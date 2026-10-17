"""Turn text into vectors with nomic, using a disk cache."""

import hashlib
import json
from pathlib import Path

from llama_cpp import Llama

from local_rag.chunker import Chunk

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


class Embedder:
    """Wraps the nomic model. Always checks the cache before computing."""

    def __init__(self, model_path: Path = EMBED_MODEL_PATH, cache: EmbeddingCache | None = None):
        self.model_name = Path(model_path).name
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
                print(f"  embedded {i}/{len(chunks)}")
        self.cache.save()
        return vectors
"""Save and load the searchable index: chunks and their vectors, index-aligned."""

import json
from dataclasses import asdict
from pathlib import Path

from local_rag.chunker import Chunk
from local_rag.config import INDEX_PATH


def save_index(chunks: list[Chunk],
               vecs: list[list[float]],
               embed_model: str,
               path: Path = INDEX_PATH) -> None:
    """Write chunks + vectors (+ the model that made them) to one JSON file."""
    if len(chunks) != len(vecs):
        raise ValueError(f"{len(chunks)} chunks but {len(vecs)} vectors: they must match")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embed_model": embed_model,
        "chunks": [asdict(c) for c in chunks],
        "vectors": vecs,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_index(path: Path = INDEX_PATH) -> tuple[list[Chunk], list[list[float]], str]:
    """Read the index back: (chunks, vectors, embed_model)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no index found at {path}. Run 'local-rag ingest' first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = [Chunk(**d) for d in payload["chunks"]]
    return chunks, payload["vectors"], payload["embed_model"]
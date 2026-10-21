"""
index.py: Stage 4. Save and load the searchable knowledge base.

CONTRACT:
  save_index(chunks, vectors, embed_model, path)  -> writes one JSON file
  load_index(path)                                -> (chunks, vectors, embed_model)

The file holds three things that must stay together:
  - the chunks (text + metadata)
  - the vectors, in the SAME ORDER (vector i belongs to chunk i)
  - the name of the embedding model that made the vectors

WHY IT IS SEPARATE:
  Ingest (slow, runs once) and ask (fast, runs every question) only need to
  agree on this file. Replacing JSON with a vector database later means
  rewriting only this module.

RUN IT ON ITS OWN:
  uv run python -m rag_from_scratch.index
"""

import json
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path

from rag_from_scratch.config import INDEX_PATH
from rag_from_scratch.models import Chunk

logger = logging.getLogger(__name__)


def save_index(
    chunks: list[Chunk],
    vectors: list[list[float]],
    embed_model: str,
    path: Path = INDEX_PATH,
) -> None:
    """Write chunks and vectors side by side in one JSON file."""
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors: they must match")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embed_model": embed_model,
        "chunks": [asdict(chunk) for chunk in chunks],
        "vectors": vectors,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    logger.info("saved index: %d chunks to %s", len(chunks), path)


def load_index(path: Path = INDEX_PATH) -> tuple[list[Chunk], list[list[float]], str]:
    """Read the index back. Returns (chunks, vectors, embed_model)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No index at {path}. Run 'rag ingest' first.")

    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = [Chunk(**item) for item in payload["chunks"]]
    vectors = payload["vectors"]
    logger.info("loaded index: %d chunks from %s", len(chunks), path)
    return chunks, vectors, payload["embed_model"]


def _demo() -> None:
    print("== index.py demo ==")
    chunks = [
        Chunk("Squats train the legs.", "demo.pdf", 1, 0, "demo.pdf:p1:c0"),
        Chunk("Planks train the core.", "demo.pdf", 2, 0, "demo.pdf:p2:c0"),
    ]
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # tiny fake vectors, just to show the shape
    print(f"INPUT : {len(chunks)} chunks, {len(vectors)} vectors, model 'demo-model'")

    with tempfile.TemporaryDirectory() as folder:  # temporary file: your real index is untouched
        path = Path(folder) / "demo_index.json"
        save_index(chunks, vectors, "demo-model", path)
        print(f"        saved {path.stat().st_size} bytes")

        loaded_chunks, loaded_vectors, model = load_index(path)
        print(f"OUTPUT: model={model!r}, {len(loaded_chunks)} chunks, {len(loaded_vectors)} vectors")
        print(f"        chunk 1 = {loaded_chunks[1].chunk_id!r}, vector 1 = {loaded_vectors[1]}")
        print(f"        round trip identical: {loaded_chunks == chunks and loaded_vectors == vectors}")

        print("ERROR CASE: loading a file that does not exist")
        try:
            load_index(Path(folder) / "missing.json")
        except FileNotFoundError as error:
            print(f"        {error}")


if __name__ == "__main__":
    _demo()
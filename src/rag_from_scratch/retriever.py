"""
retriever.py: Stage 5. Find the chunks closest in meaning to a question.

CONTRACT:
  INPUT : question (str), chunks, vectors, an embedder (anything with .embed_query)
  OUTPUT: list of (Chunk, score), best first. score is cosine similarity.

THE MATH (cosine similarity):
  cos(a, b) = (a . b) / (|a| * |b|)
  - a . b   : multiply matching numbers, add them up
  - |a|     : the length of the vector
  Dividing by both lengths leaves only the ANGLE between the vectors:
  1.0 = same direction (same meaning), 0 = unrelated, -1 = opposite.
  torch.nn.functional.cosine_similarity does this for every chunk at once.

WHY IT IS SEPARATE:
  It needs no model. Give it vectors and a way to embed the question and it works.

RUN IT ON ITS OWN:
  uv run python -m rag_from_scratch.retriever
"""

import logging

from rag_from_scratch.config import TOP_K
from rag_from_scratch.models import Chunk

logger = logging.getLogger(__name__)


def retrieve(
    question: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
    embedder,
    top_k: int = TOP_K,
) -> list[tuple[Chunk, float]]:
    """Return the top_k most similar chunks as (chunk, score), best first."""
    import torch  # imported here: heavy, and only needed when we actually search
    import torch.nn.functional as F

    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors: they must match")
    if not chunks:
        return []

    query = torch.tensor(embedder.embed_query(question))   # shape: (768,)
    matrix = torch.tensor(vectors)                          # shape: (n_chunks, 768)

    scores = F.cosine_similarity(query, matrix, dim=1)      # shape: (n_chunks,)
    k = min(top_k, len(chunks))
    best = torch.topk(scores, k)

    hits = [(chunks[i], float(s)) for s, i in zip(best.values, best.indices.tolist())]
    logger.info("retrieved %d of %d chunks, best score %.3f", k, len(chunks), hits[0][1])
    for rank, (chunk, score) in enumerate(hits, start=1):
        logger.debug("  #%d %s score=%.3f", rank, chunk.chunk_id, score)
    return hits


class _FakeEmbedder:
    """Stand-in for the real embedder, so this demo needs no model."""

    def embed_query(self, question: str) -> list[float]:
        return [0.9, 0.1, 0.0]  # "points mostly toward legs"


def _demo() -> None:
    print("== retriever.py demo ==")
    chunks = [
        Chunk("Squats train the legs.", "demo.pdf", 1, 0, "demo.pdf:p1:c0"),
        Chunk("Planks train the core.", "demo.pdf", 2, 0, "demo.pdf:p2:c0"),
        Chunk("Running uses legs and lungs.", "demo.pdf", 3, 0, "demo.pdf:p3:c0"),
    ]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]]  # tiny fake 3-number vectors
    question = "Which exercise works the legs?"

    print(f"INPUT : question = {question!r}")
    for chunk, vector in zip(chunks, vectors):
        print(f"        {chunk.chunk_id}  {vector}  {chunk.text}")

    hits = retrieve(question, chunks, vectors, _FakeEmbedder(), top_k=3)
    print("OUTPUT: ranked by cosine similarity")
    for rank, (chunk, score) in enumerate(hits, start=1):
        print(f"        #{rank}  score={score:.3f}  {chunk.chunk_id}  {chunk.text}")


if __name__ == "__main__":
    _demo()
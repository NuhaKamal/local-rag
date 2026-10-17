"""
retrieval/retriever.py — Stage 4: find the chunks most relevant to a query.

HOW IT WORKS:
  1. Embed the query with the same model used for chunks.
  2. Compute cosine similarity between query vector and all chunk vectors.
  3. Return the top-K highest-scoring chunks as evidence for the LLM.

WHY COSINE SIMILARITY (not Euclidean distance)?
  Embedding vectors vary in magnitude. Cosine similarity normalises for
  magnitude — it measures the *angle* between vectors, which is a purer
  proxy for semantic direction (meaning).

  Score =  1.0 → same direction   (very relevant)
  Score =  0.0 → perpendicular    (unrelated)
  Score = -1.0 → opposite

THE MATH (what torch computes for us):
  cosine(a, b) = (a · b) / (|a| * |b|)

DEVICE NOTE:
  Our embedder (llama.cpp) returns plain Python lists, which become CPU
  tensors. We still move the chunk tensor to the query tensor's device
  with .to(q_vec.device) — never hardcode 'cpu' or 'cuda' — so the code
  keeps working if the query ever lands on a GPU.
"""

from dataclasses import asdict

import torch

from local_rag.chunker import Chunk
from local_rag.config import TOP_K
from local_rag.embedder import Embedder


def retrieve(query: str,
             vecs: list[list[float]],
             chunks: list[Chunk],
             embedder: Embedder,
             k: int = TOP_K) -> list[dict]:
    """
    Embed `query` and return the top-k most similar chunks.

    Args:
        query    : the user's question
        vecs     : (N, D) chunk embeddings, as returned by Embedder.embed_chunks
        chunks   : list of Chunk objects — must be index-aligned with vecs
        embedder : the same Embedder that produced vecs
        k        : number of chunks to return

    Returns:
        list of chunk dicts (text, source, page_number, chunk_index, chunk_id)
        each with an extra "score" key, sorted highest → lowest similarity.
    """
    if len(chunks) != len(vecs):
        raise ValueError(f"{len(chunks)} chunks but {len(vecs)} vectors: they must match")
    if not chunks:
        return []

    # embed query and wrap it as a tensor
    q_vec = torch.tensor(embedder.embed_query(query), dtype=torch.float32)   # (D,)
    c_vec = torch.tensor(vecs, dtype=torch.float32).to(q_vec.device)         # (N, D) 


    # single matrix operation: no Python loop over chunks
    scores = torch.nn.functional.cosine_similarity(
        q_vec.unsqueeze(0),   # (1, D) → broadcast over all chunk rows
        c_vec,                # (N, D)
    )                         # → (N,) one score per chunk

    top_scores, top_idx = torch.topk(scores, k=min(k, len(chunks)))

    results = []
    for score, i in zip(top_scores.tolist(), top_idx.tolist()):
        results.append({**asdict(chunks[i]), "score": round(score, 4)})

    return results   # already sorted highest → lowest by torch.topk

"""
A returned item looks like:
{"text": "...", "source": "book.pdf", "page_number": 45,
 "chunk_index": 0, "chunk_id": "book.pdf:p45:c0", "score": 0.7312}
"""
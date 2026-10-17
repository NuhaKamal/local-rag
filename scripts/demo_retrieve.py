"""Quick demo: show the top chunks for a question. (The CLI replaces this in Phase 8.)"""

import sys

from local_rag.chunker import chunk_pages
from local_rag.embedder import Embedder
from local_rag.loader import load_pdf
from local_rag.retrieval.retriever import retrieve

question = " ".join(sys.argv[1:]) or "What does a score of 3 mean?"

meta, pages = load_pdf("data/book.pdf")
chunks = chunk_pages(pages)

embedder = Embedder()
print(f"Embedding {len(chunks)} chunks (cached ones are instant)...")
vecs = embedder.embed_chunks(chunks)
print(f"cache hits: {embedder.cache.hits}, misses: {embedder.cache.misses}\n")

print(f"Question: {question}\n")
for rank, r in enumerate(retrieve(question, vecs, chunks, embedder), start=1):
    preview = r["text"][:200].replace("\n", " ")
    print(f"#{rank}  score={r['score']}  {r['source']} p.{r['page_number']} ({r['chunk_id']})")
    print(f"    {preview}\n")
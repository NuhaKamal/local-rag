"""Full RAG demo: retrieve, build prompt, generate. (The CLI replaces this in Phase 8.)"""

import sys
import time

from local_rag.chunker import chunk_pages
from local_rag.embedder import Embedder
from local_rag.generation.generator import Generator, build_messages
from local_rag.loader import load_pdf
from local_rag.retrieval.retriever import retrieve

args = [a for a in sys.argv[1:] if a != "--show-prompt"]
show_prompt = "--show-prompt" in sys.argv
question = " ".join(args) or "What does a score of 3 mean?"

meta, pages = load_pdf("data/book.pdf")
chunks = chunk_pages(pages)

embedder = Embedder()
vecs = embedder.embed_chunks(chunks)  # instant when cached

results = retrieve(question, vecs, chunks, embedder)
print(f"Question: {question}\n")
print("Retrieved:")
for r in results:
    print(f"  p.{r['page_number']:>3}  score={r['score']}")

if show_prompt:
    print("\n----- PROMPT SENT TO THE MODEL -----")
    for m in build_messages(question, results):
        print(f"[{m['role']}]\n{m['content']}\n")
    print("------------------------------------")

generator = Generator()
start = time.time()
out = generator.answer(question, results)

print(f"\nANSWER:\n{out['answer']}\n")
print("Sources:", [(s["page_number"], s["score"]) for s in out["sources"]])
print(f"Prompt tokens: {out['prompt_tokens']} / 2048   |   {round(time.time() - start, 1)} s")
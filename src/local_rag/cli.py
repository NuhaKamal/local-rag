"""
cli.py — Stage 7: the command-line interface.

USAGE:
  local-rag ingest [PDF ...]       read PDFs, chunk, embed, save the index
                                   (no PDF given = every PDF in data/)
  local-rag ask "your question"    retrieve the best chunks and answer with TinyLlama

DESIGN:
  ingest = slow, run once per document (embedding is cached)
  ask    = fast, run many times (reads the saved index)
  Heavy libraries (llama.cpp, torch) are imported inside the commands,
  so `local-rag --help` stays instant.
"""

import argparse
import sys
import time
from pathlib import Path

from local_rag.config import CHUNK_SIZE, INDEX_PATH, OVERLAP, TOP_K


def cmd_ingest(args: argparse.Namespace) -> int:
    from local_rag.chunker import chunk_pages
    from local_rag.embedder import Embedder
    from local_rag.index import save_index
    from local_rag.loader import load_pdf

    pdfs = args.pdfs or sorted(Path("data").glob("*.pdf"))
    if not pdfs:
        print("Error: no PDF given and none found in data/", file=sys.stderr)
        return 1

    all_chunks = []
    for pdf in pdfs:
        if not pdf.exists():
            print(f"Error: file not found: {pdf}", file=sys.stderr)
            return 1
        meta, pages = load_pdf(pdf)
        if not pages:
            print(f"Error: no extractable text in {pdf.name} (scanned PDF?)", file=sys.stderr)
            return 1
        try:
            chunks = chunk_pages(pages, args.chunk_size, args.overlap)
        except ValueError as err:
            print(f"Error: {err}", file=sys.stderr)
            return 1
        print(f"{pdf.name}: {meta['num_pages']} pages ({len(pages)} with text) -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\nEmbedding {len(all_chunks)} chunks (cached ones are instant)...")
    start = time.time()
    embedder = Embedder()
    vecs = embedder.embed_chunks(all_chunks)
    save_index(all_chunks, vecs, embedder.model_name)

    print(f"Cache hits: {embedder.cache.hits}, misses: {embedder.cache.misses}")
    print(f"Index saved to {INDEX_PATH} ({time.time() - start:.1f} s)")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from local_rag.embedder import Embedder
    from local_rag.generation.generator import Generator, build_messages
    from local_rag.index import load_index
    from local_rag.retrieval.retriever import retrieve

    try:
        chunks, vecs, index_model = load_index(INDEX_PATH)
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    embedder = Embedder()
    if embedder.model_name != index_model:
        print(
            f"Error: the index was built with '{index_model}' but the current "
            f"embedder is '{embedder.model_name}'. Run 'local-rag ingest' again.",
            file=sys.stderr,
        )
        return 1

    question = " ".join(args.question)
    results = retrieve(question, vecs, chunks, embedder, k=args.top_k)

    print(f"\nQuestion: {question}\n")
    print("Retrieved:")
    for rank, r in enumerate(results, start=1):
        print(f"  #{rank}  p.{r['page_number']:<4} score={r['score']}  {r['source']}")

    if args.show_prompt:
        print("\n----- PROMPT SENT TO THE MODEL -----")
        for m in build_messages(question, results):
            print(f"[{m['role']}]\n{m['content']}\n")
        print("------------------------------------")

    if args.no_llm:
        return 0

    generator = Generator()
    start = time.time()
    out = generator.answer(question, results)

    print(f"\nAnswer:\n{out['answer']}\n")
    if out["sources"]:
        pages = ", ".join(f"{s['source']} p.{s['page_number']}" for s in out["sources"])
        print(f"Sources: {pages}")
    print(f"({out['prompt_tokens']} prompt tokens, {time.time() - start:.1f} s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-rag",
        description="Ask questions about your PDFs, fully offline (llama.cpp + TinyLlama).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="read PDFs, chunk, embed and save the index")
    ingest.add_argument("pdfs", nargs="*", type=Path,
                        help="PDF files (default: every PDF in data/)")
    ingest.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"characters per chunk (default {CHUNK_SIZE})")
    ingest.add_argument("--overlap", type=int, default=OVERLAP,
                        help=f"characters shared between chunks (default {OVERLAP})")
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="answer a question from the ingested documents")
    ask.add_argument("question", nargs="+", help="your question (put it in quotes)")
    ask.add_argument("-k", "--top-k", type=int, default=TOP_K,
                     help=f"chunks to retrieve (default {TOP_K})")
    ask.add_argument("--no-llm", action="store_true",
                     help="only show the retrieved chunks, skip the model")
    ask.add_argument("--show-prompt", action="store_true",
                     help="print the exact prompt sent to the model")
    ask.set_defaults(func=cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
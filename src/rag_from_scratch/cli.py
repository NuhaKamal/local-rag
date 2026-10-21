"""
cli.py: the command-line interface. The only file that knows the order of the stages.

USAGE:
  rag ingest [PDF ...]       read PDFs, chunk, embed, save the index
                             (no PDF given = every PDF in data/)
  rag ask "your question"    retrieve the best chunks and answer with TinyLlama
  add -v for progress messages, -vv for every detail (always saved to .cache/rag.log)

THE TWO WORKFLOWS:
  ingest:  loader -> chunker -> embedder -> index          (slow, once per document)
  ask:     index -> retriever (uses embedder) -> generator (fast, every question)

OUTPUT vs LOGGING:
  print()  = the result the user asked for (answer, sources)
  logger   = what the program is doing (stages, counts, timings)

Heavy libraries (llama.cpp, torch) are imported inside the commands,
so `rag --help` stays instant.

RUN IT ON ITS OWN:
  uv run rag --help
  uv run python -m rag_from_scratch.cli --help
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from rag_from_scratch.config import CHUNK_SIZE, INDEX_PATH, OVERLAP, TOP_K
from rag_from_scratch.logs import setup_logging

logger = logging.getLogger(__name__)


def cmd_ingest(args: argparse.Namespace) -> int:
    from rag_from_scratch.chunker import chunk_pages
    from rag_from_scratch.embedder import Embedder
    from rag_from_scratch.index import save_index
    from rag_from_scratch.loader import load_pdf

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
    vectors = embedder.embed_chunks(all_chunks)
    save_index(all_chunks, vectors, embedder.model_name)

    print(f"Cache hits: {embedder.cache.hits}, misses: {embedder.cache.misses}")
    print(f"Index saved to {INDEX_PATH} ({time.time() - start:.1f} s)")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from rag_from_scratch.index import load_index

    try:
        chunks, vectors, index_model = load_index(INDEX_PATH)
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    # heavy imports come after the cheap checks, so a missing index fails instantly
    from rag_from_scratch.embedder import Embedder
    from rag_from_scratch.generator import Generator, build_messages
    from rag_from_scratch.retriever import retrieve

    embedder = Embedder()
    if embedder.model_name != index_model:
        print(
            f"Error: the index was built with '{index_model}' but the current "
            f"embedder is '{embedder.model_name}'. Run 'rag ingest' again.",
            file=sys.stderr,
        )
        return 1

    question = " ".join(args.question)
    hits = retrieve(question, chunks, vectors, embedder, top_k=args.top_k)

    print(f"\nQuestion: {question}\n")
    print("Retrieved:")
    for rank, (chunk, score) in enumerate(hits, start=1):
        print(f"  #{rank}  p.{chunk.page_number:<4} score={score:.2f}  {chunk.source}")

    if args.show_prompt:
        print("\n----- PROMPT SENT TO THE MODEL -----")
        for m in build_messages(question, hits):
            print(f"[{m['role']}]\n{m['content']}\n")
        print("------------------------------------")

    if args.no_llm:
        return 0

    generator = Generator()
    start = time.time()
    out = generator.answer(question, hits)

    print(f"\nAnswer:\n{out['answer']}\n")
    if out["sources"]:
        pages = ", ".join(f"{s['source']} p.{s['page_number']}" for s in out["sources"])
        print(f"Sources: {pages}")
    print(f"({out['prompt_tokens']} prompt tokens, {time.time() - start:.1f} s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)  # options shared by every command
    common.add_argument("-v", "--verbose", action="count", default=0,
                        help="show progress messages (-v) or every detail (-vv)")

    parser = argparse.ArgumentParser(
        prog="rag",
        description="A RAG pipeline built from scratch: ask questions about your PDFs, fully offline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", parents=[common],
                            help="read PDFs, chunk, embed and save the index")
    ingest.add_argument("pdfs", nargs="*", type=Path,
                        help="PDF files (default: every PDF in data/)")
    ingest.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"characters per chunk (default {CHUNK_SIZE})")
    ingest.add_argument("--overlap", type=int, default=OVERLAP,
                        help=f"characters shared between chunks (default {OVERLAP})")
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", parents=[common],
                         help="answer a question from the ingested documents")
    ask.add_argument("question", nargs="+", help="your question (put it in quotes)")
    ask.add_argument("-k", "--top-k", type=int, default=TOP_K,
                     help=f"chunks to retrieve (default {TOP_K})")
    ask.add_argument("--no-llm", action="store_true",
                     help="only show the retrieved chunks, skip the model")
    ask.add_argument("--show-prompt", action="store_true",
                     help="print the exact prompt sent to the model")
    ask.set_defaults(func=cmd_ask)

    return parser


def screen_level(verbose: int) -> str:
    """0 -> only warnings and errors, -v -> progress (INFO), -vv -> everything (DEBUG)."""
    return {0: "WARNING", 1: "INFO"}.get(verbose, "DEBUG")

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=screen_level(args.verbose))
    logger.debug("command=%s args=%s", args.command, vars(args))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
"""
chunker.py: Stage 2. Split pages into fixed-size, overlapping chunks.

CONTRACT:
  INPUT : list[Page], chunk_size, overlap
  OUTPUT: list[Chunk]  (each keeps source, page_number, chunk_index, chunk_id)

HOW IT WORKS:
  A window of chunk_size characters slides over each page's text, moving
  forward by step = chunk_size - overlap, so neighbouring chunks share
  `overlap` characters. Chunks never cross a page, so every chunk has an
  exact page number for citations.

WHY IT IS SEPARATE:
  It only knows how to cut text. It needs no PDF, no model and no files, so it
  can be tested and tuned instantly, and replaced (sentence or paragraph
  chunking) without touching any other file.

RUN IT ON ITS OWN:
  uv run python -m rag_from_scratch.chunker
"""

import logging

from rag_from_scratch.models import Chunk, Page

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Cut text into windows of chunk_size characters that overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    step = chunk_size - overlap
    pieces = []
    for start in range(0, len(text), step):
        piece = text[start:start + chunk_size].strip()
        if piece:
            pieces.append(piece)
        if start + chunk_size >= len(text):  # reached the end: stop
            break
    return pieces


def chunk_pages(pages: list[Page], chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Chunk every page, attaching metadata to each piece."""
    chunks = []
    for page in pages:
        for i, piece in enumerate(chunk_text(page.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    text=piece,
                    source=page.source,
                    page_number=page.page_number,
                    chunk_index=i,
                    chunk_id=f"{page.source}:p{page.page_number}:c{i}",
                )
            )
    logger.info("chunked %d pages into %d chunks (size=%d, overlap=%d)",
                len(pages), len(chunks), chunk_size, overlap)
    return chunks


def _demo() -> None:
    text = (
        "Fixed-size chunking cuts text every N characters. Overlap repeats the end "
        "of one chunk at the start of the next, so an idea cut at a boundary still "
        "appears whole in one chunk."
    )
    size, overlap = 60, 15
    step = size - overlap
    page = Page(text=text, source="demo.pdf", page_number=1)

    print("== chunker.py demo ==")
    print(f"INPUT : 1 Page, {len(text)} characters; chunk_size={size}, overlap={overlap} (step={step})")
    chunks = chunk_pages([page], size, overlap)
    print(f"OUTPUT: {len(chunks)} chunks")
    for c in chunks:
        print(f"  {c.chunk_id:<14} starts at {c.chunk_index * step:>3}  {len(c.text):>2} chars  {c.text!r}")
    print(f"\nEach window moves forward by {step}, so neighbours share {overlap} characters.")


if __name__ == "__main__":
    _demo()
"""Split pages into fixed-size, overlapping chunks that keep their metadata."""

from dataclasses import dataclass

from local_rag.loader import Page


@dataclass
class Chunk:
    """A piece of text plus where it came from."""

    text: str
    source: str       # file name
    page_number: int  # page it came from
    chunk_index: int  # position within that page (0, 1, 2, ...)
    chunk_id: str     # unique id, e.g. "book.pdf:p45:c0"


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
    return chunks
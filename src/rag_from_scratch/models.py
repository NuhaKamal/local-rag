"""
models.py: the shared data shapes ("contracts") that flow between components.

Components never import each other to exchange data. They only agree on these
objects, which is what keeps them decoupled:

  loader.py    -> list[Page]
  chunker.py   -> list[Chunk]
  embedder.py  -> list[list[float]]   (one vector per chunk)
  retriever.py -> list[dict]          (chunk fields + "score")
  generator.py -> dict                ({"answer", "sources", "prompt_tokens"})

RUN IT ON ITS OWN:
  uv run python -m rag_from_scratch.models
"""

from dataclasses import dataclass


@dataclass
class Page:
    """The text of one PDF page, plus where it came from."""

    text: str
    source: str       # file name, e.g. "book.pdf"
    page_number: int  # 1-based, like in a PDF viewer


@dataclass
class Chunk:
    """A piece of text plus where it came from."""

    text: str
    source: str       # file name
    page_number: int  # page it came from
    chunk_index: int  # position within that page (0, 1, 2, ...)
    chunk_id: str     # unique id, e.g. "book.pdf:p45:c0"


def _demo() -> None:
    page = Page(text="Score of 3: Maintenance ...", source="book.pdf", page_number=45)
    chunk = Chunk(text="Score of 3: Maintenance ...", source="book.pdf",
                  page_number=45, chunk_index=0, chunk_id="book.pdf:p45:c0")
    print("== models.py: the data shapes components hand to each other ==\n")
    print("Page   (loader -> chunker):")
    print(f"  {page}\n")
    print("Chunk  (chunker -> embedder, index, retriever):")
    print(f"  {chunk}")


if __name__ == "__main__":
    _demo()
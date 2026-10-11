"""Load a PDF and extract text plus metadata."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Page:
    """The text of one PDF page, plus where it came from."""

    text: str
    source: str       # file name, e.g. "book.pdf"
    page_number: int  # 1-based, like in a PDF viewer


def extract_metadata(reader: PdfReader, path: Path) -> dict:
    """Document-level metadata: who/what/how big."""
    info = reader.metadata  # can be None if the PDF has none
    return {
        "file_name": path.name,
        "num_pages": len(reader.pages),
        "title": info.title if info else None,
        "author": info.author if info else None,
    }


def load_pdf(path: str | Path) -> tuple[dict, list[Page]]:
    """Return (document metadata, list of pages)."""
    path = Path(path)
    reader = PdfReader(path)
    metadata = extract_metadata(reader, path)

    pages = []
    for index, pdf_page in enumerate(reader.pages, start=1):
        text = (pdf_page.extract_text() or "").strip()
        if text:  # skip empty pages (images, blank pages)
            pages.append(Page(text=text, source=path.name, page_number=index))

    return metadata, pages
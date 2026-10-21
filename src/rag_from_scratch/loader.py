"""
loader.py: Stage 1. Read a PDF and extract text plus metadata.

CONTRACT:
  INPUT : path to a text-based PDF
  OUTPUT: (metadata dict, list[Page])
          Page(text, source, page_number); pages with no text are skipped

WHY IT IS SEPARATE:
  It is the only file that knows about PDFs. To read another format, or use
  another PDF library (PyMuPDF, ...), only this file changes.

RUN IT ON ITS OWN:
  uv run python -m rag_from_scratch.loader                 # tiny built-in sample PDF
  uv run python -m rag_from_scratch.loader data/book.pdf   # your own PDF
"""

import logging
import sys
import tempfile
from pathlib import Path

from pypdf import PdfReader

from rag_from_scratch.models import Page

logger = logging.getLogger(__name__)


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
        else:
            logger.debug("page %d of %s has no extractable text, skipped", index, path.name)

    logger.info("loaded %s: %d pages, %d with text", path.name, metadata["num_pages"], len(pages))
    return metadata, pages


# ---------------------------------------------------------------------------
# Demo helpers (only used when you run this file on its own, and in tests)
# ---------------------------------------------------------------------------

def _sample_pdf_bytes(page_texts: list[str]) -> bytes:
    """Build a tiny valid PDF with one line of text per page (so the demo needs no file)."""
    count = len(page_texts)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(count))
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(page_texts):
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {5 + 2 * i} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")

    data = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref_at = len(data)
    data += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        data += f"{offset:010d} 00000 n \n".encode()
    data += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return data


def _show(path: Path) -> None:
    print("== loader.py demo ==")
    print(f"INPUT : {path}")
    metadata, pages = load_pdf(path)
    print("OUTPUT:")
    print(f"  metadata : {metadata}")
    print(f"  pages    : {len(pages)} with text")
    for page in pages[:2]:
        preview = page.text[:100].replace("\n", " ")
        print(f"  Page(source={page.source!r}, page_number={page.page_number}, text={preview!r})")


def _demo(path: str | None = None) -> None:
    if path:
        _show(Path(path))
        return
    with tempfile.TemporaryDirectory() as folder:
        sample = Path(folder) / "sample.pdf"
        sample.write_bytes(_sample_pdf_bytes([
            "Page one: a functional movement is a pattern used in daily life.",
            "Page two: a score of 3 means the movement is performed well.",
        ]))
        _show(sample)


if __name__ == "__main__":
    _demo(sys.argv[1] if len(sys.argv) > 1 else None)
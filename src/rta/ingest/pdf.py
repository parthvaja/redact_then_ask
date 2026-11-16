"""PDF ingestion via PyMuPDF.

Extracts text from every page, records per-page character ranges, and
returns a normalised ``Document``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pymupdf

from rta.ingest.normalize import normalize_text
from rta.schema import Document

logger = logging.getLogger(__name__)


def load_pdf(path: Path) -> Document:
    """Load a PDF file and return a ``Document``.

    Pages are joined with ``\\n\\n``.  If a page contains no extractable
    text, a warning is logged (the page is still represented in the
    ``pages`` list as a zero-length range so page numbering stays
    consistent).
    """
    path = Path(path)
    doc = pymupdf.open(str(path))

    page_texts: list[str] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        raw = page.get_text()
        if not raw.strip():
            logger.warning("Page %d of %s has no text layer", page_num, path.name)
        page_texts.append(raw)

    doc.close()

    joined = "\n\n".join(page_texts)
    text = normalize_text(joined)

    # Build page ranges by re-splitting on the same separator.
    pages: list[tuple[int, int]] = []
    offset = 0
    for i, pt in enumerate(page_texts):
        # Each page's normalised length can differ from the raw length,
        # so we normalise each page individually to find its length.
        norm_page = normalize_text(pt)
        start = offset
        end = offset + len(norm_page)
        pages.append((start, end))
        offset = end
        # Account for the "\n\n" separator between pages (except after the last).
        if i < len(page_texts) - 1:
            offset += 2  # len("\n\n")

    doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
    return Document(
        doc_id=doc_id,
        text=text,
        source_path=str(path),
        pages=pages,
    )

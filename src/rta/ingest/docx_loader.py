"""DOCX ingestion via python-docx.

Reads paragraphs and table cells in document order and returns a
normalised ``Document``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from docx import Document as DocxDocument

from rta.ingest.normalize import normalize_text
from rta.schema import Document

logger = logging.getLogger(__name__)


def load_docx(path: Path) -> Document:
    """Load a DOCX file and return a ``Document``.

    Reading order: body paragraphs first, then for each table every row's
    cells are collected.  Paragraphs are joined with ``\\n``; a ``\\n\\n``
    separator is inserted before each table section.
    """
    path = Path(path)
    docx = DocxDocument(str(path))

    parts: list[str] = []

    # Body paragraphs.
    para_texts = [p.text for p in docx.paragraphs]
    if para_texts:
        parts.append("\n".join(para_texts))

    # Tables: iterate rows → cells.
    for table in docx.tables:
        cell_texts: list[str] = []
        for row in table.rows:
            for cell in row.cells:
                cell_texts.append(cell.text)
        if cell_texts:
            parts.append("\n".join(cell_texts))

    joined = "\n\n".join(parts)
    text = normalize_text(joined)

    doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
    return Document(
        doc_id=doc_id,
        text=text,
        source_path=str(path),
    )

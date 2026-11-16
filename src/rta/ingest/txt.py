"""Plain-text ingestion with automatic encoding detection.

Uses ``charset_normalizer`` to detect encoding so the caller does not
need to know whether a file is UTF-8, Latin-1, etc.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from charset_normalizer import from_path

from rta.ingest.normalize import normalize_text
from rta.schema import Document

logger = logging.getLogger(__name__)


def load_txt(path: Path) -> Document:
    """Load a plain-text file with automatic encoding detection.

    Returns a normalised ``Document`` with no page ranges.
    """
    path = Path(path)

    result = from_path(path)
    best = result.best()
    if best is None:
        raise ValueError(f"Could not detect encoding for {path}")

    raw = str(best)
    text = normalize_text(raw)

    doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
    return Document(
        doc_id=doc_id,
        text=text,
        source_path=str(path),
    )

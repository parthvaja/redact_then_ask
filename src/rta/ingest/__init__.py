"""Document ingestion — PDF, DOCX, TXT → ``Document``.

The public entry point is :func:`load`, which dispatches by file
extension or wraps a plain string directly.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from rta.ingest.normalize import normalize_text
from rta.schema import Document

logger = logging.getLogger(__name__)

# Extension → loader mapping.  Loaders are imported lazily inside
# ``load()`` so that heavy packages (pymupdf, python-docx) are only
# loaded when actually needed.
_EXT_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
}


def load(path_or_text: Path | str) -> Document:
    """Load a document from *path_or_text*.

    * If *path_or_text* is a ``Path`` (or a string that points to an
      existing file), the appropriate format-specific loader is called.
    * If it is a plain ``str`` that is **not** an existing file path, it
      is treated as raw text, normalised, and wrapped in a ``Document``
      (handy for testing and REPL work).

    Raises:
        ValueError: If the file extension is not supported.
    """
    # If it is already a Path, use it directly.
    if isinstance(path_or_text, Path):
        return _load_file(path_or_text)

    # It is a str — could be a file path or raw text.
    candidate = Path(path_or_text)
    if candidate.exists() and candidate.is_file():
        return _load_file(candidate)

    # Treat as raw text.
    text = normalize_text(path_or_text)
    doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
    return Document(
        doc_id=doc_id,
        text=text,
        source_path=None,
    )


def _load_file(path: Path) -> Document:
    """Dispatch to the right loader based on *path*'s extension."""
    ext = path.suffix.lower()
    if ext not in _EXT_MAP:
        raise ValueError(
            f"Unsupported file extension {ext!r}. Supported: {', '.join(sorted(_EXT_MAP))}"
        )

    loader_name = _EXT_MAP[ext]

    if loader_name == "pdf":
        from rta.ingest.pdf import load_pdf

        return load_pdf(path)
    elif loader_name == "docx":
        from rta.ingest.docx_loader import load_docx

        return load_docx(path)
    else:
        from rta.ingest.txt import load_txt

        return load_txt(path)

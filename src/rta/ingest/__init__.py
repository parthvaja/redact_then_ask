"""Document ingestion — PDF, DOCX, TXT → ``Document``.

Implemented in Phase 1.  This stub exports the public interface so other
modules can import it without errors during Phase 0.
"""

from __future__ import annotations

from pathlib import Path

from rta.schema import Document


def load(path_or_text: Path | str) -> Document:
    """Load a document from *path_or_text*.

    Args:
        path_or_text: A file path (PDF/DOCX/TXT) or a plain string.

    Returns:
        A ``Document`` with normalised text and character-offset page ranges.

    Raises:
        NotImplementedError: Until Phase 1 is complete.
    """
    raise NotImplementedError("rta.ingest.load — implemented in Phase 1")

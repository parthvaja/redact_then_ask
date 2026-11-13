"""Regex-based PII detection.

Implemented in Phase 4.
"""

from __future__ import annotations

from rta.schema import Document, Span


def run(doc: Document, doc_type: str | None = None) -> list[Span]:
    """Run all regex detectors on *doc*.

    Args:
        doc:      An ingested ``Document``.
        doc_type: Predicted document type, used to select rule packs.

    Returns:
        Unresolved ``Span`` list with ``detector`` starting with ``"regex:"``.

    Raises:
        NotImplementedError: Until Phase 4 is complete.
    """
    raise NotImplementedError("rta.detect.regex.run — implemented in Phase 4")

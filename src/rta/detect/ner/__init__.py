"""NER-based PII detection.

Implemented in Phase 5.
"""

from __future__ import annotations

from rta.schema import Document, Span


def run(doc: Document) -> list[Span]:
    """Run the NER detector on *doc*.

    Args:
        doc: An ingested ``Document``.

    Returns:
        Unresolved ``Span`` list with ``detector`` starting with ``"ner:"``.

    Raises:
        NotImplementedError: Until Phase 5 is complete.
    """
    raise NotImplementedError("rta.detect.ner.run — implemented in Phase 5")

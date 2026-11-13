"""Redaction and pseudonymization.

Implemented in Phase 8.
"""

from __future__ import annotations

from rta.schema import Document, RedactionResult, Span


def apply(
    doc: Document,
    spans: list[Span],
    mode: str | None = None,
) -> RedactionResult:
    """Apply redaction to *doc* using the resolved *spans*.

    Args:
        doc:   The source document.
        spans: Non-overlapping, resolved spans from ``rta.resolve.merge``.
        mode:  Override for redaction mode.  If ``None``, reads from config.
               One of ``"mask"``, ``"placeholder"``, ``"fake"``.

    Returns:
        A ``RedactionResult`` with the redacted text, placeholder mapping,
        mention clusters and an offset alignment map.

    Raises:
        NotImplementedError: Until Phase 8 is complete.
    """
    raise NotImplementedError("rta.redact.apply — implemented in Phase 8")

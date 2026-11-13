"""Language identification — ``Document`` → ISO language code + confidence.

Implemented in Phase 1.
"""

from __future__ import annotations

from rta.schema import Document


def detect(doc: Document) -> tuple[str, float]:
    """Detect the language of *doc*.

    Args:
        doc: An ingested ``Document``.

    Returns:
        A tuple ``(iso_code, confidence)`` where ``iso_code`` is a
        2-letter ISO 639-1 code and ``confidence`` is in [0, 1].

    Raises:
        NotImplementedError: Until Phase 1 is complete.
    """
    raise NotImplementedError("rta.langid.detect — implemented in Phase 1")

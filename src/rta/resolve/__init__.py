"""Span resolution — merge overlapping and conflicting spans.

Implemented in Phase 7.
"""

from __future__ import annotations

from rta.schema import Span


def merge(spans: list[Span]) -> list[Span]:
    """Merge and resolve *spans* into a non-overlapping final list.

    Applies the resolution rules from §6.7 of ``docs/PROJECT_DOC.md``:
    1. Validated beats unvalidated.
    2. Same range, different labels → label precedence then score.
    3. Partial overlap, same label → take the union (recall first).
    4. Partial overlap, different labels → keep the longer unless shorter is validated.
    5. Drop spans below ``resolve.min_score``.
    6. Trim leading/trailing whitespace and punctuation.

    Args:
        spans: Raw, potentially overlapping spans from all detectors.

    Returns:
        Non-overlapping, resolved ``Span`` list sorted by ``start``.

    Raises:
        NotImplementedError: Until Phase 7 is complete.
    """
    raise NotImplementedError("rta.resolve.merge — implemented in Phase 7")

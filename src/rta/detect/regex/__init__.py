"""Regex-based PII detection (Phase 4).

Loads rule packs from ``configs/rules/``, compiles patterns, validates
matches, and returns unresolved ``Span`` objects.
"""

from __future__ import annotations

import logging

from rta.detect.regex.engine import load_packs, reset_cache, scan
from rta.schema import Document, Span

logger = logging.getLogger(__name__)


def run(doc: Document, doc_type: str | None = None) -> list[Span]:
    """Run all regex detectors on *doc*.

    Args:
        doc:      An ingested ``Document``.
        doc_type: Predicted document type, used to select rule packs.

    Returns:
        Unresolved ``Span`` list with ``detector`` starting with ``"regex:"``.
    """
    rules = load_packs(doc_type=doc_type)
    spans = scan(doc, rules)
    logger.info("Regex detection: %d spans from %d rules", len(spans), len(rules))
    return spans


__all__ = ["load_packs", "reset_cache", "run", "scan"]

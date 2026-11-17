"""Presidio baseline runner (Phase 9).

Wraps Microsoft Presidio's ``AnalyzerEngine`` and maps results through
``rta.taxonomy`` so they are directly comparable to our system's output.
"""

from __future__ import annotations

import logging

from rta.schema import Span
from rta.taxonomy import PRESIDIO_TO_LABEL, normalize_label

logger = logging.getLogger(__name__)


def run_presidio(text: str, language: str = "en") -> list[Span]:
    """Run Presidio on *text* and return taxonomy-mapped spans.

    Returns an empty list if ``presidio-analyzer`` is not installed.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError:
        logger.warning("presidio-analyzer not installed -- skipping baseline")
        return []

    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=text, language=language)

    spans: list[Span] = []
    for r in results:
        try:
            label = normalize_label(r.entity_type, PRESIDIO_TO_LABEL)
        except KeyError:
            logger.debug("Presidio label %r not in taxonomy -- skipping", r.entity_type)
            continue
        if label is None:
            continue

        spans.append(
            Span(
                start=r.start,
                end=r.end,
                label=label,
                text=text[r.start : r.end],
                score=r.score,
                detector=f"presidio:{r.entity_type}",
            )
        )

    return spans

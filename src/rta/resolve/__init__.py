"""Span resolution -- merge overlapping and conflicting spans.

Applies the resolution rules from section 6.7 of ``docs/PROJECT_DOC.md``:

1. A validated span wins over an unvalidated one.
2. Exact same range, different labels: label precedence then score.
3. Partial overlap, same label: take the union (recall first).
4. Partial overlap, different labels: keep the longer unless shorter is validated.
5. Drop spans below ``resolve.min_score``.
6. Trim leading/trailing whitespace and punctuation.
"""

from __future__ import annotations

import logging
import string

from rta.config import get_config
from rta.schema import Span
from rta.taxonomy import label_precedence_rank

logger = logging.getLogger(__name__)

_TRIM_CHARS = string.whitespace + string.punctuation


def merge(spans: list[Span], doc_text: str | None = None) -> list[Span]:
    """Merge and resolve *spans* into a non-overlapping final list.

    Args:
        spans:    Raw, potentially overlapping spans from all detectors.
        doc_text: Original document text, needed for union text and trimming.

    Returns:
        Non-overlapping, resolved ``Span`` list sorted by ``start``.
    """
    if not spans:
        return []

    cfg = get_config()
    min_score = cfg.resolve.min_score

    # Rule 5: drop below min_score
    spans = [s for s in spans if s.score >= min_score]
    if not spans:
        return []

    # Rule 3: union same-label overlapping spans
    spans = _merge_same_label(spans, doc_text)

    # Rules 1, 2, 4: resolve different-label overlaps
    spans = _resolve_cross_label(spans)

    # Rule 6: trim whitespace and punctuation
    if doc_text:
        trimmed = []
        for s in spans:
            t = _trim_span(s, doc_text)
            if t is not None:
                trimmed.append(t)
        spans = trimmed

    spans.sort(key=lambda s: s.start)
    return spans


# ---------------------------------------------------------------------------
# Rule 3: union same-label overlaps
# ---------------------------------------------------------------------------


def _merge_same_label(spans: list[Span], doc_text: str | None) -> list[Span]:
    """Merge overlapping spans with the same label into their union."""
    by_label: dict[str, list[Span]] = {}
    for s in spans:
        by_label.setdefault(s.label, []).append(s)

    result: list[Span] = []
    for label, group in by_label.items():
        group.sort(key=lambda s: s.start)
        merged = [group[0]]
        for s in group[1:]:
            last = merged[-1]
            if s.start < last.end:  # overlap -> union
                new_start = min(last.start, s.start)
                new_end = max(last.end, s.end)
                text = doc_text[new_start:new_end] if doc_text else last.text
                logger.debug(
                    "Rule 3: union same-label %s [%d,%d)+[%d,%d) -> [%d,%d)",
                    label,
                    last.start,
                    last.end,
                    s.start,
                    s.end,
                    new_start,
                    new_end,
                )
                merged[-1] = Span(
                    start=new_start,
                    end=new_end,
                    label=label,
                    text=text,
                    score=max(last.score, s.score),
                    detector=last.detector if last.score >= s.score else s.detector,
                    validated=last.validated or s.validated,
                )
            else:
                merged.append(s)
        result.extend(merged)

    return result


# ---------------------------------------------------------------------------
# Rules 1, 2, 4: cross-label overlap resolution
# ---------------------------------------------------------------------------


def _resolve_cross_label(spans: list[Span]) -> list[Span]:
    """Resolve overlaps between different-label spans."""
    if not spans:
        return []

    # Sort by start, group overlapping spans
    spans.sort(key=lambda s: s.start)
    groups = _build_overlap_groups(spans)

    result: list[Span] = []
    for group in groups:
        if len(group) == 1:
            result.append(group[0])
        else:
            result.extend(_resolve_group(group))

    return result


def _build_overlap_groups(spans: list[Span]) -> list[list[Span]]:
    """Partition sorted spans into groups of mutually overlapping spans."""
    groups: list[list[Span]] = []
    current = [spans[0]]
    max_end = spans[0].end

    for s in spans[1:]:
        if s.start < max_end:
            current.append(s)
            max_end = max(max_end, s.end)
        else:
            groups.append(current)
            current = [s]
            max_end = s.end
    groups.append(current)
    return groups


def _resolve_group(group: list[Span]) -> list[Span]:
    """Resolve an overlap group using greedy priority selection.

    Priority ordering:
    1. Validated spans first.
    2. Longer spans first (rule 4).
    3. Higher label precedence first (rule 2 tie-breaker for same length).
    4. Higher score first.
    """
    group.sort(
        key=lambda s: (
            not s.validated,  # True first -> False sorts lower
            -(s.end - s.start),  # longer first
            label_precedence_rank(s.label),  # lower rank = higher priority
            -s.score,  # higher score first
        )
    )

    selected: list[Span] = []
    for span in group:
        if not any(span.overlaps(s) for s in selected):
            selected.append(span)
            logger.debug(
                "Rule 1/2/4: selected %s [%d,%d) score=%.2f val=%s",
                span.label,
                span.start,
                span.end,
                span.score,
                span.validated,
            )
        else:
            logger.debug(
                "Rule 1/2/4: dropped  %s [%d,%d) score=%.2f val=%s (overlap)",
                span.label,
                span.start,
                span.end,
                span.score,
                span.validated,
            )

    return selected


# ---------------------------------------------------------------------------
# Rule 6: trim whitespace and punctuation
# ---------------------------------------------------------------------------


def _trim_span(span: Span, doc_text: str) -> Span | None:
    """Trim leading/trailing whitespace and punctuation from *span*."""
    start = max(span.start, 0)
    end = min(span.end, len(doc_text))
    while start < end and doc_text[start] in _TRIM_CHARS:
        start += 1
    while end > start and doc_text[end - 1] in _TRIM_CHARS:
        end -= 1

    if start >= end:
        logger.debug("Rule 6: trimmed %s [%d,%d) to nothing", span.label, span.start, span.end)
        return None

    if start == span.start and end == span.end:
        return span

    return Span(
        start=start,
        end=end,
        label=span.label,
        text=doc_text[start:end],
        score=span.score,
        detector=span.detector,
        validated=span.validated,
    )

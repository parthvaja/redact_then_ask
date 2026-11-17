"""Redaction and pseudonymization (Phase 8).

Three modes: ``mask`` (black bars), ``placeholder`` (``[LABEL_N]``),
and ``fake`` (Faker surrogates seeded by cluster for consistency).
"""

from __future__ import annotations

import logging

from rta.config import get_config
from rta.redact.cluster import cluster_spans
from rta.redact.surrogates import generate_surrogate
from rta.schema import Document, RedactionResult, Span

logger = logging.getLogger(__name__)


def apply(
    doc: Document,
    spans: list[Span],
    mode: str | None = None,
) -> RedactionResult:
    """Apply redaction to *doc* using the resolved *spans*.

    Args:
        doc:   The source document.
        spans: Non-overlapping, resolved spans from ``rta.resolve.merge``.
        mode:  Override for redaction mode (``"mask"``, ``"placeholder"``,
               ``"fake"``).  Reads from config if ``None``.

    Returns:
        A ``RedactionResult`` with the redacted text, placeholder mapping,
        mention clusters and an offset alignment map.
    """
    cfg = get_config()
    mode = mode or cfg.redact.mode

    if not spans:
        return RedactionResult(text=doc.text)

    # Cluster mentions
    clusters = cluster_spans(spans)

    # Build replacement lookup: span id -> replacement string
    span_replacement: dict[int, str] = {}
    mapping: dict[str, str] = {}
    label_counters: dict[str, int] = {}

    for cluster_idx, cluster in enumerate(clusters):
        label = cluster[0].label

        if mode == "mask":
            for span in cluster:
                span_replacement[id(span)] = "\u2588" * span.length
        elif mode == "placeholder":
            count = label_counters.get(label, 0) + 1
            label_counters[label] = count
            placeholder = f"[{label}_{count}]"
            canonical = max(cluster, key=lambda s: s.length)
            mapping[placeholder] = canonical.text
            for span in cluster:
                span_replacement[id(span)] = placeholder
        elif mode == "fake":
            surrogate = generate_surrogate(label, cluster_idx)
            canonical = max(cluster, key=lambda s: s.length)
            mapping[surrogate] = canonical.text
            for span in cluster:
                span_replacement[id(span)] = surrogate
        else:
            raise ValueError(f"Unknown redaction mode: {mode!r}")

    # Build redacted text and offset map
    all_spans = sorted(spans, key=lambda s: s.start)
    parts: list[str] = []
    offset_map: list[tuple[int, int, int, int]] = []
    pos = 0
    red_pos = 0

    for span in all_spans:
        # Text before this span
        if span.start > pos:
            gap = doc.text[pos : span.start]
            parts.append(gap)
            red_pos += len(gap)

        # Replacement
        replacement = span_replacement.get(id(span), f"[{span.label}]")
        red_start = red_pos
        parts.append(replacement)
        red_pos += len(replacement)
        offset_map.append((span.start, span.end, red_start, red_pos))
        pos = span.end

    # Trailing text
    if pos < len(doc.text):
        parts.append(doc.text[pos:])

    redacted_text = "".join(parts)
    logger.info(
        "Redaction (%s): %d spans, %d clusters, %d chars -> %d chars",
        mode,
        len(all_spans),
        len(clusters),
        len(doc.text),
        len(redacted_text),
    )

    return RedactionResult(
        text=redacted_text,
        mapping=mapping,
        clusters=clusters,
        offset_map=offset_map,
    )

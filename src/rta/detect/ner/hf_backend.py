"""Hugging Face token-classification NER backend.

Uses ``transformers.pipeline("token-classification", ...)`` with
``aggregation_strategy="simple"`` to group BIO tokens.  For long
documents, splits by character windows and maps offsets back.
"""

from __future__ import annotations

import logging

from rta.config import get_config
from rta.schema import Document, Span
from rta.taxonomy import HF_NER_TO_LABEL, normalize_label

logger = logging.getLogger(__name__)

_pipe_cache = None


def _get_pipeline():
    """Load the HF NER pipeline (cached)."""
    global _pipe_cache
    if _pipe_cache is not None:
        return _pipe_cache

    from transformers import pipeline

    cfg = get_config()
    model_name = cfg.detect.ner.hf_model
    logger.info("Loading HF NER pipeline: %s", model_name)
    _pipe_cache = pipeline(
        "token-classification",
        model=model_name,
        aggregation_strategy="simple",
    )
    return _pipe_cache


def reset_pipeline_cache() -> None:
    """Clear the cached pipeline (for tests)."""
    global _pipe_cache
    _pipe_cache = None


def _map_entities(
    entities: list[dict],
    offset: int = 0,
    source_text: str = "",
) -> list[Span]:
    """Map HF entity dicts to taxonomy-mapped Spans."""
    cfg = get_config()
    min_score = cfg.detect.ner.min_score
    spans: list[Span] = []

    for ent in entities:
        raw_label = ent.get("entity_group", ent.get("entity", ""))
        score = float(ent.get("score", 0.0))

        if score < min_score:
            continue

        try:
            label = normalize_label(raw_label, HF_NER_TO_LABEL)
        except KeyError:
            logger.debug("Unknown HF entity type: %s", raw_label)
            continue

        if label is None:
            continue

        start = ent["start"] + offset
        end = ent["end"] + offset
        text = source_text[start : start + (end - start)] if source_text else ent.get("word", "")

        spans.append(
            Span(
                start=start,
                end=end,
                label=label,
                text=text,
                score=score,
                detector="ner:hf",
            )
        )

    return spans


def _dedupe_spans(spans: list[Span]) -> list[Span]:
    """Remove duplicate spans (same start, end, label)."""
    seen: set[tuple[int, int, str]] = set()
    unique: list[Span] = []
    for s in spans:
        key = (s.start, s.end, s.label)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def detect(doc: Document) -> list[Span]:
    """Run HF NER on *doc* with chunking for long documents."""
    pipe = _get_pipeline()
    cfg = get_config()
    text = doc.text

    # HF models have a max token length; approximate by characters.
    # 400 tokens ≈ ~2000 chars for English.  Use a char-based window.
    max_chars = cfg.detect.ner.window_tokens * 5  # rough char estimate
    stride_chars = cfg.detect.ner.stride * 5

    if len(text) <= max_chars:
        entities = pipe(text)
        return _map_entities(entities, offset=0, source_text=text)

    # Sliding window
    logger.debug(
        "Document has %d chars, using sliding window (max_chars=%d, stride=%d)",
        len(text),
        max_chars,
        stride_chars,
    )

    all_spans: list[Span] = []
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        chunk = text[pos:end]
        entities = pipe(chunk)
        chunk_spans = _map_entities(entities, offset=pos, source_text=text)
        all_spans.extend(chunk_spans)

        if end >= len(text):
            break
        pos += stride_chars

    return _dedupe_spans(all_spans)

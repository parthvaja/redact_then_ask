"""spaCy NER backend for PII detection.

Uses ``en_core_web_trf`` (GPU/fast CPU) or ``en_core_web_lg`` (CPU
fallback).  For documents longer than ``window_tokens``, applies a
sliding window with overlap and deduplicates resulting spans.
"""

from __future__ import annotations

import logging

import spacy
from spacy.tokens import Doc as SpacyDoc

from rta.config import get_config
from rta.schema import Document, Span
from rta.taxonomy import SPACY_TO_LABEL, normalize_label

logger = logging.getLogger(__name__)

_nlp_cache: spacy.language.Language | None = None


def _get_nlp() -> spacy.language.Language:
    """Load the spaCy model (cached after first call)."""
    global _nlp_cache
    if _nlp_cache is not None:
        return _nlp_cache

    cfg = get_config()
    model_name = cfg.detect.ner.spacy_model

    try:
        _nlp_cache = spacy.load(model_name)
        logger.info("Loaded spaCy model: %s", model_name)
    except OSError:
        fallback = cfg.detect.ner.spacy_model_cpu
        logger.warning(
            "spaCy model %s not found, falling back to %s",
            model_name,
            fallback,
        )
        _nlp_cache = spacy.load(fallback)
        logger.info("Loaded spaCy fallback model: %s", fallback)

    return _nlp_cache


def reset_nlp_cache() -> None:
    """Clear the cached NLP model (for tests)."""
    global _nlp_cache
    _nlp_cache = None


def _entities_from_doc(spacy_doc: SpacyDoc, offset: int = 0) -> list[Span]:
    """Extract taxonomy-mapped spans from a spaCy Doc object."""
    cfg = get_config()
    min_score = cfg.detect.ner.min_score
    spans: list[Span] = []

    for ent in spacy_doc.ents:
        try:
            label = normalize_label(ent.label_, SPACY_TO_LABEL)
        except KeyError:
            logger.debug("Unknown spaCy entity type: %s", ent.label_)
            continue

        if label is None:
            continue

        start = ent.start_char + offset
        end = ent.end_char + offset
        text = ent.text

        # spaCy doesn't provide entity-level confidence directly;
        # use a default score that can be overridden by title_cue boosts.
        score = 0.85

        if score < min_score:
            continue

        spans.append(
            Span(
                start=start,
                end=end,
                label=label,
                text=text,
                score=score,
                detector="ner:spacy",
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
    """Run spaCy NER on *doc* with sliding window for long documents."""
    nlp = _get_nlp()
    cfg = get_config()
    text = doc.text

    window_tokens = cfg.detect.ner.window_tokens
    stride = cfg.detect.ner.stride

    # For short documents, process in one go
    spacy_doc = nlp(text)
    n_tokens = len(spacy_doc)

    if n_tokens <= window_tokens:
        return _entities_from_doc(spacy_doc)

    # Sliding window for long documents
    logger.debug(
        "Document has %d tokens, using sliding window (window=%d, stride=%d)",
        n_tokens,
        window_tokens,
        stride,
    )

    all_spans: list[Span] = []
    token_starts = [token.idx for token in spacy_doc]
    token_ends = [token.idx + len(token.text) for token in spacy_doc]

    pos = 0
    while pos < n_tokens:
        end_tok = min(pos + window_tokens, n_tokens)
        char_start = token_starts[pos]
        char_end = token_ends[end_tok - 1]

        window_text = text[char_start:char_end]
        window_doc = nlp(window_text)
        window_spans = _entities_from_doc(window_doc, offset=char_start)
        all_spans.extend(window_spans)

        if end_tok >= n_tokens:
            break
        pos += stride

    return _dedupe_spans(all_spans)

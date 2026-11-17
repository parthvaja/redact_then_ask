"""Routing and rule-pack selection (Phase 6).

Decides which regex rule packs to apply based on document classification
confidence, and boosts NER PERSON scores when title cues appear nearby.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from rta.config import get_config
from rta.schema import Span

logger = logging.getLogger(__name__)

_cue_cache: dict[str, list[str]] = {}


def for_type(
    doc_type: str | None,
    type_probs: dict[str, float] | None = None,
) -> list[str]:
    """Decide which doc-type packs to apply.

    Always returns a list of doc_type strings whose packs should be loaded
    (in addition to ``common``).  If the top prediction's probability is
    below ``routing.min_conf``, the top-k types are returned (recall-first).
    """
    if doc_type is None:
        return []

    if type_probs is None:
        return [doc_type]

    cfg = get_config()
    min_conf = cfg.routing.min_conf
    fallback_top_k = cfg.routing.fallback_top_k

    top_prob = type_probs.get(doc_type, 0.0)

    if top_prob >= min_conf:
        logger.debug(
            "Routing: doc_type=%s conf=%.2f >= %.2f -> single pack",
            doc_type,
            top_prob,
            min_conf,
        )
        return [doc_type]

    sorted_types = sorted(type_probs.items(), key=lambda x: x[1], reverse=True)
    top_k = [t for t, _ in sorted_types[:fallback_top_k]]
    logger.debug(
        "Routing: doc_type=%s conf=%.2f < %.2f -> fallback top-%d: %s",
        doc_type,
        top_prob,
        min_conf,
        fallback_top_k,
        top_k,
    )
    return top_k


def get_title_cues(doc_types: list[str]) -> list[str]:
    """Load ``title_cues`` from the YAML rule packs for *doc_types*."""
    cfg = get_config()
    rule_dir = Path(cfg.routing.rule_dir)
    cues: list[str] = []
    seen: set[str] = set()

    for dt in doc_types:
        if dt in _cue_cache:
            for c in _cue_cache[dt]:
                if c not in seen:
                    cues.append(c)
                    seen.add(c)
            continue

        path = rule_dir / f"{dt}.yaml"
        if not path.exists():
            _cue_cache[dt] = []
            continue

        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        dt_cues = data.get("title_cues", []) or []
        _cue_cache[dt] = dt_cues
        for c in dt_cues:
            if c not in seen:
                cues.append(c)
                seen.add(c)

    return cues


def boost_ner_scores(
    spans: list[Span],
    text: str,
    title_cues: list[str],
    boost: float = 0.1,
    proximity: int = 100,
) -> list[Span]:
    """Boost NER PERSON span scores when title cues appear nearby.

    ``Span`` is frozen, so boosted spans are recreated.
    """
    if not title_cues or not spans:
        return list(spans)

    text_lower = text.lower()
    cue_positions: list[int] = []
    for cue in title_cues:
        cue_lower = cue.lower()
        start = 0
        while True:
            pos = text_lower.find(cue_lower, start)
            if pos == -1:
                break
            cue_positions.append(pos)
            start = pos + 1

    if not cue_positions:
        return list(spans)

    result: list[Span] = []
    for span in spans:
        if span.label == "PERSON" and span.detector.startswith("ner:"):
            near_cue = any(
                abs(span.start - cp) <= proximity or abs(span.end - cp) <= proximity
                for cp in cue_positions
            )
            if near_cue:
                new_score = min(span.score + boost, 1.0)
                result.append(
                    Span(
                        start=span.start,
                        end=span.end,
                        label=span.label,
                        text=span.text,
                        score=new_score,
                        detector=span.detector,
                        validated=span.validated,
                    )
                )
                logger.debug(
                    "Boosted PERSON span %r %.2f -> %.2f (title cue nearby)",
                    span.text,
                    span.score,
                    new_score,
                )
                continue
        result.append(span)

    return result


def reset_cue_cache() -> None:
    """Clear the cue cache (for tests)."""
    _cue_cache.clear()

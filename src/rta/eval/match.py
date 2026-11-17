"""Span matching strategies for evaluation (Phase 9).

Three modes: strict (exact start/end/label), overlap (any char overlap +
same label), and token-level (per-character, label-agnostic).
"""

from __future__ import annotations

from rta.schema import Span


def match_strict(
    pred: list[Span],
    gold: list[Span],
) -> tuple[list[Span], list[Span], list[Span]]:
    """Strict matching: same start, end and label.

    Returns ``(tp, fp, fn)`` where each element is a list of spans.
    """
    gold_set = {(s.start, s.end, s.label) for s in gold}
    pred_set = {(s.start, s.end, s.label) for s in pred}

    tp = [s for s in pred if (s.start, s.end, s.label) in gold_set]
    fp = [s for s in pred if (s.start, s.end, s.label) not in gold_set]
    fn = [s for s in gold if (s.start, s.end, s.label) not in pred_set]

    return tp, fp, fn


def match_overlap(
    pred: list[Span],
    gold: list[Span],
) -> tuple[list[Span], list[Span], list[Span]]:
    """Overlap matching: any char overlap + same label.

    Each gold span is matched to at most one prediction.
    Returns ``(tp, fp, fn)``.
    """
    tp: list[Span] = []
    fp: list[Span] = []
    matched_gold: set[int] = set()

    for p in pred:
        found = False
        for i, g in enumerate(gold):
            if i in matched_gold:
                continue
            if p.label == g.label and p.overlaps(g):
                tp.append(p)
                matched_gold.add(i)
                found = True
                break
        if not found:
            fp.append(p)

    fn = [g for i, g in enumerate(gold) if i not in matched_gold]
    return tp, fp, fn


def match_token(
    pred: list[Span],
    gold: list[Span],
) -> tuple[int, int, int]:
    """Token-level (character-level) matching, label-agnostic.

    Returns ``(tp_chars, fp_chars, fn_chars)`` as integer counts.
    """
    pred_chars: set[int] = set()
    for s in pred:
        pred_chars.update(range(s.start, s.end))

    gold_chars: set[int] = set()
    for s in gold:
        gold_chars.update(range(s.start, s.end))

    tp = len(pred_chars & gold_chars)
    fp = len(pred_chars - gold_chars)
    fn = len(gold_chars - pred_chars)
    return tp, fp, fn

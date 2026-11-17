"""Evaluation metrics: P / R / F1 / F2 per label, micro/macro, bootstrap CI."""

from __future__ import annotations

from typing import Any

import numpy as np

from rta.schema import Span


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall and F1 from counts."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    """Compute F-beta score (default F2, recall-weighted)."""
    if precision + recall == 0:
        return 0.0
    b2 = beta**2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def per_label_metrics(
    tp_spans: list[Span],
    fp_spans: list[Span],
    fn_spans: list[Span],
) -> dict[str, dict[str, Any]]:
    """Compute per-label and aggregate metrics from matched span lists.

    Returns a dict keyed by label (plus ``"micro"`` and ``"macro"``).
    """
    labels: set[str] = set()
    for s in tp_spans:
        labels.add(s.label)
    for s in fp_spans:
        labels.add(s.label)
    for s in fn_spans:
        labels.add(s.label)

    results: dict[str, dict[str, Any]] = {}
    for label in sorted(labels):
        tp = sum(1 for s in tp_spans if s.label == label)
        fp = sum(1 for s in fp_spans if s.label == label)
        fn = sum(1 for s in fn_spans if s.label == label)
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        results[label] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "f2": f_beta(p, r, 2.0),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # Micro average
    total_tp = len(tp_spans)
    total_fp = len(fp_spans)
    total_fn = len(fn_spans)
    p, r, f1 = precision_recall_f1(total_tp, total_fp, total_fn)
    results["micro"] = {
        "precision": p,
        "recall": r,
        "f1": f1,
        "f2": f_beta(p, r, 2.0),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }

    # Macro average
    per_label = [v for k, v in results.items() if k not in ("micro", "macro")]
    if per_label:
        results["macro"] = {
            "precision": float(np.mean([v["precision"] for v in per_label])),
            "recall": float(np.mean([v["recall"] for v in per_label])),
            "f1": float(np.mean([v["f1"] for v in per_label])),
            "f2": float(np.mean([v["f2"] for v in per_label])),
        }

    return results


def per_label_metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
) -> dict[str, dict[str, Any]]:
    """Build metrics dict from raw counts (used by token-level matching)."""
    p, r, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "ALL": {
            "precision": p,
            "recall": r,
            "f1": f1,
            "f2": f_beta(p, r, 2.0),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
    }


def leak_rate(
    gold_spans: list[Span],
    redacted_text: str,
    original_text: str,
) -> float:
    """Fraction of gold PII characters remaining in the redacted text."""
    total = sum(s.end - s.start for s in gold_spans)
    if total == 0:
        return 0.0

    leaked = 0
    for s in gold_spans:
        pii_text = original_text[s.start : s.end]
        if pii_text in redacted_text:
            leaked += len(pii_text)

    return leaked / total


def bootstrap_ci(
    scores: list[float],
    n_resamples: int = 1000,
    seed: int = 13,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a list of per-document scores."""
    if not scores:
        return (0.0, 0.0)

    rng = np.random.RandomState(seed)
    n = len(scores)
    boot_means = []
    for _ in range(n_resamples):
        sample = rng.choice(scores, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    return lower, upper

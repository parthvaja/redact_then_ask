"""Evaluation harness -- precision, recall and F1 per PII type (Phase 9).

Public API:
    ``score()``           -- score predictions against gold (used by experiments).
    ``run_evaluation()``  -- full CLI-driven evaluation pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rta.config import get_config
from rta.eval.match import match_overlap, match_strict, match_token
from rta.eval.metrics import bootstrap_ci as bootstrap_ci
from rta.eval.metrics import leak_rate as leak_rate
from rta.eval.metrics import per_label_metrics, per_label_metrics_from_counts
from rta.eval.report import generate_comparison_table, generate_report
from rta.schema import GoldDoc, Span

logger = logging.getLogger(__name__)


def score(
    pred_spans: list[Span],
    gold_spans: list[Span],
    mode: str = "overlap",
) -> dict[str, dict[str, Any]]:
    """Score *pred_spans* against *gold_spans*.

    Args:
        pred_spans: Predicted spans.
        gold_spans: Gold-standard spans.
        mode:       ``"strict"``, ``"overlap"`` or ``"token"``.

    Returns:
        Dict keyed by label (plus ``"micro"``/``"macro"``) with P/R/F1/F2.
    """
    if mode == "strict":
        tp, fp, fn = match_strict(pred_spans, gold_spans)
        return per_label_metrics(tp, fp, fn)
    if mode == "overlap":
        tp, fp, fn = match_overlap(pred_spans, gold_spans)
        return per_label_metrics(tp, fp, fn)
    if mode == "token":
        tp_c, fp_c, fn_c = match_token(pred_spans, gold_spans)
        return per_label_metrics_from_counts(tp_c, fp_c, fn_c)
    raise ValueError(f"Unknown matching mode: {mode!r}")


def run_evaluation(
    dataset: str,
    split: str = "test",
    compare: str | None = None,
    system_name: str = "ours",
) -> None:
    """Run detection evaluation and write reports.

    Loads gold documents, runs our pipeline (and optionally Presidio),
    scores, and writes tables to ``reports/tables/``.
    """
    cfg = get_config()
    reports_dir = Path(cfg.paths.reports) / "tables"
    matching_mode = cfg.eval.matching

    gold_docs = _load_gold(dataset, split)
    if not gold_docs:
        logger.error("No gold docs found for %s/%s", dataset, split)
        return

    # Score our system
    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    our_metrics = _evaluate_system(gold_docs, matching_mode)
    all_results[system_name] = our_metrics
    generate_report(our_metrics, system_name, dataset, reports_dir)

    # Optionally run Presidio baseline
    if compare == "presidio":
        presidio_metrics = _evaluate_presidio(gold_docs, matching_mode)
        all_results["presidio"] = presidio_metrics
        generate_report(presidio_metrics, "presidio", dataset, reports_dir)

    # Combined comparison
    if len(all_results) > 1:
        generate_comparison_table(all_results, dataset, reports_dir)

    logger.info("Evaluation complete for %s/%s", dataset, split)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_gold(dataset: str, split: str) -> list[GoldDoc]:
    """Load gold documents for the given dataset and split."""
    from rta.schema import read_gold_docs_jsonl

    cfg = get_config()
    data_dir = Path(cfg.paths.data) / "gold"

    path = data_dir / f"{dataset}_{split}.jsonl"
    if not path.exists():
        # Try without split suffix
        path = data_dir / f"{dataset}.jsonl"
    if not path.exists():
        logger.warning("Gold file not found: %s", path)
        return []

    return read_gold_docs_jsonl(path)


def _evaluate_system(
    gold_docs: list[GoldDoc],
    mode: str,
) -> dict[str, dict[str, Any]]:
    """Run our pipeline on each gold doc and aggregate metrics."""
    from rta.detect.ner import run as ner_run
    from rta.detect.regex import run as regex_run
    from rta.ingest import load
    from rta.resolve import merge

    all_tp: list[Span] = []
    all_fp: list[Span] = []
    all_fn: list[Span] = []

    for gdoc in gold_docs:
        doc = load(gdoc.text)
        regex_spans = regex_run(doc, doc_type=gdoc.doc_type)
        try:
            ner_spans = ner_run(doc)
        except Exception:
            ner_spans = []
        pred = merge(regex_spans + ner_spans, doc_text=doc.text)

        if mode == "token":
            # Accumulate separately for token mode
            continue

        tp, fp, fn = (
            match_strict(pred, gdoc.spans) if mode == "strict" else match_overlap(pred, gdoc.spans)
        )
        all_tp.extend(tp)
        all_fp.extend(fp)
        all_fn.extend(fn)

    return per_label_metrics(all_tp, all_fp, all_fn)


def _evaluate_presidio(
    gold_docs: list[GoldDoc],
    mode: str,
) -> dict[str, dict[str, Any]]:
    """Run Presidio on each gold doc and aggregate metrics."""
    from rta.eval.presidio_baseline import run_presidio

    all_tp: list[Span] = []
    all_fp: list[Span] = []
    all_fn: list[Span] = []

    for gdoc in gold_docs:
        pred = run_presidio(gdoc.text)

        tp, fp, fn = (
            match_strict(pred, gdoc.spans) if mode == "strict" else match_overlap(pred, gdoc.spans)
        )
        all_tp.extend(tp)
        all_fp.extend(fp)
        all_fn.extend(fn)

    return per_label_metrics(all_tp, all_fp, all_fn)

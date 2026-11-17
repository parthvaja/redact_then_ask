"""E1: Privacy vs. utility experiment.

Measures QA accuracy at four redaction levels (L0-L3) and compares
mask vs placeholder at L2.  Plots the privacy-utility trade-off curve.

Levels:
    L0 — no redaction (baseline)
    L1 — regex only
    L2 — full pipeline (placeholder)
    L2-mask — full pipeline (mask mode)
    L3 — aggressive (all labels, min_score 0.15)

Output:
    reports/figures/e1_privacy_utility.{png,svg}
    reports/tables/e1_privacy_utility.csv
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure the project is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rta.config import get_config
from rta.eval.match import match_overlap
from rta.ingest import load
from rta.redact import apply
from rta.resolve import merge
from rta.schema import GoldDoc, Span, read_gold_docs_jsonl
from rta.taxonomy import DIRECT_IDENTIFIERS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QA evaluation (token F1 against gold answer, no LLM needed)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Simple whitespace+punctuation tokenizer for answer matching."""
    import re

    return [t.lower() for t in re.findall(r"\w+", text)]


def _token_f1(predicted: str, gold: str) -> float:
    """Compute token-level F1 between predicted and gold answer strings."""
    pred_tokens = _tokenize(predicted)
    gold_tokens = _tokenize(gold)
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = sum(1 for t in pred_tokens if t in gold_tokens)
    if common == 0:
        return 0.0
    p = common / len(pred_tokens)
    r = common / len(gold_tokens)
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Redaction at each level
# ---------------------------------------------------------------------------


def _redact_at_level(
    doc_text: str,
    gold_doc: GoldDoc,
    level: str,
) -> tuple[str, list[Span]]:
    """Redact *doc_text* at the specified level.

    Returns (redacted_text, predicted_spans).
    """
    cfg = get_config()
    doc = load(doc_text)

    if level == "L0":
        return doc_text, []

    if level == "L1":
        from rta.detect.regex import run as regex_run

        pred = regex_run(doc, doc_type=gold_doc.doc_type)
        pred = merge(pred, doc_text=doc.text)
        result = apply(doc, pred, mode="placeholder")
        return result.text, pred

    if level in ("L2", "L2-mask"):
        from rta.detect.ner import run as ner_run
        from rta.detect.regex import run as regex_run

        regex_spans = regex_run(doc, doc_type=gold_doc.doc_type)
        try:
            ner_spans = ner_run(doc)
        except Exception:
            ner_spans = []
        pred = merge(regex_spans + ner_spans, doc_text=doc.text)
        mode = "mask" if level == "L2-mask" else "placeholder"
        result = apply(doc, pred, mode=mode)
        return result.text, pred

    if level == "L3":
        from rta.detect.ner import run as ner_run
        from rta.detect.regex import run as regex_run

        # Save and override min_score for aggressive mode
        orig_min = cfg.resolve.min_score
        try:
            object.__setattr__(cfg.resolve, "min_score", 0.15)
            regex_spans = regex_run(doc, doc_type=gold_doc.doc_type)
            try:
                ner_spans = ner_run(doc)
            except Exception:
                ner_spans = []
            pred = merge(regex_spans + ner_spans, doc_text=doc.text)
            result = apply(doc, pred, mode="placeholder")
            return result.text, pred
        finally:
            object.__setattr__(cfg.resolve, "min_score", orig_min)

    raise ValueError(f"Unknown level: {level}")


def _simulate_qa_answer(
    redacted_text: str,
    question: str,
    gold_answer: str,
    answer_type: str,
) -> str:
    """Simulate a QA answer without an LLM.

    For non_pii questions, search the redacted text for the gold answer.
    For pii_dependent questions, the answer is in placeholder form.
    """
    # Simple extractive approach: if gold answer is in the redacted text, return it
    if gold_answer.lower() in redacted_text.lower():
        return gold_answer

    # For placeholder answers, check if placeholder is in text
    if answer_type == "pii_dependent" and "[" in gold_answer:
        return gold_answer if gold_answer in redacted_text else ""

    # Fallback: return empty (couldn't find answer)
    return ""


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment() -> None:
    """Run the E1 privacy-vs-utility experiment."""
    cfg = get_config()
    gold_path = Path(cfg.paths.data) / "gold" / "handlabel.jsonl"
    qa_path = Path(cfg.paths.data) / "gold" / "qa_template.jsonl"
    fig_dir = Path(cfg.paths.reports) / "figures"
    tbl_dir = Path(cfg.paths.reports) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    if not gold_path.exists():
        logger.error("Gold annotations not found: %s", gold_path)
        return
    if not qa_path.exists():
        logger.error("QA pairs not found: %s", qa_path)
        return

    gold_docs = read_gold_docs_jsonl(gold_path)

    with qa_path.open(encoding="utf-8") as f:
        qa_pairs = [json.loads(line) for line in f if line.strip()]

    levels = ["L0", "L1", "L2", "L2-mask", "L3"]
    results: dict[str, dict[str, Any]] = {}

    for level in levels:
        logger.info("--- Level %s ---", level)
        pii_recall_scores: list[float] = []
        qa_scores_nonpii: list[float] = []
        qa_scores_pii: list[float] = []

        # Cache redacted texts per doc
        redacted_cache: dict[str, tuple[str, list[Span]]] = {}

        for gdoc in gold_docs:
            redacted_text, pred_spans = _redact_at_level(gdoc.text, gdoc, level)
            redacted_cache[gdoc.doc_id] = (redacted_text, pred_spans)

            # Compute PII recall
            if gdoc.spans and level != "L0":
                tp, _fp, fn = match_overlap(pred_spans, gdoc.spans)
                # Direct-ID recall
                tp_direct = [s for s in tp if s.label in DIRECT_IDENTIFIERS]
                fn_direct = [s for s in fn if s.label in DIRECT_IDENTIFIERS]
                total = len(tp_direct) + len(fn_direct)
                if total > 0:
                    pii_recall_scores.append(len(tp_direct) / total)
            elif level == "L0":
                pii_recall_scores.append(0.0)

        # Score QA pairs
        for qa in qa_pairs:
            doc_id = qa["doc_id"]
            if doc_id not in redacted_cache:
                continue
            redacted_text, _spans = redacted_cache[doc_id]
            predicted = _simulate_qa_answer(
                redacted_text, qa["question"], qa["answer"], qa["answer_type"]
            )
            f1 = _token_f1(predicted, qa["answer"])
            if qa["answer_type"] == "pii_dependent":
                qa_scores_pii.append(f1)
            else:
                qa_scores_nonpii.append(f1)

        pii_recall = float(np.mean(pii_recall_scores)) if pii_recall_scores else 0.0
        qa_nonpii = float(np.mean(qa_scores_nonpii)) if qa_scores_nonpii else 0.0
        qa_pii = float(np.mean(qa_scores_pii)) if qa_scores_pii else 0.0

        results[level] = {
            "pii_recall": pii_recall,
            "qa_nonpii": qa_nonpii,
            "qa_pii": qa_pii,
        }
        logger.info(
            "  PII recall=%.3f  QA(non-pii)=%.3f  QA(pii-dep)=%.3f",
            pii_recall,
            qa_nonpii,
            qa_pii,
        )

    # --- Write CSV ---
    csv_path = tbl_dir / "e1_privacy_utility.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "pii_recall", "qa_token_f1_nonpii", "qa_token_f1_pii"])
        for level in levels:
            r = results[level]
            writer.writerow(
                [level, f"{r['pii_recall']:.4f}", f"{r['qa_nonpii']:.4f}", f"{r['qa_pii']:.4f}"]
            )
    logger.info("Wrote %s", csv_path)

    # --- Write Markdown ---
    md_path = tbl_dir / "e1_privacy_utility.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# E1: Privacy vs. Utility\n\n")
        f.write(
            "| Level | Description | QA Token F1 (non-PII) | QA Token F1 (PII-dep) | PII Recall |\n"
        )
        f.write("|---|---|---|---|---|\n")
        descriptions = {
            "L0": "No redaction",
            "L1": "Regex only",
            "L2": "Full pipeline (placeholder)",
            "L2-mask": "Full pipeline (mask)",
            "L3": "Aggressive (min_score=0.15)",
        }
        for level in levels:
            r = results[level]
            f.write(
                f"| {level} | {descriptions[level]} "
                f"| {r['qa_nonpii']:.4f} "
                f"| {r['qa_pii']:.4f} "
                f"| {r['pii_recall']:.4f} |\n"
            )
    logger.info("Wrote %s", md_path)

    # --- Plot ---
    _plot_curve(results, levels, fig_dir)


def _plot_curve(
    results: dict[str, dict[str, Any]],
    levels: list[str],
    fig_dir: Path,
) -> None:
    """Plot the privacy-utility trade-off curve."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")
        return

    # Style
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    xs = [results[lv]["pii_recall"] for lv in levels]
    ys_nonpii = [results[lv]["qa_nonpii"] for lv in levels]
    ys_pii = [results[lv]["qa_pii"] for lv in levels]

    ax.plot(
        xs, ys_nonpii, "o-", color="#0F9D8A", label="QA F1 (non-PII)", linewidth=2, markersize=8
    )
    ax.plot(
        xs, ys_pii, "s--", color="#F2A93B", label="QA F1 (PII-dependent)", linewidth=2, markersize=8
    )

    for i, level in enumerate(levels):
        ax.annotate(
            level,
            (xs[i], ys_nonpii[i]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            color="#5B6472",
        )

    ax.set_xlabel("PII Recall (Direct Identifiers)")
    ax.set_ylabel("QA Token F1")
    ax.set_title("E1: Privacy vs. Utility Trade-off")
    ax.legend(loc="best")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(fig_dir / "e1_privacy_utility.png", dpi=200)
    fig.savefig(fig_dir / "e1_privacy_utility.svg")
    plt.close(fig)
    logger.info("Wrote figures to %s", fig_dir)


if __name__ == "__main__":
    run_experiment()

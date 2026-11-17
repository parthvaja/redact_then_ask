"""E3: Error propagation experiment.

Oracle ablation: replace each upstream stage with ground truth in turn
and measure the change in final direct-ID recall on the hand-labeled set.

Also injects controlled errors (force wrong doc type for 20% of docs)
and measures the recall drop.

Output:
    reports/figures/e3_error_propagation.{png,svg}
    reports/tables/e3_error_propagation.csv
"""

from __future__ import annotations

import csv
import logging
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rta.config import get_config
from rta.eval.match import match_overlap
from rta.ingest import load
from rta.resolve import merge
from rta.schema import GoldDoc, Span, read_gold_docs_jsonl
from rta.taxonomy import DIRECT_IDENTIFIERS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _direct_id_recall(pred_spans: list[Span], gold_spans: list[Span]) -> float:
    """Compute recall over direct identifiers only."""
    gold_direct = [s for s in gold_spans if s.label in DIRECT_IDENTIFIERS]
    if not gold_direct:
        return 1.0  # No direct IDs to miss

    tp, _fp, fn = match_overlap(pred_spans, gold_direct)
    total = len(tp) + len(fn)
    return len(tp) / total if total > 0 else 0.0


def _run_pipeline_full(gdoc: GoldDoc) -> list[Span]:
    """Run the full pipeline (no oracles) and return predicted spans."""
    from rta.detect.ner import run as ner_run
    from rta.detect.regex import run as regex_run

    doc = load(gdoc.text)
    regex_spans = regex_run(doc, doc_type=gdoc.doc_type)
    try:
        ner_spans = ner_run(doc)
    except Exception:
        ner_spans = []
    return merge(regex_spans + ner_spans, doc_text=doc.text)


def _run_oracle_langid(gdoc: GoldDoc) -> list[Span]:
    """Oracle language ID: use gold language (always 'en')."""
    # Language doesn't affect regex/NER for English, so this is the same
    # as baseline — the oracle shows no effect for monolingual English.
    return _run_pipeline_full(gdoc)


def _run_oracle_doctype(gdoc: GoldDoc) -> list[Span]:
    """Oracle doc type: use gold doc_type for routing."""
    from rta.detect.ner import run as ner_run
    from rta.detect.regex import run as regex_run
    from rta.detect.rules import boost_ner_scores, get_title_cues

    doc = load(gdoc.text)
    # Use the gold doc type directly
    doc_types = [gdoc.doc_type] if gdoc.doc_type else []
    regex_spans = regex_run(doc, doc_types=doc_types)
    try:
        ner_spans = ner_run(doc)
        title_cues = get_title_cues(doc_types)
        if title_cues:
            ner_spans = boost_ner_scores(ner_spans, doc.text, title_cues)
    except Exception:
        ner_spans = []
    return merge(regex_spans + ner_spans, doc_text=doc.text)


def _run_oracle_regex(gdoc: GoldDoc) -> list[Span]:
    """Oracle regex: inject gold spans for regex-detectable types."""
    from rta.detect.ner import run as ner_run

    doc = load(gdoc.text)
    # Use gold spans for regex-detectable labels
    regex_labels = {
        "EMAIL",
        "PHONE",
        "SSN",
        "CREDIT_CARD",
        "IBAN",
        "IP_ADDRESS",
        "URL",
        "DATE",
        "AGE",
        "MRN",
        "CASE_NUMBER",
        "ID_NUMBER",
    }
    gold_regex = [s for s in gdoc.spans if s.label in regex_labels]
    try:
        ner_spans = ner_run(doc)
    except Exception:
        ner_spans = []
    return merge(gold_regex + ner_spans, doc_text=doc.text)


def _run_oracle_ner(gdoc: GoldDoc) -> list[Span]:
    """Oracle NER: inject gold spans for NER-detectable types."""
    from rta.detect.regex import run as regex_run

    doc = load(gdoc.text)
    ner_labels = {"PERSON", "ORG", "LOCATION", "ADDRESS"}
    gold_ner = [s for s in gdoc.spans if s.label in ner_labels]
    regex_spans = regex_run(doc, doc_type=gdoc.doc_type)
    return merge(regex_spans + gold_ner, doc_text=doc.text)


def _run_wrong_doctype(gdoc: GoldDoc, wrong_types: list[str]) -> list[Span]:
    """Force a wrong doc type to measure misclassification cost."""
    from rta.detect.ner import run as ner_run
    from rta.detect.regex import run as regex_run
    from rta.detect.rules import boost_ner_scores, get_title_cues

    doc = load(gdoc.text)
    # Pick a random wrong type
    rng = random.Random(13)
    candidates = [t for t in wrong_types if t != gdoc.doc_type]
    wrong_type = rng.choice(candidates) if candidates else gdoc.doc_type
    doc_types = [wrong_type]

    regex_spans = regex_run(doc, doc_types=doc_types)
    try:
        ner_spans = ner_run(doc)
        title_cues = get_title_cues(doc_types)
        if title_cues:
            ner_spans = boost_ner_scores(ner_spans, doc.text, title_cues)
    except Exception:
        ner_spans = []
    return merge(regex_spans + ner_spans, doc_text=doc.text)


def run_experiment() -> None:
    """Run the E3 error propagation experiment."""
    cfg = get_config()
    gold_path = Path(cfg.paths.data) / "gold" / "handlabel.jsonl"
    fig_dir = Path(cfg.paths.reports) / "figures"
    tbl_dir = Path(cfg.paths.reports) / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    if not gold_path.exists():
        logger.error("Gold annotations not found: %s", gold_path)
        return

    gold_docs = read_gold_docs_jsonl(gold_path)
    all_doc_types = list({d.doc_type for d in gold_docs if d.doc_type})

    ablations = {
        "Baseline (no oracle)": _run_pipeline_full,
        "Oracle language ID": _run_oracle_langid,
        "Oracle doc type": _run_oracle_doctype,
        "Oracle regex": _run_oracle_regex,
        "Oracle NER": _run_oracle_ner,
    }

    results: dict[str, float] = {}

    for name, fn in ablations.items():
        logger.info("--- %s ---", name)
        recalls = []
        for gdoc in gold_docs:
            pred = fn(gdoc)
            r = _direct_id_recall(pred, gdoc.spans)
            recalls.append(r)
        mean_recall = float(np.mean(recalls))
        results[name] = mean_recall
        logger.info("  Direct-ID recall = %.4f", mean_recall)

    # Controlled error: wrong doc type for 20% of docs
    logger.info("--- Wrong doc type (20%%) ---")
    rng = random.Random(13)
    recalls_error = []
    for gdoc in gold_docs:
        if rng.random() < 0.2:
            pred = _run_wrong_doctype(gdoc, all_doc_types)
        else:
            pred = _run_pipeline_full(gdoc)
        r = _direct_id_recall(pred, gdoc.spans)
        recalls_error.append(r)
    results["Wrong doc type (20%)"] = float(np.mean(recalls_error))
    logger.info("  Direct-ID recall = %.4f", results["Wrong doc type (20%)"])

    baseline = results["Baseline (no oracle)"]

    # --- Write CSV ---
    csv_path = tbl_dir / "e3_error_propagation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ablation", "direct_id_recall", "delta_vs_baseline"])
        for name, recall in results.items():
            delta = recall - baseline
            writer.writerow([name, f"{recall:.4f}", f"{delta:+.4f}"])
    logger.info("Wrote %s", csv_path)

    # --- Write Markdown ---
    md_path = tbl_dir / "e3_error_propagation.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# E3: Error Propagation\n\n")
        f.write("| Ablation | Direct-ID Recall | Delta vs Baseline |\n")
        f.write("|---|---|---|\n")
        for name, recall in results.items():
            delta = recall - baseline
            f.write(f"| {name} | {recall:.4f} | {delta:+.4f} |\n")
    logger.info("Wrote %s", md_path)

    # --- Plot ---
    _plot_bars(results, baseline, fig_dir)


def _plot_bars(
    results: dict[str, float],
    baseline: float,
    fig_dir: Path,
) -> None:
    """Plot the error propagation bar chart."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")
        return

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

    # Only oracle ablations (not baseline or error injection)
    oracle_names = [n for n in results if n.startswith("Oracle")]
    deltas = [results[n] - baseline for n in oracle_names]
    short_names = [n.replace("Oracle ", "") for n in oracle_names]

    # Add the error injection
    if "Wrong doc type (20%)" in results:
        short_names.append("Wrong type (20%)")
        deltas.append(results["Wrong doc type (20%)"] - baseline)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#0F9D8A" if d >= 0 else "#DC3232" for d in deltas]
    bars = ax.barh(short_names, deltas, color=colors, edgecolor="white", height=0.5)

    ax.set_xlabel("Change in Direct-ID Recall vs. Baseline")
    ax.set_title(f"E3: Error Propagation (Baseline recall = {baseline:.3f})")
    ax.axvline(x=0, color="#0E1726", linewidth=0.8)

    for bar, delta in zip(bars, deltas, strict=False):
        ax.text(
            bar.get_width() + 0.005 * (1 if delta >= 0 else -1),
            bar.get_y() + bar.get_height() / 2,
            f"{delta:+.3f}",
            va="center",
            ha="left" if delta >= 0 else "right",
            fontsize=10,
            color="#5B6472",
        )

    fig.tight_layout()
    fig.savefig(fig_dir / "e3_error_propagation.png", dpi=200)
    fig.savefig(fig_dir / "e3_error_propagation.svg")
    plt.close(fig)
    logger.info("Wrote figures to %s", fig_dir)


if __name__ == "__main__":
    run_experiment()

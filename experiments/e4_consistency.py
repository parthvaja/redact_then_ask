"""E4: Pseudonym consistency experiment.

Measures cluster purity and placeholder purity on documents with
repeated mentions, with and without fuzzy matching.

Cluster purity: fraction of gold clusters whose mentions all received
one placeholder (all mentions of the same person → same [PERSON_N]).

Placeholder purity: fraction of placeholders that map to exactly one
gold entity.

Output:
    reports/tables/e4_consistency.csv
    reports/tables/e4_consistency.md
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rta.config import get_config
from rta.ingest import load
from rta.redact import apply
from rta.resolve import merge
from rta.schema import GoldDoc, Span, read_gold_docs_jsonl

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_gold_clusters(gold_spans: list[Span]) -> list[list[Span]]:
    """Group gold spans that refer to the same entity (by text)."""
    by_text: dict[str, list[Span]] = {}
    for s in gold_spans:
        key = s.text.lower().strip()
        by_text.setdefault(key, []).append(s)
    return list(by_text.values())


def _compute_cluster_purity(
    gold_clusters: list[list[Span]],
    redacted_text: str,
    offset_map: list[tuple[int, int, int, int]],
    original_text: str,
) -> float:
    """Fraction of gold clusters whose spans all map to one placeholder."""
    if not gold_clusters:
        return 1.0

    # Build a mapping from original offset to placeholder
    span_to_placeholder: dict[tuple[int, int], str] = {}
    for orig_start, orig_end, red_start, red_end in offset_map:
        placeholder = redacted_text[red_start:red_end]
        span_to_placeholder[(orig_start, orig_end)] = placeholder

    pure_count = 0
    for cluster in gold_clusters:
        placeholders_seen = set()
        for s in cluster:
            # Find the closest matching offset
            ph = span_to_placeholder.get((s.start, s.end))
            if ph is None:
                # Try to find overlapping offset
                for (os_, oe), p in span_to_placeholder.items():
                    if os_ <= s.start and s.end <= oe:
                        ph = p
                        break
            if ph is not None:
                placeholders_seen.add(ph)
        if len(placeholders_seen) == 1:
            pure_count += 1

    return pure_count / len(gold_clusters)


def _compute_placeholder_purity(
    gold_clusters: list[list[Span]],
    redacted_text: str,
    offset_map: list[tuple[int, int, int, int]],
) -> float:
    """Fraction of placeholders that map to exactly one gold entity."""
    if not offset_map:
        return 1.0

    # Map each placeholder to the gold texts it covers
    placeholder_to_golds: dict[str, set[str]] = {}
    for orig_start, orig_end, red_start, red_end in offset_map:
        placeholder = redacted_text[red_start:red_end]
        # Find which gold cluster this span belongs to
        for cluster in gold_clusters:
            for gs in cluster:
                if gs.start == orig_start and gs.end == orig_end:
                    gold_key = cluster[0].text.lower().strip()
                    placeholder_to_golds.setdefault(placeholder, set()).add(gold_key)
                    break

    if not placeholder_to_golds:
        return 1.0

    pure = sum(1 for golds in placeholder_to_golds.values() if len(golds) == 1)
    return pure / len(placeholder_to_golds)


def _run_with_fuzzy(gdoc: GoldDoc, fuzzy: bool) -> tuple:
    """Run the pipeline with or without fuzzy clustering."""
    cfg = get_config()
    doc = load(gdoc.text)

    from rta.detect.ner import run as ner_run
    from rta.detect.regex import run as regex_run

    regex_spans = regex_run(doc, doc_type=gdoc.doc_type)
    try:
        ner_spans = ner_run(doc)
    except Exception:
        ner_spans = []
    pred = merge(regex_spans + ner_spans, doc_text=doc.text)

    # Temporarily adjust fuzzy threshold
    orig_threshold = cfg.redact.fuzzy_threshold
    if not fuzzy:
        object.__setattr__(cfg.redact, "fuzzy_threshold", 101)  # Effectively disable fuzzy

    try:
        result = apply(doc, pred, mode="placeholder")
    finally:
        object.__setattr__(cfg.redact, "fuzzy_threshold", orig_threshold)

    return result, pred


def run_experiment() -> None:
    """Run the E4 pseudonym consistency experiment."""
    cfg = get_config()
    gold_path = Path(cfg.paths.data) / "gold" / "handlabel.jsonl"
    tbl_dir = Path(cfg.paths.reports) / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)

    if not gold_path.exists():
        logger.error("Gold annotations not found: %s", gold_path)
        return

    gold_docs = read_gold_docs_jsonl(gold_path)

    conditions = [
        ("With fuzzy matching", True),
        ("Without fuzzy matching", False),
    ]

    results: dict[str, dict[str, float]] = {}

    for cond_name, fuzzy in conditions:
        logger.info("--- %s ---", cond_name)
        cluster_purities: list[float] = []
        placeholder_purities: list[float] = []

        for gdoc in gold_docs:
            redaction_result, _pred_spans = _run_with_fuzzy(gdoc, fuzzy)
            gold_clusters = _build_gold_clusters(gdoc.spans)

            cp = _compute_cluster_purity(
                gold_clusters,
                redaction_result.text,
                redaction_result.offset_map,
                gdoc.text,
            )
            pp = _compute_placeholder_purity(
                gold_clusters,
                redaction_result.text,
                redaction_result.offset_map,
            )
            cluster_purities.append(cp)
            placeholder_purities.append(pp)

        mean_cp = float(np.mean(cluster_purities)) if cluster_purities else 0.0
        mean_pp = float(np.mean(placeholder_purities)) if placeholder_purities else 0.0

        results[cond_name] = {
            "cluster_purity": mean_cp,
            "placeholder_purity": mean_pp,
        }
        logger.info("  Cluster purity=%.4f  Placeholder purity=%.4f", mean_cp, mean_pp)

    # --- Write CSV ---
    csv_path = tbl_dir / "e4_consistency.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "cluster_purity", "placeholder_purity"])
        for name, metrics in results.items():
            writer.writerow(
                [name, f"{metrics['cluster_purity']:.4f}", f"{metrics['placeholder_purity']:.4f}"]
            )
    logger.info("Wrote %s", csv_path)

    # --- Write Markdown ---
    md_path = tbl_dir / "e4_consistency.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# E4: Pseudonym Consistency\n\n")
        f.write("| Condition | Cluster Purity | Placeholder Purity |\n")
        f.write("|---|---|---|\n")
        for name, metrics in results.items():
            f.write(
                f"| {name} | {metrics['cluster_purity']:.4f} | {metrics['placeholder_purity']:.4f} |\n"
            )
        f.write("\nCluster purity: fraction of gold entity clusters mapped to one placeholder.\n")
        f.write("Placeholder purity: fraction of placeholders mapping to one gold entity.\n")
    logger.info("Wrote %s", md_path)


if __name__ == "__main__":
    run_experiment()

"""Report generation for evaluation results (Phase 9).

Writes CSV and Markdown tables to ``reports/tables/``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_report(
    results: dict[str, dict[str, Any]],
    system_name: str,
    dataset: str,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write CSV and Markdown tables from evaluation *results*.

    Args:
        results:     Dict keyed by label with P/R/F1/F2/TP/FP/FN.
        system_name: System label for the filename.
        dataset:     Dataset label for the filename.
        output_dir:  Directory to write into (created if needed).

    Returns:
        ``(csv_path, md_path)``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"detection_{dataset}_{system_name}"

    # --- CSV ---
    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "precision", "recall", "f1", "f2", "tp", "fp", "fn"])
        for label in sorted(results):
            m = results[label]
            writer.writerow(
                [
                    label,
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['f1']:.4f}",
                    f"{m.get('f2', 0.0):.4f}",
                    m.get("tp", ""),
                    m.get("fp", ""),
                    m.get("fn", ""),
                ]
            )

    # --- Markdown ---
    md_path = output_dir / f"{stem}.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Detection results: {system_name} on {dataset}\n\n")
        fh.write("| Label | P | R | F1 | F2 | TP | FP | FN |\n")
        fh.write("|---|---|---|---|---|---|---|---|\n")
        for label in sorted(results):
            m = results[label]
            fh.write(
                f"| {label} "
                f"| {m['precision']:.4f} "
                f"| {m['recall']:.4f} "
                f"| {m['f1']:.4f} "
                f"| {m.get('f2', 0.0):.4f} "
                f"| {m.get('tp', '')} "
                f"| {m.get('fp', '')} "
                f"| {m.get('fn', '')} |\n"
            )

    logger.info("Wrote reports to %s and %s", csv_path, md_path)
    return csv_path, md_path


def generate_comparison_table(
    all_results: dict[str, dict[str, dict[str, Any]]],
    dataset: str,
    output_dir: str | Path,
) -> Path:
    """Write a combined Markdown comparison table for multiple systems.

    Args:
        all_results: ``{system_name: {label: metrics_dict}}``.
        dataset:     Dataset label.
        output_dir:  Output directory.

    Returns:
        Path to the written Markdown file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"comparison_{dataset}.md"
    systems = sorted(all_results)

    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Comparison on {dataset}\n\n")
        header = "| System | Direct-ID Recall | Micro F1 | F2 | Leak rate |\n"
        fh.write(header)
        fh.write("|---|---|---|---|---|\n")

        for sys in systems:
            micro = all_results[sys].get("micro", {})
            fh.write(
                f"| {sys} "
                f"| {micro.get('recall', 0.0):.4f} "
                f"| {micro.get('f1', 0.0):.4f} "
                f"| {micro.get('f2', 0.0):.4f} "
                f"| -- |\n"
            )

    return md_path

"""Text Anonymization Benchmark (TAB) dataset loader.

Downloads from GitHub (NorskRegnesentral/text-anonymization-benchmark),
converts ECHR court-case annotations to ``GoldDoc`` JSONL.

TAB JSON structure (verified 2026-11-13)::

    [                                    # list of documents
      {
        "doc_id":        "001-83927",
        "text":          "PROCEDURE\\n...",
        "dataset_type":  "dev",          # "train", "dev", or "test"
        "task":          "Task: ...",
        "quality_checked": true,
        "meta": { "year": 2007, "countries": "TUR", ... },
        "annotations": {
          "annotator1": {
            "entity_mentions": [
              {
                "entity_type":       "PERSON",       # CODE, ORG, DEM, PERSON, DATETIME, LOC, QUANTITY, MISC
                "start_offset":      254,
                "end_offset":        269,
                "span_text":         "Mr Cengiz Polat",
                "identifier_type":   "DIRECT",       # DIRECT, QUASI, NO_MASK
                "entity_id":         "001-83927_a1_e4",
                "entity_mention_id": "001-83927_a1_em4",
                "confidential_status": "NOT_CONFIDENTIAL",
                "edit_type":         "check",
              },
              ...
            ]
          },
          "annotator2": { ... },
          ...
        }
      },
      ...
    ]

We use **annotator1**'s mentions as ground truth, keeping only
``identifier_type`` in ``{"DIRECT", "QUASI"}`` (i.e. not ``NO_MASK``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from rta.config import get_config
from rta.schema import GoldDoc, Span, write_jsonl
from rta.taxonomy import TAB_TO_LABEL, normalize_label

logger = logging.getLogger(__name__)

_REPO_RAW = (
    "https://raw.githubusercontent.com/NorskRegnesentral/text-anonymization-benchmark/master"
)
_SPLITS: dict[str, str] = {
    "train": f"{_REPO_RAW}/echr_train.json",
    "dev": f"{_REPO_RAW}/echr_dev.json",
    "test": f"{_REPO_RAW}/echr_test.json",
}


def fetch(data_dir: Path | None = None) -> Path:
    """Download the three TAB JSON files from GitHub.

    Args:
        data_dir: Override for the raw data directory.  If ``None``,
                  uses ``<cfg.paths.data>/raw/tab/``.

    Returns:
        Path to the directory containing the downloaded files.
    """
    cfg = get_config()
    out_dir = Path(data_dir) if data_dir else Path(cfg.paths.data) / "raw" / "tab"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, url in _SPLITS.items():
        dest = out_dir / f"echr_{split_name}.json"
        if dest.exists():
            logger.info("TAB %s already downloaded: %s", split_name, dest)
            continue

        logger.info("Downloading TAB %s from %s ...", split_name, url)
        resp = httpx.get(url, timeout=120, follow_redirects=True)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        logger.info("  → %s (%d bytes)", dest, len(resp.content))

    return out_dir


def convert(
    split: str = "test",
    annotator: str = "annotator1",
    data_dir: Path | None = None,
    max_docs: int | None = None,
) -> Path:
    """Convert TAB JSON to ``GoldDoc`` JSONL.

    Args:
        split:     ``"train"``, ``"dev"``, or ``"test"``.
        annotator: Which annotator's mentions to use as gold.
        data_dir:  Override for ``<cfg.paths.data>/raw/tab/``.
        max_docs:  Cap the number of output documents.

    Returns:
        Path to the written JSONL file.
    """
    cfg = get_config()
    raw_dir = Path(data_dir) if data_dir else Path(cfg.paths.data) / "raw" / "tab"
    src_path = raw_dir / f"echr_{split}.json"

    if not src_path.exists():
        raise FileNotFoundError(
            f"TAB {split} file not found at {src_path}. Run `rta data fetch` first."
        )

    out_dir = Path(cfg.paths.data) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tab_{split}.jsonl"

    logger.info("Converting TAB %s from %s ...", split, src_path)
    with src_path.open(encoding="utf-8") as fh:
        raw_docs: list[dict] = json.load(fh)

    gold_docs: list[GoldDoc] = []
    skipped_labels: dict[str, int] = {}
    total_spans = 0

    for raw in raw_docs:
        doc_id: str = raw["doc_id"]
        text: str = raw["text"]

        # Pick the annotator's mentions
        annotations = raw.get("annotations", {})
        ann_data = annotations.get(annotator)
        if ann_data is None:
            # Fall back to whichever annotator is first
            if annotations:
                first_key = next(iter(annotations))
                ann_data = annotations[first_key]
                logger.debug("Doc %s: %s missing, using %s", doc_id, annotator, first_key)
            else:
                logger.debug("Doc %s has no annotations", doc_id)
                continue

        mentions: list[dict] = ann_data.get("entity_mentions", [])

        spans: list[Span] = []
        for m in mentions:
            # Only keep DIRECT and QUASI identifiers
            id_type = m.get("identifier_type", "")
            if id_type == "NO_MASK":
                continue

            entity_type: str = m.get("entity_type", "")
            start: int = m["start_offset"]
            end: int = m["end_offset"]

            # Map to taxonomy
            try:
                label = normalize_label(entity_type, TAB_TO_LABEL)
            except KeyError:
                skipped_labels[entity_type] = skipped_labels.get(entity_type, 0) + 1
                continue

            if label is None:
                continue

            # Validate offsets
            if start < 0 or end <= start or end > len(text):
                logger.debug(
                    "Skipping bad offsets [%d:%d] in doc %s (text len %d)",
                    start,
                    end,
                    doc_id,
                    len(text),
                )
                continue

            span_text = text[start:end]
            spans.append(
                Span(
                    start=start,
                    end=end,
                    label=label,
                    text=span_text,
                    score=1.0,
                    detector="gold",
                )
            )
            total_spans += 1

        if spans:
            gold_docs.append(
                GoldDoc(
                    doc_id=doc_id,
                    text=text,
                    spans=spans,
                    doc_type="court_case",
                    language="en",
                    source="tab",
                )
            )

        if max_docs is not None and len(gold_docs) >= max_docs:
            break

    write_jsonl(gold_docs, out_path)

    if skipped_labels:
        logger.warning("Unknown TAB labels skipped: %s", skipped_labels)

    logger.info(
        "Wrote %d documents (%d spans) to %s",
        len(gold_docs),
        total_spans,
        out_path,
    )
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetch()
    for s in ("train", "dev", "test"):
        convert(split=s)

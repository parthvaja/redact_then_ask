"""ai4privacy/pii-masking-300k dataset loader.

Downloads from Hugging Face Datasets, filters to English, converts to
``GoldDoc`` JSONL using the taxonomy mapping in ``rta.taxonomy``.

Dataset fields (verified 2026-11-13)::

    source_text   str               The document text (PII is synthetic)
    privacy_mask  list[dict]        [{value, start, end, label}, ...]
    language      str               e.g. "English"
    id            str               Document ID like "40767A"
    set           str               "train" or "test"

``privacy_mask[].label`` values (27 unique, mapped in ``AI4PRIVACY_TO_LABEL``):
  BOD BUILDING CITY COUNTRY DATE DRIVERLICENSE EMAIL GEOCOORD
  GIVENNAME1 GIVENNAME2 IDCARD IP LASTNAME1 LASTNAME2 LASTNAME3
  PASS PASSPORT POSTCODE SECADDRESS SEX SOCIALNUMBER STATE STREET
  TEL TIME TITLE USERNAME
"""

from __future__ import annotations

import logging
from pathlib import Path

from rta.config import get_config
from rta.schema import GoldDoc, Span, write_jsonl
from rta.taxonomy import AI4PRIVACY_TO_LABEL, normalize_label

logger = logging.getLogger(__name__)

_DATASET_NAME = "ai4privacy/pii-masking-300k"


def fetch() -> Path:
    """Download the ai4privacy dataset from Hugging Face and cache locally.

    Returns:
        Path to the cache directory used by ``datasets``.
    """
    from datasets import load_dataset

    logger.info("Downloading %s from Hugging Face ...", _DATASET_NAME)
    # Trigger download and caching — streaming=False so it is cached on disk
    load_dataset(_DATASET_NAME, split="train")
    logger.info("Dataset cached.")
    cfg = get_config()
    return Path(cfg.paths.data) / "raw" / "ai4privacy"


def convert(
    split: str = "train",
    language: str = "English",
    max_docs: int | None = None,
) -> Path:
    """Convert ai4privacy records to ``GoldDoc`` JSONL.

    Args:
        split:    HF split name (``"train"``).
        language: Keep only records whose ``language`` field matches.
        max_docs: Cap the number of output documents (``None`` = all).

    Returns:
        Path to the written JSONL file.
    """
    from datasets import load_dataset

    cfg = get_config()
    out_dir = Path(cfg.paths.data) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    lang_tag = language.lower()[:2]  # "English" → "en"
    out_path = out_dir / f"ai4privacy_{lang_tag}_{split}.jsonl"

    logger.info("Loading %s split=%s, filtering language=%s ...", _DATASET_NAME, split, language)
    ds = load_dataset(_DATASET_NAME, split=split, streaming=True)

    gold_docs: list[GoldDoc] = []
    skipped_labels: dict[str, int] = {}
    total_spans = 0

    for record in ds:
        if record["language"] != language:
            continue

        doc_id = str(record["id"])
        text: str = record["source_text"]
        masks: list[dict] = record["privacy_mask"]

        spans: list[Span] = []
        for mask in masks:
            raw_label: str = mask["label"]
            start: int = mask["start"]
            end: int = mask["end"]

            # Map to taxonomy
            try:
                label = normalize_label(raw_label, AI4PRIVACY_TO_LABEL)
            except KeyError:
                skipped_labels[raw_label] = skipped_labels.get(raw_label, 0) + 1
                continue

            if label is None:
                # Label intentionally discarded (TIME, TITLE, SEX, etc.)
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
                    language=lang_tag,
                    source="ai4privacy",
                )
            )

        if max_docs is not None and len(gold_docs) >= max_docs:
            break

    write_jsonl(gold_docs, out_path)

    if skipped_labels:
        logger.warning("Unknown ai4privacy labels skipped: %s", skipped_labels)

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
    convert(split="train", max_docs=None)

"""Classification corpus builder.

Downloads ~300 documents per class from public datasets and writes them
as a single JSONL file (``data/processed/classify_corpus.jsonl``) with
fields ``doc_id``, ``text``, ``label``.

Sources (§6.3):
  - resume:     HF ``ahmedheakl/resume-atlas``
  - contract:   HF ``theatticusproject/cuad``  (full_contract_txt field)
  - medical:    MTSamples transcriptions (CSV from Kaggle mirror on HF)
  - email:      Enron emails — HF ``kddcup/enron_email``
  - court_case: TAB ECHR texts already downloaded in Phase 2

Leakage guard: ``strip_artifacts()`` removes formatting cues that let a
classifier cheat (e.g. "From:", "Subject:" headers in Enron mail, CUAD
boilerplate front-matter).  Both raw and stripped accuracies are reported.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from rta.config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Leakage-guard text strippers
# ---------------------------------------------------------------------------

_ENRON_HEADER_RE = re.compile(
    r"^(Message-ID|Date|From|To|Subject|Cc|Bcc|Mime-Version|Content-Type|"
    r"Content-Transfer-Encoding|X-From|X-To|X-cc|X-bcc|X-Folder|"
    r"X-Origin|X-FileName):.*$",
    re.MULTILINE,
)

_CUAD_BOILERPLATE_RE = re.compile(
    r"(?i)^(exhibit\s+\d+|this\s+agreement|"
    r"by\s+and\s+between|hereinafter\s+referred\s+to\s+as).*$",
    re.MULTILINE,
)


def strip_artifacts(text: str, label: str) -> str:
    """Remove formatting artifacts that would leak the document source."""
    if label == "email":
        text = _ENRON_HEADER_RE.sub("", text)
    elif label == "contract":
        text = _CUAD_BOILERPLATE_RE.sub("", text)
    # Collapse resulting blank-line runs
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# ---------------------------------------------------------------------------
# Per-class loaders (each returns list[dict] with doc_id, text, label)
# ---------------------------------------------------------------------------


def _load_resumes(max_docs: int = 300) -> list[dict]:
    """Load resume texts from HF ``ahmedheakl/resume-atlas``."""
    from datasets import load_dataset

    logger.info("Loading resumes from HF ahmedheakl/resume-atlas ...")
    ds = load_dataset("ahmedheakl/resume-atlas", split="train", streaming=True)

    docs: list[dict] = []
    for i, rec in enumerate(ds):
        # The dataset has a 'resume_text' or 'text' field
        text = rec.get("resume_text") or rec.get("text") or ""
        if not text or len(text) < 100:
            continue
        docs.append({"doc_id": f"resume_{i:05d}", "text": text, "label": "resume"})
        if len(docs) >= max_docs:
            break

    logger.info("  Loaded %d resume documents", len(docs))
    return docs


def _load_contracts(max_docs: int = 300) -> list[dict]:
    """Load contract texts from HF ``theatticusproject/cuad``."""
    from datasets import load_dataset

    logger.info("Loading contracts from HF theatticusproject/cuad ...")
    ds = load_dataset("theatticusproject/cuad", split="test", streaming=True)

    docs: list[dict] = []
    seen_titles: set[str] = set()
    for i, rec in enumerate(ds):
        # CUAD has context field with the contract text
        title = rec.get("title", f"contract_{i}")
        if title in seen_titles:
            continue
        seen_titles.add(title)

        text = rec.get("context") or ""
        if not text or len(text) < 100:
            continue
        docs.append({"doc_id": f"contract_{i:05d}", "text": text, "label": "contract"})
        if len(docs) >= max_docs:
            break

    logger.info("  Loaded %d contract documents", len(docs))
    return docs


def _load_medical(max_docs: int = 300) -> list[dict]:
    """Load medical transcriptions from MTSamples (HF mirror)."""
    from datasets import load_dataset

    logger.info("Loading medical transcriptions from HF mtsamples ...")
    # Try the HF mirror of MTSamples
    ds = load_dataset("mteb/mtsamples", split="train", streaming=True)

    docs: list[dict] = []
    for i, rec in enumerate(ds):
        text = rec.get("text") or rec.get("transcription") or ""
        if not text or len(text) < 100:
            continue
        docs.append({"doc_id": f"medical_{i:05d}", "text": text, "label": "medical"})
        if len(docs) >= max_docs:
            break

    logger.info("  Loaded %d medical documents", len(docs))
    return docs


def _load_emails(max_docs: int = 300) -> list[dict]:
    """Load Enron emails from HF."""
    from datasets import load_dataset

    logger.info("Loading Enron emails from HF ...")
    ds = load_dataset("yangwang825/enron-email-corpus", split="train", streaming=True)

    docs: list[dict] = []
    for i, rec in enumerate(ds):
        text = rec.get("text") or rec.get("email") or ""
        if not text or len(text) < 100:
            continue
        docs.append({"doc_id": f"email_{i:05d}", "text": text, "label": "email"})
        if len(docs) >= max_docs:
            break

    logger.info("  Loaded %d email documents", len(docs))
    return docs


def _load_court_cases(max_docs: int = 300) -> list[dict]:
    """Load court case texts from TAB data (already downloaded in Phase 2)."""
    cfg = get_config()
    raw_dir = Path(cfg.paths.data) / "raw" / "tab"

    docs: list[dict] = []
    for split in ("train", "dev", "test"):
        src = raw_dir / f"echr_{split}.json"
        if not src.exists():
            logger.warning("TAB %s not found at %s — skipping", split, src)
            continue
        with src.open(encoding="utf-8") as fh:
            raw_docs = json.load(fh)
        for raw in raw_docs:
            text = raw.get("text", "")
            if not text or len(text) < 100:
                continue
            docs.append(
                {
                    "doc_id": raw.get("doc_id", f"court_{len(docs):05d}"),
                    "text": text,
                    "label": "court_case",
                }
            )
            if len(docs) >= max_docs:
                break
        if len(docs) >= max_docs:
            break

    logger.info("  Loaded %d court_case documents", len(docs))
    return docs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_corpus(
    max_per_class: int = 300,
    strip: bool = True,
) -> Path:
    """Build the classification corpus JSONL.

    Args:
        max_per_class: Maximum documents per class.
        strip:         Whether to apply leakage-guard stripping.

    Returns:
        Path to the written JSONL file.
    """
    cfg = get_config()
    out_dir = Path(cfg.paths.data) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_stripped" if strip else "_raw"
    out_path = out_dir / f"classify_corpus{suffix}.jsonl"

    loaders = [
        _load_resumes,
        _load_contracts,
        _load_medical,
        _load_emails,
        _load_court_cases,
    ]

    all_docs: list[dict] = []
    for loader in loaders:
        docs = loader(max_docs=max_per_class)
        if strip:
            for d in docs:
                d["text"] = strip_artifacts(d["text"], d["label"])
        all_docs.extend(docs)

    with out_path.open("w", encoding="utf-8") as fh:
        for doc in all_docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info(
        "Wrote classification corpus: %d docs to %s",
        len(all_docs),
        out_path,
    )

    # Summary by class
    from collections import Counter

    counts = Counter(d["label"] for d in all_docs)
    for label, count in sorted(counts.items()):
        logger.info("  %s: %d", label, count)

    return out_path

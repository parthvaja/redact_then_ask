"""NER-based PII detection (Phase 5).

Two backends: ``spacy`` (default) and ``hf`` (Hugging Face token
classification).  Long documents use a sliding window with dedup.
All entity labels are mapped through ``rta.taxonomy`` before output.
"""

from __future__ import annotations

import logging

from rta.config import get_config
from rta.schema import Document, Span

logger = logging.getLogger(__name__)


def run(doc: Document) -> list[Span]:
    """Run the NER detector on *doc*.

    Args:
        doc: An ingested ``Document``.

    Returns:
        Unresolved ``Span`` list with ``detector`` starting with ``"ner:"``.
    """
    cfg = get_config()
    backend = cfg.detect.ner.backend

    if backend == "spacy":
        from rta.detect.ner.spacy_backend import detect

        spans = detect(doc)
    elif backend == "hf":
        from rta.detect.ner.hf_backend import detect

        spans = detect(doc)
    else:
        raise ValueError(f"Unknown NER backend: {backend!r}")

    logger.info("NER detection (%s): %d spans", backend, len(spans))
    return spans

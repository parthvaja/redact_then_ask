"""Document classification — ``Document`` → doc_type + probabilities.

Baseline: TF-IDF + Logistic Regression (Phase 3).
"""

from __future__ import annotations

import logging

from rta.schema import Document

logger = logging.getLogger(__name__)


def predict(doc: Document) -> tuple[str, dict[str, float]]:
    """Predict the document type of *doc*.

    Args:
        doc: An ingested ``Document``.

    Returns:
        A tuple ``(doc_type, probs)`` where ``doc_type`` is one of
        ``resume | contract | medical | email | court_case`` and
        ``probs`` is a dict of label → probability.
    """
    from rta.classify.baseline import predict_text

    doc_type, probs = predict_text(doc.text)
    doc.meta["doc_type"] = doc_type
    doc.meta["type_probs"] = probs
    return doc_type, probs


def train() -> None:
    """Train the TF-IDF + LR classifier.

    Builds the corpus (if not already built), then trains and saves
    the model.
    """
    from pathlib import Path

    from rta.classify.baseline import train_model
    from rta.config import get_config

    cfg = get_config()
    corpus_path = Path(cfg.paths.data) / "processed" / "classify_corpus_stripped.jsonl"

    if not corpus_path.exists():
        logger.info("Corpus not found — building it first ...")
        from rta.classify.corpus import build_corpus

        build_corpus()

    train_model(corpus_path)


def evaluate() -> None:
    """Evaluate the trained classifier on the held-out test split."""
    from rta.classify.baseline import evaluate_model

    evaluate_model()

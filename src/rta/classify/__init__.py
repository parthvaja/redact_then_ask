"""Document classification — ``Document`` → doc_type + probabilities.

Implemented in Phase 3.
"""

from __future__ import annotations

from rta.schema import Document


def predict(doc: Document) -> tuple[str, dict[str, float]]:
    """Predict the document type of *doc*.

    Args:
        doc: An ingested ``Document``.

    Returns:
        A tuple ``(doc_type, probs)`` where ``doc_type`` is one of
        ``resume | contract | medical | email | court_case`` and
        ``probs`` is a dict of label → probability.

    Raises:
        NotImplementedError: Until Phase 3 is complete.
    """
    raise NotImplementedError("rta.classify.predict — implemented in Phase 3")


def train() -> None:
    """Train the TF-IDF + LR classifier.  Implemented in Phase 3."""
    raise NotImplementedError("rta.classify.train — implemented in Phase 3")


def evaluate() -> None:
    """Evaluate the trained classifier.  Implemented in Phase 3."""
    raise NotImplementedError("rta.classify.evaluate — implemented in Phase 3")

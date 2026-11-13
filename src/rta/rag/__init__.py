"""Retrieval-augmented generation over redacted documents.

Implemented in Phase 10.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Answer:
    """An answer from the RAG system.

    Args:
        text:    The model's answer text.
        sources: Chunk IDs or indices that were retrieved as context.
    """

    text: str
    sources: list[int | str] = field(default_factory=list)


def index(redacted_text: str) -> object:
    """Chunk and embed *redacted_text*, store in a local vector DB.

    Args:
        redacted_text: The fully redacted document text.

    Returns:
        An opaque store handle passed to ``answer()``.

    Raises:
        NotImplementedError: Until Phase 10 is complete.
    """
    raise NotImplementedError("rta.rag.index — implemented in Phase 10")


def answer(store: object, question: str) -> Answer:
    """Retrieve relevant chunks and generate an answer to *question*.

    The original PII mapping is **never** placed in any prompt.

    Args:
        store:    Store handle returned by ``index()``.
        question: User's question in natural language.

    Returns:
        An ``Answer`` with the generated text and cited chunk IDs.

    Raises:
        NotImplementedError: Until Phase 10 is complete.
    """
    raise NotImplementedError("rta.rag.answer — implemented in Phase 10")

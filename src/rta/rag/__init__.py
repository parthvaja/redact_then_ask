"""Retrieval-augmented generation over redacted documents (Phase 10).

The original PII mapping is **never** placed in any prompt.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

from rta.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    """An answer from the RAG system."""

    text: str
    sources: list[int | str] = field(default_factory=list)


def index(redacted_text: str) -> object:
    """Chunk and embed *redacted_text*, store in a local vector DB.

    Returns an opaque store handle to pass to ``answer()``.
    """
    from rta.rag.chunk import chunk_text
    from rta.rag.index import VectorStore

    cfg = get_config()
    doc_hash = hashlib.sha256(redacted_text.encode()).hexdigest()
    chunks = chunk_text(
        redacted_text,
        chunk_chars=cfg.rag.chunk_chars,
        overlap=cfg.rag.overlap,
    )
    store = VectorStore(doc_hash, chunks)
    logger.info("Indexed %d chunks for document %s", len(chunks), doc_hash[:8])
    return store


def answer(store: object, question: str) -> Answer:
    """Retrieve relevant chunks and generate an answer to *question*.

    The original PII mapping is **never** placed in any prompt.
    """
    from rta.rag.index import VectorStore
    from rta.rag.llm import chat
    from rta.rag.prompts import SYSTEM_PROMPT, build_rag_prompt

    cfg = get_config()
    assert isinstance(store, VectorStore), "store must be a VectorStore"

    chunks = store.query(question, top_k=cfg.rag.top_k)
    user_prompt = build_rag_prompt(chunks, question)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response_text = chat(messages)
    except (ConnectionError, RuntimeError) as exc:
        logger.error("LLM unavailable: %s", exc)
        return Answer(
            text=f"Error: Could not reach local LLM. {exc}",
            sources=[c["chunk_id"] for c in chunks],
        )

    return Answer(
        text=response_text,
        sources=[c["chunk_id"] for c in chunks],
    )

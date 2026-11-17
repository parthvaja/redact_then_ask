"""Text chunking for RAG (Phase 10).

Splits text at sentence boundaries with configurable overlap.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_chars: int = 800,
    overlap: int = 150,
) -> list[dict]:
    """Split *text* into overlapping chunks at sentence boundaries.

    Args:
        text:        Input text to chunk.
        chunk_chars: Target chunk size in characters.
        overlap:     Overlap between consecutive chunks in characters.

    Returns:
        List of dicts with ``text`` and ``chunk_id`` keys.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[dict] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)

        if current_len + sent_len > chunk_chars and current:
            chunk_text_str = " ".join(current)
            chunks.append({"text": chunk_text_str, "chunk_id": len(chunks)})

            # Build overlap from the tail of current sentences
            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s)

            current = [*overlap_sents, sent]
            current_len = sum(len(s) for s in current)
        else:
            current.append(sent)
            current_len += sent_len

    if current:
        chunk_text_str = " ".join(current)
        chunks.append({"text": chunk_text_str, "chunk_id": len(chunks)})

    logger.debug("Chunked %d chars -> %d chunks", len(text), len(chunks))
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preferring pysbd if available."""
    try:
        import pysbd

        seg = pysbd.Segmenter(language="en", clean=False)
        return seg.segment(text)
    except ImportError:
        pass

    # Fallback: regex split on sentence-ending punctuation
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]

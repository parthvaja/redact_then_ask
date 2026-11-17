"""RAG prompt templates (Phase 10).

The system prompt instructs the LLM to respect privacy placeholders and
answer only from the provided context.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about documents.

IMPORTANT RULES:
1. The document text has been redacted for privacy. Bracketed tokens \
like [PERSON_1] or [EMAIL_2] are privacy placeholders.
2. You MUST keep these placeholders exactly as they appear. \
Never guess or reveal the real identity behind a placeholder.
3. Answer ONLY based on the provided context. Do not use external knowledge.
4. If the context does not contain enough information to answer, say so clearly.
5. Cite which chunk(s) you used in your answer."""

RAG_USER_TEMPLATE = """\
Context chunks from the document:

{context}

Question: {question}

Answer based only on the above context. Cite chunk numbers used."""


def build_rag_prompt(chunks: list[dict], question: str) -> str:
    """Build the user prompt from retrieved *chunks* and *question*."""
    context_parts = [f"[Chunk {c['chunk_id']}]\n{c['text']}" for c in chunks]
    context = "\n\n".join(context_parts)
    return RAG_USER_TEMPLATE.format(context=context, question=question)

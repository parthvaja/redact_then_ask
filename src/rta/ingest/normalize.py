"""Text normalisation for the ingest pipeline.

Applied **before** any character offsets are recorded so that all
downstream stages see a clean, canonical string.

Steps:
1. Unicode NFC normalisation.
2. Unify line endings (``\\r\\n`` → ``\\n``, then stray ``\\r`` → ``\\n``).
3. Collapse runs of horizontal whitespace (spaces / tabs) into a single
   space, but preserve newlines.
"""

from __future__ import annotations

import re
import unicodedata

# Pre-compiled patterns (module-level for reuse).
_CRLF = re.compile(r"\r\n")
_CR = re.compile(r"\r")
_HSPACE_RUN = re.compile(r"[^\S\n]+")  # one or more whitespace chars *except* newline


def normalize_text(raw: str) -> str:
    """Return a normalised copy of *raw*.

    The function is pure — it has no side effects and does not depend on
    any external state.
    """
    text = unicodedata.normalize("NFC", raw)
    text = _CRLF.sub("\n", text)
    text = _CR.sub("\n", text)
    text = _HSPACE_RUN.sub(" ", text)
    return text

"""Redact-Then-Ask: privacy-first document pipeline.

Detect and pseudonymize PII, then chat with the redacted text through a
local LLM.  See ``docs/PROJECT_DOC.md`` for the full technical spec.

Top-level re-exports for convenience::

    from rta import Span, Document, RedactionResult, GoldDoc
    from rta import get_config
"""

from __future__ import annotations

from rta.config import get_config
from rta.schema import Document, GoldDoc, RedactionResult, Span

__all__ = [
    "Document",
    "GoldDoc",
    "RedactionResult",
    "Span",
    "get_config",
]

__version__ = "0.1.0"

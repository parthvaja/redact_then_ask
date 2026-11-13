"""ai4privacy/pii-masking-300k dataset loader.

Downloads from Hugging Face Datasets, filters to English, converts to
``GoldDoc`` JSONL.  Implemented in Phase 2.
"""

from __future__ import annotations


def fetch() -> None:
    """Download the ai4privacy dataset from HF.  Phase 2."""
    raise NotImplementedError("rta.data.ai4privacy.fetch — Phase 2")


def convert() -> None:
    """Convert raw ai4privacy records to GoldDoc JSONL.  Phase 2."""
    raise NotImplementedError("rta.data.ai4privacy.convert — Phase 2")

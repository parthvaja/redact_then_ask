"""Text Anonymization Benchmark (TAB) loader.

Clones or downloads from NorskRegnesentral/text-anonymization-benchmark,
converts to ``GoldDoc`` JSONL.  Implemented in Phase 2.
"""

from __future__ import annotations


def fetch() -> None:
    """Download TAB dataset.  Phase 2."""
    raise NotImplementedError("rta.data.tab.fetch — Phase 2")


def convert() -> None:
    """Convert TAB annotations to GoldDoc JSONL.  Phase 2."""
    raise NotImplementedError("rta.data.tab.convert — Phase 2")

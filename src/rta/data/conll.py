"""CoNLL-2003 loader (PERSON/ORG/LOC sanity check).

Implemented in Phase 2.
"""

from __future__ import annotations


def fetch() -> None:
    """Download CoNLL-2003 from HF.  Phase 2."""
    raise NotImplementedError("rta.data.conll.fetch — Phase 2")


def convert() -> None:
    """Convert CoNLL-2003 to GoldDoc JSONL.  Phase 2."""
    raise NotImplementedError("rta.data.conll.convert — Phase 2")

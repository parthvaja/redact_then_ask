"""Label Studio annotation round-trip.

Exports pipeline pre-annotations to Label Studio JSON format and
imports corrected annotations back as ``GoldDoc`` JSONL.
Implemented in Phase 9.
"""

from __future__ import annotations


def export() -> None:
    """Export pipeline pre-annotations to Label Studio JSON.  Phase 9."""
    raise NotImplementedError("rta.data.labelstudio.export — Phase 9")


def import_annotations() -> None:
    """Import corrected annotations from Label Studio.  Phase 9."""
    raise NotImplementedError("rta.data.labelstudio.import_annotations — Phase 9")

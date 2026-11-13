"""Evaluation harness — precision, recall and F1 per PII type.

Implemented in Phase 9.
"""

from __future__ import annotations


def run_evaluation(
    dataset: str,
    split: str = "test",
    compare: str | None = None,
    system_name: str = "ours",
) -> None:
    """Run detection evaluation and write reports.

    Args:
        dataset:     Dataset name: ``"ai4privacy"``, ``"tab"``, or
                     ``"handlabel"``.
        split:       Dataset split: ``"train"``, ``"test"``, or ``"val"``.
        compare:     Optional baseline to compare against: ``"presidio"``.
        system_name: Label for our system in the output tables.

    Raises:
        NotImplementedError: Until Phase 9 is complete.
    """
    raise NotImplementedError("rta.eval.run_evaluation — implemented in Phase 9")

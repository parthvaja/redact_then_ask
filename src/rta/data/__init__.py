"""Dataset loaders and converters.

Sub-modules:
  - ``ai4privacy`` — ai4privacy/pii-masking-300k (HF)
  - ``tab``        — Text Anonymization Benchmark
  - ``conll``      — CoNLL-2003 (sanity check)
  - ``labelstudio`` — Label Studio annotation round-trip

All loaders convert to ``GoldDoc`` / ``Span`` using ``rta.taxonomy``
mappings.  Raw dataset labels never escape this package.
"""

from __future__ import annotations

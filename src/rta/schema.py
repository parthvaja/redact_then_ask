"""Central data model for the Redact-Then-Ask pipeline.

All stages communicate through the types defined here.  Offsets are
character offsets into ``Document.text``, end-exclusive.

Invariant (checked at construction)::

    span.text == document.text[span.start : span.end]

Records are dataclasses with ``to_dict`` / ``from_dict`` helpers for
JSONL serialisation.  ``Span`` is frozen (hash-able, usable in sets).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    """A single detected PII mention.

    Args:
        start:     Inclusive character offset into ``Document.text``.
        end:       Exclusive character offset into ``Document.text``.
        label:     Taxonomy label (see ``rta.taxonomy.LABELS``).
        text:      The literal substring ``Document.text[start:end]``.
        score:     Confidence in [0, 1].
        detector:  Source tag: ``"regex:<rule_id>"``, ``"ner:<model>"``,
                   ``"rule:<pack>"``, ``"gold"``, or ``"presidio"``.
        validated: ``True`` when a validator (Luhn, phonenumbers, …)
                   confirmed the match.
    """

    start: int
    end: int
    label: str
    text: str
    score: float
    detector: str
    validated: bool = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"Span.start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"Span.end ({self.end}) must be > Span.start ({self.start})")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"Span.score must be in [0, 1], got {self.score}")
        if not self.label:
            raise ValueError("Span.label must not be empty")
        if not self.detector:
            raise ValueError("Span.detector must not be empty")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Span:
        """Reconstruct a ``Span`` from ``to_dict()`` output."""
        return cls(
            start=int(d["start"]),
            end=int(d["end"]),
            label=str(d["label"]),
            text=str(d["text"]),
            score=float(d["score"]),
            detector=str(d["detector"]),
            validated=bool(d.get("validated", False)),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def length(self) -> int:
        """Number of characters covered."""
        return self.end - self.start

    def overlaps(self, other: Span) -> bool:
        """Return True if this span and *other* share at least one character."""
        return self.start < other.end and other.start < self.end

    def contains(self, other: Span) -> bool:
        """Return True if this span fully contains *other*."""
        return self.start <= other.start and self.end >= other.end

    def check_text(self, source: str) -> bool:
        """Return True if ``self.text`` matches ``source[start:end]``."""
        return self.text == source[self.start : self.end]


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A document after ingestion.

    ``text`` is the normalised, NFC-encoded plain text.  It is never
    mutated after construction.  All spans index into ``text``.

    Args:
        doc_id:      Stable identifier (usually a hash of the source file).
        text:        Normalised plain text.
        source_path: Original file path, or ``None`` for in-memory documents.
        pages:       Per-page character ranges ``(start, end)``; empty list
                     for single-page or plain-text documents.
        meta:        Mutable metadata bag.  Keys added by stages:
                     ``language`` (ISO code), ``lang_conf`` (float),
                     ``doc_type`` (str), ``type_probs`` (dict[str, float]),
                     ``n_chars`` (int), ``unsupported_language`` (bool).
    """

    doc_id: str
    text: str
    source_path: str | None
    pages: list[tuple[int, int]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Populate n_chars if not already set
        self.meta.setdefault("n_chars", len(self.text))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "source_path": self.source_path,
            "pages": self.pages,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Document:
        return cls(
            doc_id=str(d["doc_id"]),
            text=str(d["text"]),
            source_path=d.get("source_path"),
            pages=[tuple(p) for p in d.get("pages", [])],  # type: ignore[misc]
            meta=dict(d.get("meta", {})),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def page_of(self, offset: int) -> int | None:
        """Return the 0-based page index for a character offset, or None."""
        for i, (start, end) in enumerate(self.pages):
            if start <= offset < end:
                return i
        return None


# ---------------------------------------------------------------------------
# RedactionResult
# ---------------------------------------------------------------------------


@dataclass
class RedactionResult:
    """The output of the redaction stage.

    Args:
        text:       Redacted text with placeholders substituted in.
        mapping:    ``{placeholder: canonical_original}``; kept in process
                    memory and **never** placed in an LLM prompt.
        clusters:   Mention groups per entity; each inner list is a list of
                    ``Span`` objects from the same resolved entity.
        offset_map: Alignment from original to redacted character positions:
                    ``(orig_start, orig_end, red_start, red_end)`` tuples
                    covering every substituted region, sorted by orig_start.
    """

    text: str
    mapping: dict[str, str] = field(default_factory=dict)
    clusters: list[list[Span]] = field(default_factory=list)
    offset_map: list[tuple[int, int, int, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "mapping": self.mapping,
            "clusters": [[s.to_dict() for s in cluster] for cluster in self.clusters],
            "offset_map": self.offset_map,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RedactionResult:
        return cls(
            text=str(d["text"]),
            mapping=dict(d.get("mapping", {})),
            clusters=[[Span.from_dict(s) for s in cluster] for cluster in d.get("clusters", [])],
            offset_map=[tuple(t) for t in d.get("offset_map", [])],  # type: ignore[misc]
        )


# ---------------------------------------------------------------------------
# GoldDoc
# ---------------------------------------------------------------------------


@dataclass
class GoldDoc:
    """A document with ground-truth PII annotations.

    Args:
        doc_id:   Stable identifier.
        text:     The annotated text (identical to the ingested text for
                  the same document).
        spans:    Gold spans with ``detector="gold"``.
        doc_type: Ground-truth document type, or ``None`` if unknown.
        language: ISO language code, or ``None`` if unknown.
        source:   Dataset provenance: ``"ai4privacy"``, ``"tab"``,
                  ``"conll"``, or ``"handlabel"``.
    """

    doc_id: str
    text: str
    spans: list[Span]
    doc_type: str | None = None
    language: str | None = None
    source: str = "handlabel"

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "spans": [s.to_dict() for s in self.spans],
            "doc_type": self.doc_type,
            "language": self.language,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoldDoc:
        return cls(
            doc_id=str(d["doc_id"]),
            text=str(d["text"]),
            spans=[Span.from_dict(s) for s in d.get("spans", [])],
            doc_type=d.get("doc_type"),
            language=d.get("language"),
            source=str(d.get("source", "handlabel")),
        )


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def write_jsonl(records: list[Any], path: Path) -> None:
    """Write a list of records (each with ``.to_dict()``) to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def read_spans_jsonl(path: Path) -> list[Span]:
    """Read a JSONL file of serialised ``Span`` dicts."""
    with path.open(encoding="utf-8") as fh:
        return [Span.from_dict(json.loads(line)) for line in fh if line.strip()]


def read_gold_docs_jsonl(path: Path) -> list[GoldDoc]:
    """Read a JSONL file of serialised ``GoldDoc`` dicts."""
    with path.open(encoding="utf-8") as fh:
        return [GoldDoc.from_dict(json.loads(line)) for line in fh if line.strip()]

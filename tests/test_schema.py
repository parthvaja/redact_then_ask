"""Tests for rta.schema — Span, Document, RedactionResult, GoldDoc.

Gate criteria for Phase 0:
- Span invariant is enforced at construction.
- to_dict / from_dict round-trips are lossless.
- Document and GoldDoc serialise and deserialise correctly.
- write_jsonl / read_spans_jsonl JSONL helpers work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rta.schema import (
    Document,
    GoldDoc,
    RedactionResult,
    Span,
    read_gold_docs_jsonl,
    read_spans_jsonl,
    write_jsonl,
)

# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


class TestSpan:
    def test_basic_construction(self) -> None:
        s = Span(start=0, end=5, label="PERSON", text="Alice", score=0.9, detector="ner:spacy")
        assert s.start == 0
        assert s.end == 5
        assert s.label == "PERSON"
        assert s.text == "Alice"
        assert s.score == 0.9
        assert s.detector == "ner:spacy"
        assert s.validated is False

    def test_validated_flag(self) -> None:
        s = Span(
            start=0,
            end=11,
            label="EMAIL",
            text="a@b.com    ",
            score=0.99,
            detector="regex:email_basic",
            validated=True,
        )
        assert s.validated is True

    def test_start_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="start"):
            Span(start=-1, end=5, label="PERSON", text="Alice", score=0.9, detector="ner:spacy")

    def test_end_le_start_raises(self) -> None:
        with pytest.raises(ValueError, match="end"):
            Span(start=5, end=5, label="PERSON", text="", score=0.9, detector="ner:spacy")

    def test_end_lt_start_raises(self) -> None:
        with pytest.raises(ValueError, match="end"):
            Span(start=5, end=3, label="PERSON", text="xy", score=0.9, detector="ner:spacy")

    def test_score_out_of_range_high(self) -> None:
        with pytest.raises(ValueError, match="score"):
            Span(start=0, end=5, label="PERSON", text="Alice", score=1.1, detector="ner:spacy")

    def test_score_out_of_range_low(self) -> None:
        with pytest.raises(ValueError, match="score"):
            Span(start=0, end=5, label="PERSON", text="Alice", score=-0.1, detector="ner:spacy")

    def test_empty_label_raises(self) -> None:
        with pytest.raises(ValueError, match="label"):
            Span(start=0, end=5, label="", text="Alice", score=0.9, detector="ner:spacy")

    def test_empty_detector_raises(self) -> None:
        with pytest.raises(ValueError, match="detector"):
            Span(start=0, end=5, label="PERSON", text="Alice", score=0.9, detector="")

    def test_to_dict_from_dict_roundtrip(self) -> None:
        s = Span(
            start=3,
            end=12,
            label="EMAIL",
            text="a@test.com",
            score=0.95,
            detector="regex:email_basic",
            validated=True,
        )
        d = s.to_dict()
        assert isinstance(d, dict)
        s2 = Span.from_dict(d)
        assert s == s2

    def test_to_dict_keys(self) -> None:
        s = Span(start=0, end=5, label="PERSON", text="Alice", score=0.9, detector="ner:spacy")
        d = s.to_dict()
        expected_keys = {"start", "end", "label", "text", "score", "detector", "validated"}
        assert set(d.keys()) == expected_keys

    def test_from_dict_coerces_types(self) -> None:
        d = {
            "start": "0",
            "end": "5",
            "label": "PERSON",
            "text": "Alice",
            "score": "0.9",
            "detector": "ner:spacy",
        }
        s = Span.from_dict(d)
        assert isinstance(s.start, int)
        assert isinstance(s.score, float)
        assert s.validated is False  # default

    def test_length_property(self) -> None:
        s = Span(start=3, end=10, label="PERSON", text="Aliceba", score=0.9, detector="ner:spacy")
        assert s.length == 7

    def test_overlaps_true(self) -> None:
        a = Span(start=0, end=10, label="PERSON", text="x" * 10, score=0.9, detector="ner:spacy")
        b = Span(start=5, end=15, label="PERSON", text="x" * 10, score=0.9, detector="ner:spacy")
        assert a.overlaps(b)
        assert b.overlaps(a)

    def test_overlaps_false_adjacent(self) -> None:
        a = Span(start=0, end=5, label="PERSON", text="Alice", score=0.9, detector="ner:spacy")
        b = Span(start=5, end=10, label="EMAIL", text="b@c.de", score=0.9, detector="regex:email")
        assert not a.overlaps(b)
        assert not b.overlaps(a)

    def test_contains(self) -> None:
        outer = Span(
            start=0, end=20, label="PERSON", text="x" * 20, score=0.9, detector="ner:spacy"
        )
        inner = Span(start=5, end=10, label="PERSON", text="x" * 5, score=0.9, detector="ner:spacy")
        assert outer.contains(inner)
        assert not inner.contains(outer)

    def test_check_text_valid(self) -> None:
        source = "Hello Alice!"
        s = Span(start=6, end=11, label="PERSON", text="Alice", score=0.9, detector="ner:spacy")
        assert s.check_text(source)

    def test_check_text_invalid(self) -> None:
        source = "Hello Alice!"
        s = Span(start=0, end=5, label="PERSON", text="World", score=0.9, detector="ner:spacy")
        assert not s.check_text(source)

    def test_span_is_frozen(self) -> None:
        s = Span(start=0, end=5, label="PERSON", text="Alice", score=0.9, detector="ner:spacy")
        with pytest.raises((AttributeError, TypeError)):
            s.start = 1  # type: ignore[misc]

    def test_span_hashable(self) -> None:
        s = Span(start=0, end=5, label="PERSON", text="Alice", score=0.9, detector="ner:spacy")
        # Should be usable in a set
        s_set = {s}
        assert s in s_set


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class TestDocument:
    def test_basic_construction(self) -> None:
        doc = Document(doc_id="d1", text="Hello world", source_path=None)
        assert doc.doc_id == "d1"
        assert doc.text == "Hello world"
        assert doc.meta["n_chars"] == 11

    def test_n_chars_set_automatically(self) -> None:
        doc = Document(doc_id="d2", text="abc", source_path="/tmp/test.txt")
        assert doc.meta["n_chars"] == 3

    def test_n_chars_not_overridden_if_present(self) -> None:
        doc = Document(
            doc_id="d3",
            text="abc",
            source_path=None,
            meta={"n_chars": 999},  # caller-provided value
        )
        # __post_init__ uses setdefault, so caller's value wins
        assert doc.meta["n_chars"] == 999

    def test_to_dict_from_dict_roundtrip(self) -> None:
        doc = Document(
            doc_id="d4",
            text="Alice works at Acme.",
            source_path="/tmp/sample.txt",
            pages=[(0, 20)],
            meta={"language": "en", "lang_conf": 0.99},
        )
        d = doc.to_dict()
        doc2 = Document.from_dict(d)
        assert doc2.doc_id == doc.doc_id
        assert doc2.text == doc.text
        assert doc2.source_path == doc.source_path
        assert doc2.pages == doc.pages
        assert doc2.meta == doc.meta

    def test_page_of(self) -> None:
        doc = Document(doc_id="d5", text="A" * 100, source_path=None, pages=[(0, 50), (50, 100)])
        assert doc.page_of(0) == 0
        assert doc.page_of(49) == 0
        assert doc.page_of(50) == 1
        assert doc.page_of(99) == 1
        assert doc.page_of(100) is None


# ---------------------------------------------------------------------------
# RedactionResult
# ---------------------------------------------------------------------------


class TestRedactionResult:
    def test_basic_construction(self) -> None:
        s = Span(start=0, end=5, label="PERSON", text="Alice", score=0.9, detector="ner:spacy")
        rr = RedactionResult(
            text="[PERSON_1] works here.",
            mapping={"[PERSON_1]": "Alice"},
            clusters=[[s]],
            offset_map=[(0, 5, 0, 10)],
        )
        assert "[PERSON_1]" in rr.text
        assert rr.mapping["[PERSON_1]"] == "Alice"

    def test_to_dict_from_dict_roundtrip(self) -> None:
        s = Span(
            start=6,
            end=11,
            label="EMAIL",
            text="a@b.c",
            score=0.95,
            detector="regex:email_basic",
            validated=True,
        )
        rr = RedactionResult(
            text="Send [EMAIL_1] an email.",
            mapping={"[EMAIL_1]": "a@b.c"},
            clusters=[[s]],
            offset_map=[(6, 11, 5, 14)],
        )
        d = rr.to_dict()
        rr2 = RedactionResult.from_dict(d)
        assert rr2.text == rr.text
        assert rr2.mapping == rr.mapping
        assert len(rr2.clusters) == 1
        assert rr2.clusters[0][0] == s


# ---------------------------------------------------------------------------
# GoldDoc
# ---------------------------------------------------------------------------


class TestGoldDoc:
    def test_basic_construction(self) -> None:
        s = Span(start=0, end=5, label="PERSON", text="Alice", score=1.0, detector="gold")
        gd = GoldDoc(doc_id="g1", text="Alice Smith", spans=[s], source="handlabel")
        assert gd.doc_id == "g1"
        assert len(gd.spans) == 1

    def test_to_dict_from_dict_roundtrip(self) -> None:
        s = Span(start=0, end=5, label="PERSON", text="Alice", score=1.0, detector="gold")
        gd = GoldDoc(
            doc_id="g2",
            text="Alice Smith",
            spans=[s],
            doc_type="resume",
            language="en",
            source="ai4privacy",
        )
        d = gd.to_dict()
        gd2 = GoldDoc.from_dict(d)
        assert gd2.doc_id == gd.doc_id
        assert gd2.doc_type == gd.doc_type
        assert gd2.language == gd.language
        assert gd2.source == gd.source
        assert gd2.spans[0] == s


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


class TestJsonlHelpers:
    def test_write_and_read_spans(self, tmp_path: Path) -> None:
        spans = [
            Span(start=0, end=5, label="PERSON", text="Alice", score=0.9, detector="ner:spacy"),
            Span(
                start=10,
                end=15,
                label="EMAIL",
                text="a@b.c",
                score=0.95,
                detector="regex:email_basic",
                validated=True,
            ),
        ]
        p = tmp_path / "spans.jsonl"
        write_jsonl(spans, p)
        loaded = read_spans_jsonl(p)
        assert len(loaded) == 2
        assert loaded[0] == spans[0]
        assert loaded[1] == spans[1]

    def test_write_and_read_gold_docs(self, tmp_path: Path) -> None:
        s = Span(start=0, end=5, label="PERSON", text="Alice", score=1.0, detector="gold")
        docs = [GoldDoc(doc_id="g1", text="Alice Smith", spans=[s])]
        p = tmp_path / "gold.jsonl"
        write_jsonl(docs, p)
        loaded = read_gold_docs_jsonl(p)
        assert len(loaded) == 1
        assert loaded[0].doc_id == "g1"
        assert loaded[0].spans[0] == s

    def test_jsonl_one_record_per_line(self, tmp_path: Path) -> None:
        spans = [
            Span(start=0, end=5, label="PERSON", text="Alice", score=0.9, detector="ner:spacy"),
            Span(
                start=6,
                end=11,
                label="EMAIL",
                text="a@b.c",
                score=0.95,
                detector="regex:email_basic",
            ),
        ]
        p = tmp_path / "test.jsonl"
        write_jsonl(spans, p)
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2
        # Each line must be valid JSON
        for line in lines:
            obj = json.loads(line)
            assert isinstance(obj, dict)

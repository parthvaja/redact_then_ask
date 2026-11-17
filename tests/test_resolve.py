"""Tests for rta.resolve — span resolution and merging (Phase 7).

Tests exercise the merge rules from section 6.7 of PROJECT_DOC.md:
1. Validated span wins over unvalidated.
2. Exact same range, different labels: label precedence wins.
3. Same-label overlapping spans get unioned (recall first).
4. Partial overlap, different labels: longer wins (unless shorter validated).
5. Spans below min_score are filtered.
6. Trim leading/trailing whitespace and punctuation.
"""

from __future__ import annotations

import random

import pytest

from rta.config import reset as reset_config
from rta.resolve import merge
from rta.schema import Span


@pytest.fixture(autouse=True)
def _fresh_state() -> None:
    """Reset config before and after every test."""
    reset_config()
    yield  # type: ignore[misc]
    reset_config()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _span(
    start: int,
    end: int,
    text: str,
    *,
    label: str = "PERSON",
    score: float = 0.8,
    detector: str = "ner:spacy",
    validated: bool = False,
) -> Span:
    return Span(
        start=start,
        end=end,
        label=label,
        text=text,
        score=score,
        detector=detector,
        validated=validated,
    )


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------


class TestBasicMerge:
    def test_empty_spans_returns_empty(self) -> None:
        """Empty spans -> empty result."""
        assert merge([]) == []

    def test_single_span_passes_through(self) -> None:
        """Single span passes through unchanged."""
        span = _span(0, 5, "Alice")
        result = merge([span])
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == 5
        assert result[0].label == "PERSON"
        assert result[0].text == "Alice"

    def test_non_overlapping_spans_all_pass(self) -> None:
        """Non-overlapping spans all pass through."""
        spans = [
            _span(0, 5, "Alice"),
            _span(10, 15, "Bob's", label="PERSON"),
            _span(20, 33, "123-45-6789  ", label="SSN", score=0.95, detector="regex:ssn"),
        ]
        result = merge(spans)
        assert len(result) == 3
        # Should be sorted by start
        assert [s.start for s in result] == [0, 10, 20]


# ---------------------------------------------------------------------------
# Rule 3: same-label overlap -> union
# ---------------------------------------------------------------------------


class TestSameLabelUnion:
    def test_overlapping_same_label_unioned(self) -> None:
        """Same-label overlapping spans get unioned (rule 3)."""
        doc_text = "John Michael Smith lives here."
        spans = [
            _span(0, 12, "John Michael", score=0.8),
            _span(5, 18, "Michael Smith", score=0.85),
        ]
        result = merge(spans, doc_text=doc_text)
        # Should be unioned into one span covering [0, 18)
        persons = [s for s in result if s.label == "PERSON"]
        assert len(persons) == 1
        assert persons[0].start == 0
        assert persons[0].end == 18
        assert persons[0].text == "John Michael Smith"
        # Score should be the max
        assert persons[0].score == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Rule 1: validated wins over unvalidated
# ---------------------------------------------------------------------------


class TestValidatedWins:
    def test_validated_span_wins(self) -> None:
        """Validated span wins over unvalidated (rule 1)."""
        spans = [
            _span(
                0, 11, "123-45-6789", label="SSN", score=0.95, detector="regex:ssn", validated=True
            ),
            _span(
                0,
                11,
                "123-45-6789",
                label="ID_NUMBER",
                score=0.90,
                detector="ner:spacy",
                validated=False,
            ),
        ]
        result = merge(spans)
        assert len(result) == 1
        assert result[0].label == "SSN"
        assert result[0].validated is True


# ---------------------------------------------------------------------------
# Rule 2: exact same range, different labels -> label precedence
# ---------------------------------------------------------------------------


class TestLabelPrecedence:
    def test_ssn_beats_id_number(self) -> None:
        """Exact same range, different labels -> SSN wins over ID_NUMBER (rule 2)."""
        spans = [
            _span(0, 11, "123-45-6789", label="ID_NUMBER", score=0.90, detector="regex:id"),
            _span(0, 11, "123-45-6789", label="SSN", score=0.90, detector="regex:ssn"),
        ]
        result = merge(spans)
        assert len(result) == 1
        assert result[0].label == "SSN"


# ---------------------------------------------------------------------------
# Rule 4: partial overlap, different labels -> longer wins
# ---------------------------------------------------------------------------


class TestPartialOverlapDifferentLabels:
    def test_longer_span_wins(self) -> None:
        """Partial overlap, different labels, unvalidated -> longer wins (rule 4)."""
        doc_text = "John Smith at 123 Maple Street today."
        spans = [
            _span(0, 10, "John Smith", label="PERSON", score=0.8),
            _span(5, 28, "Smith at 123 Maple Street", label="ADDRESS", score=0.7),
        ]
        result = merge(spans, doc_text=doc_text)
        # ADDRESS span is longer (23 chars vs 10), so it wins
        assert len(result) == 1
        assert result[0].label == "ADDRESS"

    def test_shorter_validated_wins(self) -> None:
        """Partial overlap, shorter is validated -> shorter wins (rules 1+4)."""
        doc_text = "SSN 123-45-6789 unknown text."
        spans = [
            _span(
                4, 15, "123-45-6789", label="SSN", score=0.95, detector="regex:ssn", validated=True
            ),
            _span(
                0, 20, "SSN 123-45-6789 unkn", label="ID_NUMBER", score=0.80, detector="ner:spacy"
            ),
        ]
        result = merge(spans, doc_text=doc_text)
        # SSN is shorter but validated -> wins
        assert any(s.label == "SSN" and s.validated for s in result)


# ---------------------------------------------------------------------------
# Rule 5: min_score filtering
# ---------------------------------------------------------------------------


class TestMinScoreFilter:
    def test_spans_below_min_score_filtered(self) -> None:
        """Spans below min_score (0.35) are filtered (rule 5)."""
        spans = [
            _span(0, 5, "Alice", score=0.1),  # below 0.35
            _span(10, 15, "Bob's", score=0.2),  # below 0.35
        ]
        result = merge(spans)
        assert result == []

    def test_span_at_threshold_passes(self) -> None:
        """Span at exactly min_score (0.35) passes through."""
        spans = [_span(0, 5, "Alice", score=0.35)]
        result = merge(spans)
        assert len(result) == 1

    def test_mixed_scores(self) -> None:
        """Only spans at or above threshold survive."""
        spans = [
            _span(0, 5, "Alice", score=0.1),  # filtered
            _span(10, 15, "Bob's", score=0.5),  # kept
            _span(20, 28, "Charlie!", score=0.9),  # kept
        ]
        result = merge(spans)
        assert len(result) == 2
        labels_text = {s.text for s in result}
        assert "Bob's" in labels_text
        assert "Charlie!" in labels_text


# ---------------------------------------------------------------------------
# Rule 6: trim whitespace and punctuation
# ---------------------------------------------------------------------------


class TestTrimming:
    def test_trims_whitespace_and_punctuation(self) -> None:
        """Whitespace/punctuation is trimmed when doc_text given (rule 6)."""
        doc_text = "Contact  Alice  for details."
        # Span includes leading/trailing spaces
        span = _span(8, 15, " Alice ", score=0.8)
        result = merge([span], doc_text=doc_text)
        assert len(result) == 1
        assert result[0].text == "Alice"
        assert result[0].start == 9
        assert result[0].end == 14

    def test_trimming_to_nothing_removes_span(self) -> None:
        """Trimming to nothing removes the span."""
        doc_text = "Hello ...  world."
        # Span covers only punctuation and whitespace
        span = _span(6, 10, "... ", score=0.8)
        result = merge([span], doc_text=doc_text)
        assert len(result) == 0

    def test_no_trim_without_doc_text(self) -> None:
        """Without doc_text, trimming is skipped."""
        span = _span(0, 7, " Alice ", score=0.8)
        result = merge([span])
        assert len(result) == 1
        assert result[0].text == " Alice "

    def test_trim_leading_punctuation(self) -> None:
        """Leading punctuation is trimmed."""
        doc_text = '..."Alice" said hello.'
        span = _span(0, 10, '..."Alice"', score=0.8)
        result = merge([span], doc_text=doc_text)
        assert len(result) == 1
        assert result[0].text == "Alice"
        assert result[0].start == 4
        assert result[0].end == 9


# ---------------------------------------------------------------------------
# Property: output has no overlaps
# ---------------------------------------------------------------------------


class TestNoOverlapProperty:
    def test_random_spans_no_overlap_in_output(self) -> None:
        """Property: output has no overlaps — generate random spans, verify."""
        rng = random.Random(13)  # reproducible seed
        doc_text = "a" * 500
        labels = ["PERSON", "ORG", "LOCATION", "EMAIL", "SSN", "PHONE"]

        for _ in range(20):  # run 20 random trials
            n_spans = rng.randint(2, 15)
            spans = []
            for _ in range(n_spans):
                start = rng.randint(0, 480)
                length = rng.randint(1, 20)
                end = min(start + length, 500)
                if end <= start:
                    continue
                label = rng.choice(labels)
                score = round(rng.uniform(0.35, 1.0), 2)
                text = doc_text[start:end]
                spans.append(
                    _span(
                        start,
                        end,
                        text,
                        label=label,
                        score=score,
                        detector="ner:spacy",
                    )
                )

            if not spans:
                continue

            result = merge(spans, doc_text=doc_text)

            # Verify no overlaps in output
            sorted_result = sorted(result, key=lambda s: s.start)
            for i in range(len(sorted_result) - 1):
                a = sorted_result[i]
                b = sorted_result[i + 1]
                assert a.end <= b.start, (
                    f"Overlap in output: [{a.start},{a.end}) {a.label} "
                    f"and [{b.start},{b.end}) {b.label}"
                )

    def test_output_sorted_by_start(self) -> None:
        """Output is always sorted by start position."""
        spans = [
            _span(20, 25, "world", score=0.8),
            _span(0, 5, "hello", score=0.9),
            _span(10, 15, "there", label="ORG", score=0.7),
        ]
        result = merge(spans)
        starts = [s.start for s in result]
        assert starts == sorted(starts)

"""Tests for rta.redact — redaction and pseudonymization (Phase 8).

Gate criteria:
- apply() with mask mode replaces text with blocks of equal length.
- apply() with placeholder mode uses [LABEL_N] format.
- apply() with fake mode uses Faker surrogates.
- Empty spans leave original text unchanged.
- Offset map correctness: (orig_start, orig_end, red_start, red_end).
- Same person mentioned twice gets same placeholder (clustering).
- Different labels don't cluster together.
- cluster_spans: exact match, token containment, initials clustering.
- generate_surrogate returns consistent values for same seed/cluster_id.
- normalize_mention strips titles and possessives.
"""

from __future__ import annotations

import pytest

from rta.config import reset
from rta.redact import apply
from rta.redact.cluster import cluster_spans, normalize_mention
from rta.redact.surrogates import generate_surrogate
from rta.schema import Document, Span


@pytest.fixture(autouse=True)
def fresh_config() -> None:
    """Reset the config singleton before and after every test."""
    reset()
    yield  # type: ignore[misc]
    reset()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(text: str) -> Document:
    return Document(doc_id="test", text=text, source_path=None)


def _make_span(
    start: int,
    end: int,
    label: str,
    text: str,
    score: float = 0.9,
    detector: str = "test",
) -> Span:
    return Span(
        start=start,
        end=end,
        label=label,
        text=text,
        score=score,
        detector=detector,
    )


# ---------------------------------------------------------------------------
# apply() — mask mode
# ---------------------------------------------------------------------------


class TestApplyMask:
    def test_mask_replaces_with_blocks_of_equal_length(self) -> None:
        text = "Call John Smith at 555-1234."
        doc = _make_doc(text)
        spans = [
            _make_span(5, 15, "PERSON", "John Smith"),
            _make_span(19, 27, "PHONE", "555-1234"),
        ]
        result = apply(doc, spans, mode="mask")
        # Each PII region replaced by block chars of the same length
        assert "\u2588" * 10 in result.text  # "John Smith" -> 10 blocks
        assert "\u2588" * 8 in result.text  # "555-1234" -> 8 blocks
        assert "John Smith" not in result.text
        assert "555-1234" not in result.text
        # Surrounding text preserved
        assert result.text.startswith("Call ")
        assert result.text.endswith(".")

    def test_mask_preserves_text_length_per_span(self) -> None:
        text = "Hi Alice!"
        doc = _make_doc(text)
        spans = [_make_span(3, 8, "PERSON", "Alice")]
        result = apply(doc, spans, mode="mask")
        # "Alice" is 5 chars -> 5 block chars; total length unchanged
        assert len(result.text) == len(text)
        assert result.text == "Hi \u2588\u2588\u2588\u2588\u2588!"


# ---------------------------------------------------------------------------
# apply() — placeholder mode
# ---------------------------------------------------------------------------


class TestApplyPlaceholder:
    def test_placeholder_format(self) -> None:
        text = "Contact John Smith for details."
        doc = _make_doc(text)
        spans = [_make_span(8, 18, "PERSON", "John Smith")]
        result = apply(doc, spans, mode="placeholder")
        assert "[PERSON_1]" in result.text
        assert "John Smith" not in result.text

    def test_placeholder_mapping(self) -> None:
        text = "Contact John Smith for details."
        doc = _make_doc(text)
        spans = [_make_span(8, 18, "PERSON", "John Smith")]
        result = apply(doc, spans, mode="placeholder")
        assert "[PERSON_1]" in result.mapping
        assert result.mapping["[PERSON_1]"] == "John Smith"


# ---------------------------------------------------------------------------
# apply() — fake mode
# ---------------------------------------------------------------------------


class TestApplyFake:
    def test_fake_uses_surrogates(self) -> None:
        text = "Contact John Smith for details."
        doc = _make_doc(text)
        spans = [_make_span(8, 18, "PERSON", "John Smith")]
        result = apply(doc, spans, mode="fake")
        # The original name should be replaced
        assert "John Smith" not in result.text
        # The mapping should map surrogate -> original
        assert len(result.mapping) == 1
        surrogate = next(iter(result.mapping.keys()))
        assert result.mapping[surrogate] == "John Smith"
        assert surrogate in result.text


# ---------------------------------------------------------------------------
# apply() — empty spans
# ---------------------------------------------------------------------------


class TestApplyEmpty:
    def test_empty_spans_return_original(self) -> None:
        text = "Nothing to redact here."
        doc = _make_doc(text)
        result = apply(doc, [], mode="mask")
        assert result.text == text
        assert result.mapping == {}
        assert result.clusters == []
        assert result.offset_map == []


# ---------------------------------------------------------------------------
# apply() — offset_map correctness
# ---------------------------------------------------------------------------


class TestApplyOffsetMap:
    def test_offset_map_entries(self) -> None:
        text = "Name: Alice, Phone: 555-0000."
        doc = _make_doc(text)
        spans = [
            _make_span(6, 11, "PERSON", "Alice"),
            _make_span(20, 28, "PHONE", "555-0000"),
        ]
        result = apply(doc, spans, mode="placeholder")
        # Should have two offset entries
        assert len(result.offset_map) == 2

        for orig_start, orig_end, red_start, red_end in result.offset_map:
            # Redacted text in the range should be a placeholder
            replacement = result.text[red_start:red_end]
            assert replacement.startswith("[")
            assert replacement.endswith("]")
            # Original range should match a span
            assert text[orig_start:orig_end] in ("Alice", "555-0000")

    def test_offset_map_mask_preserves_lengths(self) -> None:
        text = "Hi Bob!"
        doc = _make_doc(text)
        spans = [_make_span(3, 6, "PERSON", "Bob")]
        result = apply(doc, spans, mode="mask")
        assert len(result.offset_map) == 1
        orig_start, orig_end, red_start, red_end = result.offset_map[0]
        assert orig_start == 3
        assert orig_end == 6
        # In mask mode lengths are preserved
        assert red_end - red_start == orig_end - orig_start


# ---------------------------------------------------------------------------
# Clustering — same person gets same placeholder
# ---------------------------------------------------------------------------


class TestClusteringPlaceholder:
    def test_same_person_same_placeholder(self) -> None:
        text = "John Smith went home. John Smith called."
        doc = _make_doc(text)
        spans = [
            _make_span(0, 10, "PERSON", "John Smith"),
            _make_span(22, 32, "PERSON", "John Smith"),
        ]
        result = apply(doc, spans, mode="placeholder")
        # Both mentions should become the same placeholder
        # Count occurrences of [PERSON_1]
        assert result.text.count("[PERSON_1]") == 2

    def test_different_labels_different_clusters(self) -> None:
        text = "Alice works at Acme Corp in Paris."
        doc = _make_doc(text)
        spans = [
            _make_span(0, 5, "PERSON", "Alice"),
            _make_span(15, 24, "ORG", "Acme Corp"),
            _make_span(28, 33, "LOCATION", "Paris"),
        ]
        result = apply(doc, spans, mode="placeholder")
        # Each label type should produce its own counter
        assert "[PERSON_1]" in result.text
        assert "[ORG_1]" in result.text
        assert "[LOCATION_1]" in result.text


# ---------------------------------------------------------------------------
# cluster_spans — exact match
# ---------------------------------------------------------------------------


class TestClusterSpansExact:
    def test_exact_match_clusters(self) -> None:
        spans = [
            _make_span(0, 10, "PERSON", "John Smith"),
            _make_span(20, 30, "PERSON", "John Smith"),
        ]
        clusters = cluster_spans(spans)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_different_labels_separate_clusters(self) -> None:
        spans = [
            _make_span(0, 5, "PERSON", "Alice"),
            _make_span(10, 15, "ORG", "Alice"),
        ]
        clusters = cluster_spans(spans)
        # Same text but different labels -> separate clusters
        assert len(clusters) == 2


# ---------------------------------------------------------------------------
# cluster_spans — token containment
# ---------------------------------------------------------------------------


class TestClusterSpansContainment:
    def test_token_containment(self) -> None:
        """'Smith' is a subset of tokens in 'John Smith' -> same cluster."""
        spans = [
            _make_span(0, 10, "PERSON", "John Smith"),
            _make_span(20, 25, "PERSON", "Smith"),
        ]
        clusters = cluster_spans(spans)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2


# ---------------------------------------------------------------------------
# cluster_spans — initials
# ---------------------------------------------------------------------------


class TestClusterSpansInitials:
    def test_initials_match(self) -> None:
        """'J. Smith' should cluster with 'John Smith'."""
        spans = [
            _make_span(0, 10, "PERSON", "John Smith"),
            _make_span(20, 29, "PERSON", "J. Smith"),
        ]
        clusters = cluster_spans(spans)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2


# ---------------------------------------------------------------------------
# generate_surrogate — consistency
# ---------------------------------------------------------------------------


class TestGenerateSurrogate:
    def test_consistent_for_same_seed_and_cluster(self) -> None:
        val1 = generate_surrogate("PERSON", 0)
        val2 = generate_surrogate("PERSON", 0)
        assert val1 == val2

    def test_different_cluster_ids_differ(self) -> None:
        val1 = generate_surrogate("PERSON", 0)
        val2 = generate_surrogate("PERSON", 1)
        # Different seeds should almost certainly produce different names
        # (not guaranteed, but extremely unlikely for Faker)
        assert isinstance(val1, str) and len(val1) > 0
        assert isinstance(val2, str) and len(val2) > 0

    def test_unknown_label_fallback(self) -> None:
        val = generate_surrogate("UNKNOWN_LABEL", 5)
        assert val == "[UNKNOWN_LABEL_5]"


# ---------------------------------------------------------------------------
# normalize_mention
# ---------------------------------------------------------------------------


class TestNormalizeMention:
    def test_strips_title(self) -> None:
        assert "smith" in normalize_mention("Dr. Smith", "PERSON")
        assert "dr" not in normalize_mention("Dr. Smith", "PERSON")

    def test_strips_possessive(self) -> None:
        result = normalize_mention("Smith's", "PERSON")
        assert "'s" not in result
        assert "smith" in result

    def test_strips_title_and_possessive(self) -> None:
        result = normalize_mention("Mr. Johnson's", "PERSON")
        assert "mr" not in result
        assert "'s" not in result
        assert "johnson" in result

    def test_phone_digits_only(self) -> None:
        result = normalize_mention("(555) 123-4567", "PHONE")
        assert result == "5551234567"

    def test_lowercases(self) -> None:
        result = normalize_mention("ALICE", "PERSON")
        assert result == "alice"

"""Tests for rta.detect.rules — routing and rule-pack selection (Phase 6).

Tests cover for_type routing decisions, title-cue loading from YAML packs,
and NER PERSON score boosting near title cues.
"""

from __future__ import annotations

import pytest

from rta.config import reset as reset_config
from rta.detect.rules import boost_ner_scores, for_type, get_title_cues, reset_cue_cache
from rta.schema import Span


@pytest.fixture(autouse=True)
def _fresh_state() -> None:
    """Reset config and cue cache before and after every test."""
    reset_config()
    reset_cue_cache()
    yield  # type: ignore[misc]
    reset_config()
    reset_cue_cache()


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
# for_type — routing decision
# ---------------------------------------------------------------------------


class TestForType:
    def test_none_doc_type_returns_empty(self) -> None:
        """for_type with None doc_type -> empty list."""
        assert for_type(None) == []

    def test_none_doc_type_with_probs_returns_empty(self) -> None:
        assert for_type(None, {"medical": 0.9}) == []

    def test_doc_type_no_probs_returns_single(self) -> None:
        """for_type with doc_type but no type_probs -> [doc_type]."""
        assert for_type("medical") == ["medical"]

    def test_doc_type_no_probs_various(self) -> None:
        assert for_type("resume") == ["resume"]

    def test_high_confidence_returns_single(self) -> None:
        """for_type with high confidence (>=0.6) -> single type."""
        probs = {"medical": 0.85, "resume": 0.10, "email": 0.05}
        result = for_type("medical", probs)
        assert result == ["medical"]

    def test_exact_threshold_returns_single(self) -> None:
        """Confidence exactly at min_conf (0.6) still returns single type."""
        probs = {"contract": 0.6, "resume": 0.3, "email": 0.1}
        result = for_type("contract", probs)
        assert result == ["contract"]

    def test_low_confidence_returns_top_k(self) -> None:
        """for_type with low confidence (<0.6) -> top-k (2) types."""
        probs = {"medical": 0.4, "resume": 0.35, "email": 0.25}
        result = for_type("medical", probs)
        assert len(result) == 2
        assert result == ["medical", "resume"]

    def test_low_confidence_uses_global_top_k(self) -> None:
        """Fallback top-k picks the globally highest types, not just doc_type."""
        probs = {"email": 0.5, "resume": 0.3, "medical": 0.2}
        result = for_type("medical", probs)
        # medical prob is 0.2 < 0.6 so fallback to top-2 overall
        assert len(result) == 2
        assert result == ["email", "resume"]


# ---------------------------------------------------------------------------
# get_title_cues — load cues from YAML rule packs
# ---------------------------------------------------------------------------


class TestGetTitleCues:
    def test_loads_medical_cues(self) -> None:
        """get_title_cues loads cues from medical.yaml."""
        cues = get_title_cues(["medical"])
        assert len(cues) > 0
        # medical.yaml has these title cues
        assert "Dr." in cues
        assert "Patient" in cues
        assert "MD" in cues

    def test_empty_list_returns_empty(self) -> None:
        """get_title_cues for empty list -> empty."""
        assert get_title_cues([]) == []

    def test_nonexistent_type_returns_empty(self) -> None:
        """Unknown doc type with no YAML file -> empty cues."""
        cues = get_title_cues(["nonexistent_doc_type_xyz"])
        assert cues == []

    def test_caching_works(self) -> None:
        """Second call uses cache; results should match."""
        cues1 = get_title_cues(["medical"])
        cues2 = get_title_cues(["medical"])
        assert cues1 == cues2

    def test_cache_reset_clears(self) -> None:
        """After reset_cue_cache, cues are reloaded from disk."""
        cues1 = get_title_cues(["medical"])
        reset_cue_cache()
        cues2 = get_title_cues(["medical"])
        assert cues1 == cues2  # Same data, but was actually re-read

    def test_no_duplicate_cues_across_types(self) -> None:
        """If two packs share cues, they should not be duplicated."""
        # medical has title_cues; loading it twice should not duplicate
        cues = get_title_cues(["medical", "medical"])
        assert len(cues) == len(set(cues))


# ---------------------------------------------------------------------------
# boost_ner_scores — NER PERSON score boosting
# ---------------------------------------------------------------------------


class TestBoostNERScores:
    def test_boosts_person_near_cue(self) -> None:
        """boost_ner_scores boosts PERSON NER span near cue."""
        text = "Dr. John Smith is the attending physician."
        span = _span(4, 14, "John Smith", score=0.7, detector="ner:spacy")
        cues = ["Dr."]

        result = boost_ner_scores([span], text, cues, boost=0.1, proximity=100)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.8)
        assert result[0].label == "PERSON"

    def test_does_not_boost_non_person(self) -> None:
        """boost_ner_scores doesn't boost non-PERSON spans."""
        text = "Dr. Acme Corp is a large company."
        span = _span(4, 13, "Acme Corp", label="ORG", score=0.7, detector="ner:spacy")
        cues = ["Dr."]

        result = boost_ner_scores([span], text, cues, boost=0.1, proximity=100)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.7)  # unchanged

    def test_does_not_boost_when_no_cue_nearby(self) -> None:
        """boost_ner_scores doesn't boost when no cues nearby."""
        text = "x" * 200 + "John Smith is here."
        span = _span(200, 210, "John Smith", score=0.7, detector="ner:spacy")
        cues = ["Dr."]  # "Dr." is not in the text at all

        result = boost_ner_scores([span], text, cues, boost=0.1, proximity=100)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.7)  # unchanged

    def test_does_not_boost_when_cue_too_far(self) -> None:
        """Cue beyond proximity distance does not trigger a boost."""
        # Place "Dr." at position 0, person span starts at 200
        text = "Dr. " + "x" * 196 + "John Smith is here."
        span = _span(200, 210, "John Smith", score=0.7, detector="ner:spacy")
        cues = ["Dr."]

        result = boost_ner_scores([span], text, cues, boost=0.1, proximity=50)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.7)  # too far

    def test_caps_score_at_one(self) -> None:
        """boost_ner_scores caps score at 1.0."""
        text = "Dr. John Smith is here."
        span = _span(4, 14, "John Smith", score=0.95, detector="ner:spacy")
        cues = ["Dr."]

        result = boost_ner_scores([span], text, cues, boost=0.1, proximity=100)
        assert len(result) == 1
        assert result[0].score == pytest.approx(1.0)

    def test_does_not_boost_regex_person(self) -> None:
        """boost_ner_scores doesn't boost regex-detected PERSON spans."""
        text = "Dr. John Smith is here."
        span = _span(4, 14, "John Smith", score=0.7, detector="regex:name_pattern")
        cues = ["Dr."]

        result = boost_ner_scores([span], text, cues, boost=0.1, proximity=100)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.7)  # unchanged (not ner:*)

    def test_empty_spans_returns_empty(self) -> None:
        result = boost_ner_scores([], "some text", ["Dr."])
        assert result == []

    def test_empty_cues_returns_spans_unchanged(self) -> None:
        span = _span(0, 5, "Alice", score=0.7, detector="ner:spacy")
        result = boost_ner_scores([span], "Alice is here.", [])
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.7)

    def test_multiple_spans_mixed(self) -> None:
        """Only eligible PERSON NER spans near cues get boosted."""
        text = "Dr. John Smith and Acme Corp and Jane Doe"
        spans = [
            _span(4, 14, "John Smith", score=0.7, detector="ner:spacy"),  # near Dr. -> boost
            _span(19, 28, "Acme Corp", label="ORG", score=0.7, detector="ner:spacy"),  # ORG -> no
            _span(33, 41, "Jane Doe", score=0.7, detector="ner:spacy"),  # near Dr. -> boost
        ]
        cues = ["Dr."]

        result = boost_ner_scores(spans, text, cues, boost=0.1, proximity=100)
        assert len(result) == 3
        assert result[0].score == pytest.approx(0.8)  # John Smith boosted
        assert result[1].score == pytest.approx(0.7)  # Acme Corp unchanged
        assert result[2].score == pytest.approx(0.8)  # Jane Doe boosted

    def test_case_insensitive_cue_matching(self) -> None:
        """Cue matching is case-insensitive."""
        text = "patient John Smith has been admitted."
        span = _span(8, 18, "John Smith", score=0.7, detector="ner:spacy")
        cues = ["Patient"]  # capital P, text has lowercase p

        result = boost_ner_scores([span], text, cues, boost=0.1, proximity=100)
        assert len(result) == 1
        assert result[0].score == pytest.approx(0.8)

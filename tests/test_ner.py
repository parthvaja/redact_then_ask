"""Tests for rta.detect.ner — NER-based PII detection (Phase 5).

Tests use mock spaCy/HF models to avoid requiring real model downloads.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rta.config import reset
from rta.schema import Document, Span


@pytest.fixture(autouse=True)
def _fresh_state() -> None:
    from rta.detect.ner.spacy_backend import reset_nlp_cache

    reset()
    reset_nlp_cache()
    yield  # type: ignore[misc]
    reset()
    reset_nlp_cache()


def _doc(text: str) -> Document:
    return Document(doc_id="test", text=text, source_path=None)


# ---------------------------------------------------------------------------
# Mock spaCy model
# ---------------------------------------------------------------------------


def _make_mock_nlp(entities: list[tuple[int, int, str]]):
    """Build a mock spaCy nlp callable that returns given entities.

    entities: list of (start_char, end_char, label_) tuples.
    """

    def mock_nlp(text: str):
        doc = MagicMock()

        # Build mock entities
        mock_ents = []
        for start, end, label in entities:
            ent = MagicMock()
            ent.start_char = start
            ent.end_char = end
            ent.label_ = label
            ent.text = text[start:end]
            mock_ents.append(ent)
        doc.ents = mock_ents

        # Build mock tokens for sliding window
        words = text.split()
        tokens = []
        pos = 0
        for w in words:
            idx = text.find(w, pos)
            tok = MagicMock()
            tok.idx = idx
            tok.text = w
            tok.__len__ = lambda self=tok: len(self.text)
            tokens.append(tok)
            pos = idx + len(w)
        doc.__len__ = lambda self=doc: len(tokens)
        doc.__iter__ = lambda self=doc: iter(tokens)

        return doc

    return mock_nlp


class TestSpacyBackend:
    def test_detects_person(self) -> None:
        """Detect a PERSON entity via spaCy."""
        from rta.detect.ner.spacy_backend import detect, reset_nlp_cache

        text = "John Smith went to the store."
        mock_nlp = _make_mock_nlp([(0, 10, "PERSON")])

        reset_nlp_cache()
        with patch("rta.detect.ner.spacy_backend._get_nlp", return_value=mock_nlp):
            doc = _doc(text)
            spans = detect(doc)

        persons = [s for s in spans if s.label == "PERSON"]
        assert len(persons) == 1
        assert persons[0].text == "John Smith"
        assert persons[0].start == 0
        assert persons[0].end == 10
        assert persons[0].detector == "ner:spacy"

    def test_detects_org(self) -> None:
        from rta.detect.ner.spacy_backend import detect, reset_nlp_cache

        text = "She works at Google Inc."
        mock_nlp = _make_mock_nlp([(13, 23, "ORG")])

        reset_nlp_cache()
        with patch("rta.detect.ner.spacy_backend._get_nlp", return_value=mock_nlp):
            spans = detect(_doc(text))

        orgs = [s for s in spans if s.label == "ORG"]
        assert len(orgs) == 1
        assert orgs[0].text == "Google Inc"

    def test_maps_gpe_to_location(self) -> None:
        from rta.detect.ner.spacy_backend import detect, reset_nlp_cache

        text = "He lives in Paris, France."
        mock_nlp = _make_mock_nlp([(12, 17, "GPE"), (19, 25, "GPE")])

        reset_nlp_cache()
        with patch("rta.detect.ner.spacy_backend._get_nlp", return_value=mock_nlp):
            spans = detect(_doc(text))

        locations = [s for s in spans if s.label == "LOCATION"]
        assert len(locations) == 2

    def test_discards_unmapped_labels(self) -> None:
        """Labels mapping to None (MONEY, PERCENT, etc.) are dropped."""
        from rta.detect.ner.spacy_backend import detect, reset_nlp_cache

        text = "The project cost $5 million."
        mock_nlp = _make_mock_nlp([(18, 28, "MONEY")])

        reset_nlp_cache()
        with patch("rta.detect.ner.spacy_backend._get_nlp", return_value=mock_nlp):
            spans = detect(_doc(text))

        assert len(spans) == 0

    def test_span_alignment(self) -> None:
        """All spans must align with doc.text."""
        from rta.detect.ner.spacy_backend import detect, reset_nlp_cache

        text = "Contact John Smith at john@example.com"
        mock_nlp = _make_mock_nlp([(8, 18, "PERSON")])

        reset_nlp_cache()
        with patch("rta.detect.ner.spacy_backend._get_nlp", return_value=mock_nlp):
            doc = _doc(text)
            spans = detect(doc)

        for span in spans:
            assert span.check_text(doc.text)

    def test_dedup_identical_spans(self) -> None:
        """Duplicate (start, end, label) tuples should be collapsed."""
        from rta.detect.ner.spacy_backend import _dedupe_spans

        spans = [
            Span(start=0, end=5, label="PERSON", text="Alice", score=0.9, detector="ner:spacy"),
            Span(start=0, end=5, label="PERSON", text="Alice", score=0.85, detector="ner:spacy"),
            Span(start=10, end=15, label="ORG", text="Acme.", score=0.8, detector="ner:spacy"),
        ]
        result = _dedupe_spans(spans)
        assert len(result) == 2

    def test_model_fallback(self) -> None:
        """Falls back to CPU model if TRF model unavailable."""
        from rta.detect.ner.spacy_backend import _get_nlp, reset_nlp_cache

        reset_nlp_cache()
        with patch("spacy.load") as mock_load:
            # First call fails (trf not installed), second succeeds
            mock_load.side_effect = [OSError("Model not found"), MagicMock()]
            _get_nlp()

        # Should have been called twice: trf then lg
        assert mock_load.call_count == 2


# ---------------------------------------------------------------------------
# HF backend
# ---------------------------------------------------------------------------


class TestHFBackend:
    def test_detects_entities(self) -> None:
        from rta.detect.ner.hf_backend import detect, reset_pipeline_cache

        text = "John Smith lives in Berlin."
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"entity_group": "PER", "score": 0.95, "start": 0, "end": 10, "word": "John Smith"},
            {"entity_group": "LOC", "score": 0.90, "start": 20, "end": 26, "word": "Berlin"},
        ]

        reset_pipeline_cache()
        with patch("rta.detect.ner.hf_backend._get_pipeline", return_value=mock_pipe):
            spans = detect(_doc(text))

        persons = [s for s in spans if s.label == "PERSON"]
        locations = [s for s in spans if s.label == "LOCATION"]
        assert len(persons) == 1
        assert len(locations) == 1
        assert persons[0].detector == "ner:hf"

    def test_filters_low_score(self) -> None:
        from rta.detect.ner.hf_backend import detect, reset_pipeline_cache

        text = "Maybe John Smith?"
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"entity_group": "PER", "score": 0.1, "start": 6, "end": 16, "word": "John Smith"},
        ]

        reset_pipeline_cache()
        with patch("rta.detect.ner.hf_backend._get_pipeline", return_value=mock_pipe):
            spans = detect(_doc(text))

        # Score 0.1 < min_score 0.4 → should be filtered
        assert len(spans) == 0

    def test_discards_misc(self) -> None:
        from rta.detect.ner.hf_backend import detect, reset_pipeline_cache

        text = "The Olympics are great."
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"entity_group": "MISC", "score": 0.9, "start": 4, "end": 12, "word": "Olympics"},
        ]

        reset_pipeline_cache()
        with patch("rta.detect.ner.hf_backend._get_pipeline", return_value=mock_pipe):
            spans = detect(_doc(text))

        assert len(spans) == 0


# ---------------------------------------------------------------------------
# Top-level run() dispatch
# ---------------------------------------------------------------------------


class TestNERRun:
    def test_dispatches_to_spacy(self) -> None:
        from rta.detect.ner import run

        with patch("rta.detect.ner.spacy_backend.detect") as mock_detect:
            mock_detect.return_value = [
                Span(start=0, end=5, label="PERSON", text="Alice", score=0.9, detector="ner:spacy")
            ]
            doc = _doc("Alice went home.")
            spans = run(doc)

        assert len(spans) == 1
        assert spans[0].label == "PERSON"

    def test_dispatches_to_hf(self) -> None:
        from rta.detect.ner import run

        with (
            patch("rta.detect.ner.get_config") as mock_cfg,
            patch("rta.detect.ner.hf_backend.detect") as mock_detect,
        ):
            cfg = MagicMock()
            cfg.detect.ner.backend = "hf"
            mock_cfg.return_value = cfg
            mock_detect.return_value = []

            run(_doc("Some text."))

        mock_detect.assert_called_once()

    def test_invalid_backend_raises(self) -> None:
        from rta.detect.ner import run

        with patch("rta.detect.ner.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.detect.ner.backend = "unknown"
            mock_cfg.return_value = cfg

            with pytest.raises(ValueError, match="Unknown NER backend"):
                run(_doc("Some text."))

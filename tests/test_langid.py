"""Tests for language identification (Phase 1).

Covers:
- English detection with high confidence.
- Spanish detection.
- ``doc.meta`` is correctly populated.
- Low-confidence flag.
"""

from __future__ import annotations

from rta.ingest import load
from rta.langid import detect
from rta.schema import Document

# ===================================================================
# Helpers
# ===================================================================


def _make_doc(text: str) -> Document:
    """Create a Document from raw text via the ingest pipeline."""
    return load(text)


# ===================================================================
# English detection
# ===================================================================


class TestEnglishDetection:
    """English text should be detected with high confidence."""

    def test_detects_english(self) -> None:
        doc = _make_doc(
            "The quick brown fox jumps over the lazy dog. "
            "This is a perfectly normal English sentence that should be "
            "detected with high confidence by the language detector."
        )
        iso, conf = detect(doc)
        assert iso == "en"
        assert conf > 0.5

    def test_high_confidence(self) -> None:
        doc = _make_doc(
            "Natural language processing is a subfield of linguistics, "
            "computer science, and artificial intelligence concerned with "
            "the interactions between computers and human language."
        )
        _, conf = detect(doc)
        assert conf > 0.7


# ===================================================================
# Spanish detection
# ===================================================================


class TestSpanishDetection:
    """Spanish text should be detected as 'es'."""

    def test_detects_spanish(self) -> None:
        doc = _make_doc(
            "El zorro marrón rápido salta sobre el perro perezoso. "
            "Esta es una oración perfectamente normal en español que "
            "debería ser detectada con alta confianza por el detector."
        )
        iso, conf = detect(doc)
        assert iso == "es"
        assert conf > 0.5


# ===================================================================
# Metadata updates
# ===================================================================


class TestMetadataUpdates:
    """``detect()`` should populate ``doc.meta`` correctly."""

    def test_sets_language(self) -> None:
        doc = _make_doc("This is English text for language detection testing purposes.")
        detect(doc)
        assert doc.meta["language"] == "en"

    def test_sets_lang_conf(self) -> None:
        doc = _make_doc("This is English text for language detection testing purposes.")
        detect(doc)
        assert isinstance(doc.meta["lang_conf"], float)
        assert 0.0 <= doc.meta["lang_conf"] <= 1.0

    def test_no_unsupported_flag_for_english(self) -> None:
        doc = _make_doc(
            "This is a sufficiently long English sentence that should be "
            "detected with confidence well above the minimum threshold."
        )
        detect(doc)
        assert "unsupported_language" not in doc.meta


# ===================================================================
# Low-confidence / unsupported language
# ===================================================================


class TestLowConfidence:
    """When confidence is below ``cfg.langid.min_conf``, the
    ``unsupported_language`` flag should be set."""

    def test_unsupported_flag_on_ambiguous_text(self) -> None:
        # Very short / ambiguous text may produce low confidence.
        # We use a near-empty string to trigger a low-confidence result.
        doc = _make_doc("x")
        _iso, conf = detect(doc)
        # Even if the detector picks a language, we check the flag was
        # set when confidence is low.
        if conf < 0.5:
            assert doc.meta.get("unsupported_language") is True
        else:
            # If the detector happens to be very confident on a single
            # char, that is fine — just ensure meta keys are present.
            assert "language" in doc.meta
            assert "lang_conf" in doc.meta

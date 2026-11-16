"""Tests for the ingest pipeline (Phase 1).

Covers:
- Text normalisation (NFC, line endings, space collapsing).
- ``load()`` dispatch for TXT, PDF, DOCX, and raw strings.
- Page-range integrity for PDFs.
- Unknown extension rejection.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from rta.ingest import load
from rta.ingest.normalize import normalize_text
from rta.schema import Document

# ===================================================================
# normalize_text
# ===================================================================


class TestNormalizeText:
    """Tests for ``rta.ingest.normalize.normalize_text``."""

    def test_nfc_normalisation(self) -> None:
        # U+0065 (e) + U+0301 (combining acute) → U+00E9 (e-acute) in NFC.
        nfd = "caf\u0065\u0301"
        assert unicodedata.is_normalized("NFD", nfd) or not unicodedata.is_normalized("NFC", nfd)
        result = normalize_text(nfd)
        assert "caf\u00e9" in result
        assert unicodedata.is_normalized("NFC", result)

    def test_crlf_to_lf(self) -> None:
        assert normalize_text("a\r\nb\r\nc") == "a\nb\nc"

    def test_cr_to_lf(self) -> None:
        assert normalize_text("a\rb\rc") == "a\nb\nc"

    def test_mixed_line_endings(self) -> None:
        assert normalize_text("a\r\nb\rc\nd") == "a\nb\nc\nd"

    def test_collapse_spaces(self) -> None:
        assert normalize_text("hello    world") == "hello world"

    def test_collapse_tabs(self) -> None:
        assert normalize_text("hello\t\tworld") == "hello world"

    def test_preserve_newlines(self) -> None:
        assert normalize_text("a\n\nb") == "a\n\nb"

    def test_mixed_space_and_newline(self) -> None:
        # Spaces around newlines should each collapse, but newlines stay.
        result = normalize_text("a   \n   b")
        assert result == "a \n b"

    def test_empty_string(self) -> None:
        assert normalize_text("") == ""

    def test_only_whitespace(self) -> None:
        result = normalize_text("   \t  ")
        assert result == " "


# ===================================================================
# load() — plain string
# ===================================================================


class TestLoadString:
    """Test ``load()`` when given a raw string."""

    def test_returns_document(self) -> None:
        doc = load("Hello, world!")
        assert isinstance(doc, Document)

    def test_text_is_normalised(self) -> None:
        doc = load("hello    world\r\n")
        assert doc.text == "hello world\n"

    def test_doc_id_is_sha256_prefix(self) -> None:
        import hashlib

        doc = load("test")
        expected_text = normalize_text("test")
        expected_id = hashlib.sha256(expected_text.encode()).hexdigest()[:16]
        assert doc.doc_id == expected_id

    def test_source_path_is_none(self) -> None:
        doc = load("some text")
        assert doc.source_path is None

    def test_pages_is_empty(self) -> None:
        doc = load("some text")
        assert doc.pages == []


# ===================================================================
# load() — TXT
# ===================================================================


class TestLoadTxt:
    """Test ``load()`` for plain-text files."""

    def test_loads_sample_txt(self, sample_txt: Path) -> None:
        doc = load(sample_txt)
        assert isinstance(doc, Document)
        assert "John Smith" in doc.text
        assert doc.source_path is not None

    def test_text_is_normalised(self, sample_txt: Path) -> None:
        doc = load(sample_txt)
        assert "\r" not in doc.text
        # Should be NFC
        assert unicodedata.is_normalized("NFC", doc.text)

    def test_doc_id_is_deterministic(self, sample_txt: Path) -> None:
        doc1 = load(sample_txt)
        doc2 = load(sample_txt)
        assert doc1.doc_id == doc2.doc_id


# ===================================================================
# load() — PDF
# ===================================================================


class TestLoadPdf:
    """Test ``load()`` for PDF files (fixture generated in conftest)."""

    def test_loads_pdf(self, sample_pdf: Path) -> None:
        doc = load(sample_pdf)
        assert isinstance(doc, Document)
        assert "Alice" in doc.text
        assert "Bob" in doc.text

    def test_page_ranges(self, sample_pdf: Path) -> None:
        doc = load(sample_pdf)
        assert len(doc.pages) == 2

        # Each page range should be valid.
        for start, end in doc.pages:
            assert 0 <= start <= end <= len(doc.text)

    def test_page_text_extractable(self, sample_pdf: Path) -> None:
        """``doc.text[start:end]`` for each page should be non-empty."""
        doc = load(sample_pdf)
        for i, (start, end) in enumerate(doc.pages):
            page_text = doc.text[start:end]
            assert len(page_text) > 0, f"Page {i} text is empty"

    def test_page_one_contains_alice(self, sample_pdf: Path) -> None:
        doc = load(sample_pdf)
        start, end = doc.pages[0]
        assert "Alice" in doc.text[start:end]

    def test_page_two_contains_bob(self, sample_pdf: Path) -> None:
        doc = load(sample_pdf)
        start, end = doc.pages[1]
        assert "Bob" in doc.text[start:end]


# ===================================================================
# load() — DOCX
# ===================================================================


class TestLoadDocx:
    """Test ``load()`` for DOCX files (fixture generated in conftest)."""

    def test_loads_docx(self, sample_docx: Path) -> None:
        doc = load(sample_docx)
        assert isinstance(doc, Document)
        assert "paragraph one" in doc.text

    def test_table_content_included(self, sample_docx: Path) -> None:
        doc = load(sample_docx)
        assert "Charlie" in doc.text
        assert "555-0199" in doc.text

    def test_email_preserved(self, sample_docx: Path) -> None:
        doc = load(sample_docx)
        assert "test@example.com" in doc.text


# ===================================================================
# load() — unsupported extension
# ===================================================================


class TestLoadUnsupported:
    """Ensure unknown extensions raise ``ValueError``."""

    def test_raises_on_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load(csv_file)

    def test_raises_on_html(self, tmp_path: Path) -> None:
        html_file = tmp_path / "page.html"
        html_file.write_text("<html></html>", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load(html_file)

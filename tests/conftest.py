"""Shared pytest fixtures for Redact-Then-Ask tests."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from docx import Document as DocxDocument

from rta.config import reset as reset_config
from rta.langid import reset_detector


@pytest.fixture(autouse=True)
def _clean_singletons() -> None:
    """Reset global singletons before and after every test."""
    reset_config()
    reset_detector()
    yield  # type: ignore[misc]
    reset_config()
    reset_detector()


# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def sample_txt() -> Path:
    """Path to the sample plain-text fixture."""
    return FIXTURES_DIR / "sample.txt"


# ---------------------------------------------------------------------------
# Programmatically-generated PDF fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    """Create a small two-page PDF and return its path."""
    pdf_path = tmp_path / "sample.pdf"
    doc = pymupdf.open()

    # Page 0
    page0 = doc.new_page(width=612, height=792)
    page0.insert_text(
        (72, 72),
        "Page one: Alice lives at 123 Maple St.",
        fontsize=12,
    )

    # Page 1
    page1 = doc.new_page(width=612, height=792)
    page1.insert_text(
        (72, 72),
        "Page two: Bob's SSN is 078-05-1120.",
        fontsize=12,
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------------
# Programmatically-generated DOCX fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_docx(tmp_path: Path) -> Path:
    """Create a DOCX with paragraphs and a table and return its path."""
    docx_path = tmp_path / "sample.docx"
    document = DocxDocument()

    document.add_paragraph("Hello from paragraph one.")
    document.add_paragraph("This is paragraph two with email test@example.com.")

    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Phone"
    table.cell(1, 0).text = "Charlie"
    table.cell(1, 1).text = "555-0199"

    document.save(str(docx_path))
    return docx_path

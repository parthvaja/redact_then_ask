"""Tests for rta.data — dataset loaders and converters.

Tests use synthetic mock data so they don't require network access or
real dataset downloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rta.config import reset
from rta.schema import read_gold_docs_jsonl


@pytest.fixture(autouse=True)
def _fresh_config() -> None:
    reset()
    yield
    reset()


# ---------------------------------------------------------------------------
# ai4privacy converter
# ---------------------------------------------------------------------------


def _make_ai4privacy_record(
    doc_id: str = "TEST01",
    text: str = "Contact John Smith at john@example.com or call 555-1234.",
    language: str = "English",
    split: str = "train",
) -> dict:
    """Build a synthetic ai4privacy-style record."""
    return {
        "id": doc_id,
        "source_text": text,
        "target_text": text,
        "privacy_mask": [
            {"value": "John Smith", "start": 8, "end": 18, "label": "GIVENNAME1"},
            {"value": "john@example.com", "start": 22, "end": 38, "label": "EMAIL"},
            {"value": "555-1234", "start": 47, "end": 55, "label": "TEL"},
        ],
        "span_labels": "[]",
        "mbert_text_tokens": [],
        "mbert_bio_labels": [],
        "language": language,
        "set": split,
    }


class TestAi4PrivacyConvert:
    def test_convert_basic(self, tmp_path: Path) -> None:
        """Convert a single synthetic record and verify the output."""
        from rta.data.ai4privacy import convert

        records = [_make_ai4privacy_record()]

        # Mock the HF dataset to yield our synthetic record
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(records))

        with (
            patch("datasets.load_dataset", return_value=mock_ds),
            patch("rta.config.get_config") as mock_cfg,
        ):
            cfg = MagicMock()
            cfg.paths.data = str(tmp_path / "data")
            mock_cfg.return_value = cfg

            out = convert(split="train", language="English", max_docs=10)

        assert out.exists()
        docs = read_gold_docs_jsonl(out)
        assert len(docs) == 1

        doc = docs[0]
        assert doc.doc_id == "TEST01"
        assert doc.source == "ai4privacy"
        assert doc.language == "en"

        # Check span labels
        labels = {s.label for s in doc.spans}
        assert "PERSON" in labels
        assert "EMAIL" in labels
        assert "PHONE" in labels

        # Check text alignment
        for span in doc.spans:
            assert span.check_text(doc.text), (
                f"Span text mismatch: {span.text!r} != {doc.text[span.start : span.end]!r}"
            )

    def test_skips_non_english(self, tmp_path: Path) -> None:
        """Non-English records are filtered out."""
        from rta.data.ai4privacy import convert

        records = [
            _make_ai4privacy_record(doc_id="EN1", language="English"),
            _make_ai4privacy_record(doc_id="FR1", language="French"),
        ]

        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(records))

        with (
            patch("datasets.load_dataset", return_value=mock_ds),
            patch("rta.config.get_config") as mock_cfg,
        ):
            cfg = MagicMock()
            cfg.paths.data = str(tmp_path / "data")
            mock_cfg.return_value = cfg

            out = convert(split="train", language="English")

        docs = read_gold_docs_jsonl(out)
        assert len(docs) == 1
        assert docs[0].doc_id == "EN1"

    def test_discarded_labels_are_skipped(self, tmp_path: Path) -> None:
        """Labels mapping to None (TIME, TITLE, SEX) produce no spans."""
        from rta.data.ai4privacy import convert

        record = {
            "id": "SKIP01",
            "source_text": "Mr. Smith at 10:30",
            "target_text": "Mr. Smith at 10:30",
            "privacy_mask": [
                {"value": "Mr.", "start": 0, "end": 3, "label": "TITLE"},
                {"value": "10:30", "start": 13, "end": 18, "label": "TIME"},
            ],
            "span_labels": "[]",
            "mbert_text_tokens": [],
            "mbert_bio_labels": [],
            "language": "English",
            "set": "train",
        }

        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter([record]))

        with (
            patch("datasets.load_dataset", return_value=mock_ds),
            patch("rta.config.get_config") as mock_cfg,
        ):
            cfg = MagicMock()
            cfg.paths.data = str(tmp_path / "data")
            mock_cfg.return_value = cfg

            out = convert(split="train")

        docs = read_gold_docs_jsonl(out)
        # No spans survived → doc not written (empty spans list)
        assert len(docs) == 0

    def test_max_docs_limit(self, tmp_path: Path) -> None:
        """max_docs caps the output."""
        from rta.data.ai4privacy import convert

        records = [_make_ai4privacy_record(doc_id=f"D{i}", language="English") for i in range(10)]

        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(records))

        with (
            patch("datasets.load_dataset", return_value=mock_ds),
            patch("rta.config.get_config") as mock_cfg,
        ):
            cfg = MagicMock()
            cfg.paths.data = str(tmp_path / "data")
            mock_cfg.return_value = cfg

            out = convert(split="train", max_docs=3)

        docs = read_gold_docs_jsonl(out)
        assert len(docs) == 3


# ---------------------------------------------------------------------------
# TAB converter
# ---------------------------------------------------------------------------


def _make_tab_document(
    doc_id: str = "001-99999",
    text: str = "Mr Cengiz Polat was born in Istanbul on 15 October 1965.",
) -> dict:
    """Build a synthetic TAB-style document."""
    return {
        "doc_id": doc_id,
        "text": text,
        "dataset_type": "test",
        "task": "Task: Annotate the document",
        "quality_checked": True,
        "meta": {"year": 2020, "countries": "TUR"},
        "annotations": {
            "annotator1": {
                "entity_mentions": [
                    {
                        "entity_type": "PERSON",
                        "entity_mention_id": f"{doc_id}_a1_em1",
                        "start_offset": 3,
                        "end_offset": 16,
                        "span_text": "Cengiz Polat",
                        "edit_type": "check",
                        "identifier_type": "DIRECT",
                        "entity_id": f"{doc_id}_a1_e1",
                        "confidential_status": "NOT_CONFIDENTIAL",
                    },
                    {
                        "entity_type": "LOC",
                        "entity_mention_id": f"{doc_id}_a1_em2",
                        "start_offset": 29,
                        "end_offset": 37,
                        "span_text": "Istanbul",
                        "edit_type": "check",
                        "identifier_type": "QUASI",
                        "entity_id": f"{doc_id}_a1_e2",
                        "confidential_status": "NOT_CONFIDENTIAL",
                    },
                    {
                        "entity_type": "DATETIME",
                        "entity_mention_id": f"{doc_id}_a1_em3",
                        "start_offset": 41,
                        "end_offset": 56,
                        "span_text": "15 October 1965",
                        "edit_type": "check",
                        "identifier_type": "QUASI",
                        "entity_id": f"{doc_id}_a1_e3",
                        "confidential_status": "NOT_CONFIDENTIAL",
                    },
                    {
                        "entity_type": "ORG",
                        "entity_mention_id": f"{doc_id}_a1_em4",
                        "start_offset": 0,
                        "end_offset": 2,
                        "span_text": "Mr",
                        "edit_type": "check",
                        "identifier_type": "NO_MASK",
                        "entity_id": f"{doc_id}_a1_e4",
                        "confidential_status": "NOT_CONFIDENTIAL",
                    },
                ]
            }
        },
    }


class TestTabConvert:
    def test_convert_basic(self, tmp_path: Path) -> None:
        """Convert a synthetic TAB document."""
        from rta.data.tab import convert

        raw_dir = tmp_path / "raw" / "tab"
        raw_dir.mkdir(parents=True)
        tab_data = [_make_tab_document()]
        (raw_dir / "echr_test.json").write_text(json.dumps(tab_data), encoding="utf-8")

        with patch("rta.config.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.paths.data = str(tmp_path / "data")
            mock_cfg.return_value = cfg

            out = convert(split="test", data_dir=raw_dir)

        assert out.exists()
        docs = read_gold_docs_jsonl(out)
        assert len(docs) == 1

        doc = docs[0]
        assert doc.doc_id == "001-99999"
        assert doc.doc_type == "court_case"
        assert doc.language == "en"
        assert doc.source == "tab"

        labels = {s.label for s in doc.spans}
        assert "PERSON" in labels
        assert "LOCATION" in labels
        assert "DATE" in labels

        # Check text alignment
        for span in doc.spans:
            assert span.check_text(doc.text)

    def test_no_mask_entries_skipped(self, tmp_path: Path) -> None:
        """Entries with identifier_type=NO_MASK are excluded."""
        from rta.data.tab import convert

        raw_dir = tmp_path / "raw" / "tab"
        raw_dir.mkdir(parents=True)
        tab_data = [_make_tab_document()]
        (raw_dir / "echr_test.json").write_text(json.dumps(tab_data), encoding="utf-8")

        with patch("rta.config.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.paths.data = str(tmp_path / "data")
            mock_cfg.return_value = cfg

            out = convert(split="test", data_dir=raw_dir)

        docs = read_gold_docs_jsonl(out)
        doc = docs[0]
        # The "Mr" mention with NO_MASK should not be present
        span_texts = {s.text for s in doc.spans}
        assert "Mr" not in span_texts
        # 3 spans remain (PERSON, LOC, DATETIME) out of 4 original
        assert len(doc.spans) == 3

    def test_missing_annotator_fallback(self, tmp_path: Path) -> None:
        """Falls back to first available annotator if requested one is missing."""
        from rta.data.tab import convert

        raw_dir = tmp_path / "raw" / "tab"
        raw_dir.mkdir(parents=True)

        doc = _make_tab_document()
        # Rename annotator1 → annotator5
        doc["annotations"]["annotator5"] = doc["annotations"].pop("annotator1")

        (raw_dir / "echr_test.json").write_text(json.dumps([doc]), encoding="utf-8")

        with patch("rta.config.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.paths.data = str(tmp_path / "data")
            mock_cfg.return_value = cfg

            out = convert(split="test", annotator="annotator1", data_dir=raw_dir)

        docs = read_gold_docs_jsonl(out)
        assert len(docs) == 1
        # Should have fallen back to annotator5
        assert len(docs[0].spans) == 3

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Raise FileNotFoundError when TAB JSON doesn't exist."""
        from rta.data.tab import convert

        raw_dir = tmp_path / "raw" / "tab"
        raw_dir.mkdir(parents=True)
        # No files created

        with patch("rta.config.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.paths.data = str(tmp_path / "data")
            mock_cfg.return_value = cfg

            with pytest.raises(FileNotFoundError):
                convert(split="test", data_dir=raw_dir)

    def test_max_docs_limit(self, tmp_path: Path) -> None:
        """max_docs caps the output."""
        from rta.data.tab import convert

        raw_dir = tmp_path / "raw" / "tab"
        raw_dir.mkdir(parents=True)

        docs = [_make_tab_document(doc_id=f"001-{i:05d}") for i in range(10)]
        (raw_dir / "echr_test.json").write_text(json.dumps(docs), encoding="utf-8")

        with patch("rta.config.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.paths.data = str(tmp_path / "data")
            mock_cfg.return_value = cfg

            out = convert(split="test", data_dir=raw_dir, max_docs=3)

        result = read_gold_docs_jsonl(out)
        assert len(result) == 3

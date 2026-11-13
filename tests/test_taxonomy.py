"""Tests for rta.taxonomy — label set and external label mappings.

Gate criteria for Phase 0:
- LABELS set contains all expected labels.
- DIRECT_IDENTIFIERS is a subset of LABELS.
- LABEL_PRECEDENCE covers all labels.
- normalize_label strips IOB prefixes and returns canonical labels.
- All mapping dicts contain only values that are either None or in LABELS.
"""

from __future__ import annotations

import pytest

from rta.taxonomy import (
    AI4PRIVACY_TO_LABEL,
    CONLL_TO_LABEL,
    DIRECT_IDENTIFIERS,
    HF_NER_TO_LABEL,
    LABEL_PRECEDENCE,
    LABELS,
    PRESIDIO_TO_LABEL,
    QUASI_IDENTIFIERS,
    SPACY_TO_LABEL,
    TAB_TO_LABEL,
    label_precedence_rank,
    normalize_label,
)


class TestLabelSet:
    def test_all_expected_labels_present(self) -> None:
        expected = {
            "PERSON",
            "ORG",
            "LOCATION",
            "ADDRESS",
            "EMAIL",
            "PHONE",
            "SSN",
            "CREDIT_CARD",
            "IBAN",
            "IP_ADDRESS",
            "URL",
            "DATE",
            "ID_NUMBER",
            "MRN",
            "CASE_NUMBER",
            "AGE",
        }
        assert expected <= LABELS

    def test_direct_identifiers_subset_of_labels(self) -> None:
        assert DIRECT_IDENTIFIERS <= LABELS

    def test_quasi_identifiers_subset_of_labels(self) -> None:
        assert QUASI_IDENTIFIERS <= LABELS

    def test_direct_and_quasi_partition_labels(self) -> None:
        assert DIRECT_IDENTIFIERS | QUASI_IDENTIFIERS == LABELS
        assert frozenset() == DIRECT_IDENTIFIERS & QUASI_IDENTIFIERS

    def test_key_direct_identifiers_present(self) -> None:
        for label in ("PERSON", "EMAIL", "PHONE", "SSN", "CREDIT_CARD"):
            assert label in DIRECT_IDENTIFIERS, f"{label} missing from DIRECT_IDENTIFIERS"

    def test_precedence_covers_all_labels(self) -> None:
        """Every label in LABELS should appear in LABEL_PRECEDENCE."""
        for label in LABELS:
            assert label in LABEL_PRECEDENCE, f"Label {label!r} missing from LABEL_PRECEDENCE"

    def test_precedence_no_duplicates(self) -> None:
        assert len(LABEL_PRECEDENCE) == len(set(LABEL_PRECEDENCE))


class TestNormalizeLabel:
    def test_simple_lookup(self) -> None:
        assert normalize_label("PERSON", SPACY_TO_LABEL) == "PERSON"

    def test_iob_b_prefix_stripped(self) -> None:
        assert normalize_label("B-PER", HF_NER_TO_LABEL) == "PERSON"

    def test_iob_i_prefix_stripped(self) -> None:
        assert normalize_label("I-PER", HF_NER_TO_LABEL) == "PERSON"

    def test_bioes_s_prefix_stripped(self) -> None:
        assert normalize_label("S-PER", HF_NER_TO_LABEL) == "PERSON"

    def test_bioes_e_prefix_stripped(self) -> None:
        assert normalize_label("E-ORG", HF_NER_TO_LABEL) == "ORG"

    def test_discard_returns_none(self) -> None:
        result = normalize_label("MISC", HF_NER_TO_LABEL)
        assert result is None

    def test_unknown_label_raises(self) -> None:
        with pytest.raises(KeyError):
            normalize_label("TOTALLY_UNKNOWN", SPACY_TO_LABEL)

    def test_case_insensitive(self) -> None:
        # The function uppercases before lookup
        result = normalize_label("person", SPACY_TO_LABEL)
        assert result == "PERSON"


class TestMappingValidity:
    """All mapping values must be either None or a valid taxonomy label."""

    def _check_mapping(self, mapping: dict, name: str) -> None:
        for raw, label in mapping.items():
            if label is not None:
                assert label in LABELS, f"{name}[{raw!r}] = {label!r} is not in LABELS"

    def test_spacy_mapping(self) -> None:
        self._check_mapping(SPACY_TO_LABEL, "SPACY_TO_LABEL")

    def test_hf_ner_mapping(self) -> None:
        self._check_mapping(HF_NER_TO_LABEL, "HF_NER_TO_LABEL")

    def test_ai4privacy_mapping(self) -> None:
        self._check_mapping(AI4PRIVACY_TO_LABEL, "AI4PRIVACY_TO_LABEL")

    def test_tab_mapping(self) -> None:
        self._check_mapping(TAB_TO_LABEL, "TAB_TO_LABEL")

    def test_conll_mapping(self) -> None:
        self._check_mapping(CONLL_TO_LABEL, "CONLL_TO_LABEL")

    def test_presidio_mapping(self) -> None:
        self._check_mapping(PRESIDIO_TO_LABEL, "PRESIDIO_TO_LABEL")


class TestLabelPrecedenceRank:
    def test_ssn_has_high_priority(self) -> None:
        """SSN is first in the precedence list → rank 0."""
        assert label_precedence_rank("SSN") == 0

    def test_age_has_low_priority(self) -> None:
        ssn_rank = label_precedence_rank("SSN")
        age_rank = label_precedence_rank("AGE")
        assert age_rank > ssn_rank

    def test_unknown_label_gets_max_rank(self) -> None:
        rank = label_precedence_rank("FOOBAR_UNKNOWN")
        assert rank == len(LABEL_PRECEDENCE)

"""Tests for rta.detect.regex — regex-based PII detection (Phase 4).

Tests exercise validators, rule loading, and the scan engine using
synthetic documents and the real rule YAML configs.
"""

from __future__ import annotations

import pytest

from rta.config import reset
from rta.schema import Document


@pytest.fixture(autouse=True)
def _fresh_state() -> None:
    from rta.detect.regex.engine import reset_cache

    reset()
    reset_cache()
    yield  # type: ignore[misc]
    reset()
    reset_cache()


def _doc(text: str) -> Document:
    return Document(doc_id="test", text=text, source_path=None)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class TestEmailValidator:
    def test_valid_email(self) -> None:
        from rta.detect.regex.validators import email_validator

        assert email_validator("user@example.com") is True

    def test_missing_tld(self) -> None:
        from rta.detect.regex.validators import email_validator

        assert email_validator("user@localhost") is False

    def test_short_tld(self) -> None:
        from rta.detect.regex.validators import email_validator

        assert email_validator("user@example.c") is False


class TestSSNValidator:
    def test_valid_ssn_dashes(self) -> None:
        from rta.detect.regex.validators import ssn_validator

        assert ssn_validator("123-45-6789") is True

    def test_valid_ssn_no_dashes(self) -> None:
        from rta.detect.regex.validators import ssn_validator

        assert ssn_validator("123456789") is True

    def test_invalid_area_000(self) -> None:
        from rta.detect.regex.validators import ssn_validator

        assert ssn_validator("000-12-3456") is False

    def test_invalid_area_666(self) -> None:
        from rta.detect.regex.validators import ssn_validator

        assert ssn_validator("666-12-3456") is False

    def test_invalid_area_9xx(self) -> None:
        from rta.detect.regex.validators import ssn_validator

        assert ssn_validator("900-12-3456") is False

    def test_invalid_group_00(self) -> None:
        from rta.detect.regex.validators import ssn_validator

        assert ssn_validator("123-00-6789") is False


class TestLuhnValidator:
    def test_valid_visa(self) -> None:
        from rta.detect.regex.validators import luhn_validator

        # Known valid test number
        assert luhn_validator("4111111111111111") is True

    def test_invalid_number(self) -> None:
        from rta.detect.regex.validators import luhn_validator

        assert luhn_validator("4111111111111112") is False

    def test_with_spaces(self) -> None:
        from rta.detect.regex.validators import luhn_validator

        assert luhn_validator("4111 1111 1111 1111") is True

    def test_too_short(self) -> None:
        from rta.detect.regex.validators import luhn_validator

        assert luhn_validator("411111") is False


class TestIBANValidator:
    def test_valid_gb_iban(self) -> None:
        from rta.detect.regex.validators import iban_validator

        assert iban_validator("GB82WEST12345698765432") is True

    def test_invalid_check_digits(self) -> None:
        from rta.detect.regex.validators import iban_validator

        assert iban_validator("GB00WEST12345698765432") is False

    def test_too_short(self) -> None:
        from rta.detect.regex.validators import iban_validator

        assert iban_validator("GB82WEST") is False


class TestIPValidator:
    def test_valid_ipv4(self) -> None:
        from rta.detect.regex.validators import ip_validator

        assert ip_validator("192.168.1.100") is True

    def test_loopback_rejected(self) -> None:
        from rta.detect.regex.validators import ip_validator

        assert ip_validator("127.0.0.1") is False

    def test_invalid_ip(self) -> None:
        from rta.detect.regex.validators import ip_validator

        assert ip_validator("999.999.999.999") is False


class TestDateValidator:
    def test_mdy_slash(self) -> None:
        from rta.detect.regex.validators import date_validator

        assert date_validator("01/15/2023") is True

    def test_written_format(self) -> None:
        from rta.detect.regex.validators import date_validator

        assert date_validator("January 15, 2023") is True

    def test_garbage(self) -> None:
        from rta.detect.regex.validators import date_validator

        assert date_validator("not a date") is False


# ---------------------------------------------------------------------------
# Rule pack loading
# ---------------------------------------------------------------------------


class TestLoadPacks:
    def test_loads_common_pack(self) -> None:
        from rta.detect.regex.engine import load_packs

        rules = load_packs(doc_type=None)
        assert len(rules) > 0
        pack_ids = {r.pack_id for r in rules}
        assert "common" in pack_ids

    def test_loads_medical_pack(self) -> None:
        from rta.detect.regex.engine import load_packs

        rules = load_packs(doc_type="medical")
        pack_ids = {r.pack_id for r in rules}
        assert "common" in pack_ids
        assert "medical" in pack_ids

    def test_unknown_doc_type_only_common(self) -> None:
        from rta.detect.regex.engine import load_packs

        rules = load_packs(doc_type="nonexistent_type")
        pack_ids = {r.pack_id for r in rules}
        assert "common" in pack_ids
        assert len(pack_ids) == 1


# ---------------------------------------------------------------------------
# Scan engine — common patterns
# ---------------------------------------------------------------------------


class TestScanEmail:
    def test_detects_email(self) -> None:
        from rta.detect.regex import run

        doc = _doc("Contact us at info@example.com for details.")
        spans = run(doc)
        emails = [s for s in spans if s.label == "EMAIL"]
        assert len(emails) >= 1
        assert emails[0].text == "info@example.com"
        assert emails[0].validated is True
        assert emails[0].score == 0.99


class TestScanSSN:
    def test_detects_ssn_dashes(self) -> None:
        from rta.detect.regex import run

        doc = _doc("SSN: 123-45-6789")
        spans = run(doc)
        ssns = [s for s in spans if s.label == "SSN"]
        assert len(ssns) >= 1
        assert ssns[0].text == "123-45-6789"
        assert ssns[0].validated is True


class TestScanCreditCard:
    def test_detects_visa(self) -> None:
        from rta.detect.regex import run

        doc = _doc("Card number: 4111111111111111 is on file.")
        spans = run(doc)
        cards = [s for s in spans if s.label == "CREDIT_CARD"]
        assert len(cards) >= 1
        assert cards[0].validated is True

    def test_rejects_invalid_luhn(self) -> None:
        from rta.detect.regex import run

        doc = _doc("Card number: 4111111111111112 is on file.")
        spans = run(doc)
        cards = [s for s in spans if s.label == "CREDIT_CARD"]
        # Should be dropped since keep_unvalidated=False by default
        assert len(cards) == 0


class TestScanIP:
    def test_detects_public_ipv4(self) -> None:
        from rta.detect.regex import run

        doc = _doc("Server at 8.8.8.8 is reachable.")
        spans = run(doc)
        ips = [s for s in spans if s.label == "IP_ADDRESS"]
        assert len(ips) >= 1
        assert ips[0].text == "8.8.8.8"
        assert ips[0].validated is True

    def test_rejects_loopback(self) -> None:
        from rta.detect.regex import run

        doc = _doc("Localhost is 127.0.0.1 always.")
        spans = run(doc)
        ips = [s for s in spans if s.label == "IP_ADDRESS"]
        assert len(ips) == 0


class TestScanURL:
    def test_detects_https_url(self) -> None:
        from rta.detect.regex import run

        doc = _doc("Visit https://example.com/page for info.")
        spans = run(doc)
        urls = [s for s in spans if s.label == "URL"]
        assert len(urls) >= 1
        assert "https://example.com" in urls[0].text


class TestScanDate:
    def test_detects_written_date(self) -> None:
        from rta.detect.regex import run

        doc = _doc("The hearing was on January 15, 2023 in court.")
        spans = run(doc)
        dates = [s for s in spans if s.label == "DATE"]
        assert len(dates) >= 1
        assert "January 15, 2023" in dates[0].text


class TestScanAge:
    def test_detects_year_old(self) -> None:
        from rta.detect.regex import run

        doc = _doc("The patient is a 45-year-old male.")
        spans = run(doc)
        ages = [s for s in spans if s.label == "AGE"]
        assert len(ages) >= 1


# ---------------------------------------------------------------------------
# Scan engine — medical patterns
# ---------------------------------------------------------------------------


class TestScanMedical:
    def test_detects_mrn(self) -> None:
        from rta.detect.regex import run

        doc = _doc("MRN: ABC-12345 Patient admitted today.")
        spans = run(doc, doc_type="medical")
        mrns = [s for s in spans if s.label == "MRN"]
        assert len(mrns) >= 1
        assert "ABC-12345" in mrns[0].text

    def test_detects_insurance_id(self) -> None:
        from rta.detect.regex import run

        doc = _doc("Member ID: XYZ789012 is active.")
        spans = run(doc, doc_type="medical")
        ids = [s for s in spans if s.label == "ID_NUMBER"]
        assert len(ids) >= 1


# ---------------------------------------------------------------------------
# Span alignment
# ---------------------------------------------------------------------------


class TestSpanAlignment:
    def test_all_spans_aligned(self) -> None:
        """Every detected span's text must match its offset slice."""
        from rta.detect.regex import run

        text = (
            "Email: alice@example.com, SSN: 123-45-6789, "
            "Card: 4111111111111111, DOB: January 15, 2023"
        )
        doc = _doc(text)
        spans = run(doc)
        for span in spans:
            assert span.check_text(doc.text), (
                f"Span {span.detector}:{span.label} text mismatch: "
                f"{span.text!r} != {doc.text[span.start : span.end]!r}"
            )


# ---------------------------------------------------------------------------
# Phone detection
# ---------------------------------------------------------------------------


class TestPhoneDetection:
    def test_detects_us_phone(self) -> None:
        """Phone detection via phonenumbers library."""
        from rta.detect.regex.engine import _detect_phones

        spans = _detect_phones("Call me at +1 (650) 253-0000 please.")
        phones = [s for s in spans if s.label == "PHONE"]
        assert len(phones) >= 1
        assert phones[0].validated is True

"""PII label taxonomy for Redact-Then-Ask.

The fixed label set is defined here.  Every external label (spaCy entity
types, Hugging Face NER tags, ai4privacy fields, Presidio recognizer
outputs, CoNLL entity types) is mapped to these labels in this module.
No raw external label should ever escape the data-loading layer.

See §5 of ``docs/PROJECT_DOC.md`` for the full taxonomy rationale.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical label set
# ---------------------------------------------------------------------------

#: All valid taxonomy labels.  ``Span.label`` must be one of these.
LABELS: frozenset[str] = frozenset(
    {
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
)

#: Direct identifiers: uniquely identify a person alone.
#: Headline recall is computed over this subset.
DIRECT_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "PERSON",
        "EMAIL",
        "PHONE",
        "SSN",
        "CREDIT_CARD",
        "IBAN",
        "ID_NUMBER",
        "MRN",
        "CASE_NUMBER",
        "ADDRESS",
    }
)

#: Quasi-identifiers: identify in combination.  Reported separately.
QUASI_IDENTIFIERS: frozenset[str] = LABELS - DIRECT_IDENTIFIERS

#: Label precedence for conflict resolution (earlier = higher priority).
#: See ``rta.resolve`` §6.7.
LABEL_PRECEDENCE: list[str] = [
    "SSN",
    "CREDIT_CARD",
    "IBAN",
    "ID_NUMBER",
    "MRN",
    "CASE_NUMBER",
    "PHONE",
    "EMAIL",
    "ADDRESS",
    "PERSON",
    "ORG",
    "LOCATION",
    "IP_ADDRESS",
    "URL",
    "DATE",
    "AGE",
]

# ---------------------------------------------------------------------------
# Mapping from spaCy entity types
# ---------------------------------------------------------------------------

#: spaCy ``en_core_web_*`` entity types → taxonomy label.
#: ``None`` means "discard" (not a PII type we track).
SPACY_TO_LABEL: dict[str, str | None] = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOCATION",  # geo-political entity
    "LOC": "LOCATION",  # non-GPE location
    "FAC": "LOCATION",  # facility (airport, stadium, …)
    "CARDINAL": None,
    "DATE": "DATE",
    "TIME": None,
    "MONEY": None,
    "PERCENT": None,
    "QUANTITY": None,
    "ORDINAL": None,
    "WORK_OF_ART": None,
    "LAW": None,
    "LANGUAGE": None,
    "EVENT": None,
    "NORP": None,
    "PRODUCT": None,
}

# ---------------------------------------------------------------------------
# Mapping from Hugging Face NER tags (dslim/bert-base-NER and similar)
# ---------------------------------------------------------------------------

#: HF IOB/BIOES tag prefixes (B-, I-, S-, E-) are stripped before lookup.
HF_NER_TO_LABEL: dict[str, str | None] = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "GPE": "LOCATION",
    "MISC": None,
    "O": None,
}

# ---------------------------------------------------------------------------
# Mapping from ai4privacy dataset fields
# ---------------------------------------------------------------------------

#: ``privacy_mask[].label`` values in the ai4privacy/pii-masking-300k
#: dataset → taxonomy label.  Verified against 500 streaming records
#: on 2026-11-13.
AI4PRIVACY_TO_LABEL: dict[str, str | None] = {
    # --- Person names --------------------------------------------------
    "GIVENNAME1": "PERSON",
    "GIVENNAME2": "PERSON",
    "LASTNAME1": "PERSON",
    "LASTNAME2": "PERSON",
    "LASTNAME3": "PERSON",
    "USERNAME": "PERSON",  # treat usernames as PERSON for recall
    "TITLE": None,  # honorific (Mr., Mrs.) — not PII on its own
    # --- Contact -------------------------------------------------------
    "EMAIL": "EMAIL",
    "TEL": "PHONE",
    "IP": "IP_ADDRESS",
    "URL": "URL",
    # --- Identifiers ---------------------------------------------------
    "SOCIALNUMBER": "SSN",
    "IDCARD": "ID_NUMBER",
    "PASSPORT": "ID_NUMBER",
    "DRIVERLICENSE": "ID_NUMBER",
    # --- Location / address -------------------------------------------
    "STREET": "ADDRESS",
    "BUILDING": "ADDRESS",
    "SECADDRESS": "ADDRESS",  # secondary address line
    "POSTCODE": "ADDRESS",
    "CITY": "LOCATION",
    "STATE": "LOCATION",
    "COUNTRY": "LOCATION",
    "GEOCOORD": None,  # lat/lon coordinates — no taxonomy match
    # --- Temporal ------------------------------------------------------
    "DATE": "DATE",
    "BOD": "DATE",  # date of birth
    "TIME": None,  # time of day — not PII on its own
    # --- Demographics --------------------------------------------------
    "SEX": None,  # gender — quasi at best, not in taxonomy
    "PASS": None,  # password — not PII in our taxonomy sense
    # --- Kept from earlier guesses for robustness ---------------------
    "NAME_STUDENT": "PERSON",
    "NAME": "PERSON",
    "FULLNAME": "PERSON",
    "FIRSTNAME": "PERSON",
    "LASTNAME": "PERSON",
    "EMAILADDRESS": "EMAIL",
    "PHONE": "PHONE",
    "PHONENUMBER": "PHONE",
    "PHONE_NUM": "PHONE",
    "SSN": "SSN",
    "SOCIALSECURITYNUMBER": "SSN",
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "CREDITCARD": "CREDIT_CARD",
    "IBAN": "IBAN",
    "BANKACCOUNT": "IBAN",
    "IPADDRESS": "IP_ADDRESS",
    "WEBSITE": "URL",
    "DATEOFBIRTH": "DATE",
    "DOB": "DATE",
    "AGE": "AGE",
    "ADDRESS": "ADDRESS",
    "STREETADDRESS": "ADDRESS",
    "ZIPCODE": "ADDRESS",
    "ORGANIZATION": "ORG",
    "COMPANY": "ORG",
    "IDNUMBER": "ID_NUMBER",
    "LICENSEPLATE": "ID_NUMBER",
    "TAXID": "ID_NUMBER",
    "MEDICALRECORD": "MRN",
    "MRN": "MRN",
    "CASENUMBER": "CASE_NUMBER",
}

# ---------------------------------------------------------------------------
# Mapping from Text Anonymization Benchmark (TAB) entity types
# ---------------------------------------------------------------------------

#: TAB entity category strings → taxonomy label.
TAB_TO_LABEL: dict[str, str | None] = {
    "PERSON": "PERSON",
    "CODE": "ID_NUMBER",  # document/case codes
    "LOC": "LOCATION",
    "ORG": "ORG",
    "DEM": None,  # demographic (generic) — too vague
    "DATETIME": "DATE",
    "QUANTITY": None,
    "MISC": None,
}

# ---------------------------------------------------------------------------
# Mapping from CoNLL-2003 entity types
# ---------------------------------------------------------------------------

#: CoNLL-2003 → taxonomy label.
CONLL_TO_LABEL: dict[str, str | None] = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION",
    "MISC": None,
}

# ---------------------------------------------------------------------------
# Mapping from Microsoft Presidio recognizer entity types
# ---------------------------------------------------------------------------

#: Presidio ``RecognizerResult.entity_type`` → taxonomy label.
PRESIDIO_TO_LABEL: dict[str, str | None] = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "US_SSN": "SSN",
    "CREDIT_CARD": "CREDIT_CARD",
    "IBAN_CODE": "IBAN",
    "IP_ADDRESS": "IP_ADDRESS",
    "URL": "URL",
    "DATE_TIME": "DATE",
    "NRP": None,  # nationality/religion/political group
    "LOCATION": "LOCATION",
    "ORGANIZATION": "ORG",  # NOTE: not a default Presidio type; added via transformer
    "US_DRIVER_LICENSE": "ID_NUMBER",
    "US_PASSPORT": "ID_NUMBER",
    "US_BANK_NUMBER": "IBAN",
    "MEDICAL_LICENSE": "ID_NUMBER",
    "CRYPTO": None,
    "AGE": "AGE",
}

# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def normalize_label(raw: str, mapping: dict[str, str | None]) -> str | None:
    """Map an external label to a taxonomy label, or return ``None`` to discard.

    Strips IOB/BIOES prefixes (``B-``, ``I-``, ``S-``, ``E-``) before lookup.
    Case-insensitive.

    Args:
        raw:     The raw label string from an external system.
        mapping: One of the ``*_TO_LABEL`` dicts in this module.

    Returns:
        A canonical taxonomy label, or ``None`` if the type should be
        discarded.

    Raises:
        KeyError: If ``raw`` (after stripping) is not in ``mapping``.
    """
    # Strip IOB/BIOES prefix
    for prefix in ("B-", "I-", "S-", "E-"):
        if raw.upper().startswith(prefix):
            raw = raw[2:]
            break
    key = raw.upper()
    if key not in mapping:
        raise KeyError(f"Unknown external label {raw!r}; add it to taxonomy.py")
    return mapping[key]


def label_precedence_rank(label: str) -> int:
    """Return the precedence rank for *label* (lower = higher priority).

    Labels not in ``LABEL_PRECEDENCE`` get the lowest priority (highest int).
    """
    try:
        return LABEL_PRECEDENCE.index(label)
    except ValueError:
        return len(LABEL_PRECEDENCE)

"""PII validators for regex-detected spans.

Each validator takes the matched text and returns ``True`` if it looks
like a genuine PII value, ``False`` otherwise.  Used to promote
confident matches (``validated=True, score=0.99``) or discard false
positives.
"""

from __future__ import annotations

import ipaddress
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


def email_validator(text: str) -> bool:
    """Check that *text* looks like a plausible email address."""
    # Very light check — the regex already matched the shape.
    # Reject if TLD is too short or domain has no dot.
    parts = text.rsplit("@", 1)
    if len(parts) != 2:
        return False
    local, domain = parts
    if not local or not domain:
        return False
    if "." not in domain:
        return False
    tld = domain.rsplit(".", 1)[-1]
    return len(tld) >= 2


# ---------------------------------------------------------------------------
# SSN (US Social Security Number)
# ---------------------------------------------------------------------------

# Area numbers that are never valid
_SSN_INVALID_AREAS = {"000", "666"}


def ssn_validator(text: str) -> bool:
    """Validate a US Social Security Number (with or without dashes)."""
    digits = re.sub(r"[\s\-]", "", text)
    if len(digits) != 9 or not digits.isdigit():
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in _SSN_INVALID_AREAS or area.startswith("9"):
        return False
    return not (group == "00" or serial == "0000")


# ---------------------------------------------------------------------------
# Credit card (Luhn algorithm)
# ---------------------------------------------------------------------------


def luhn_validator(text: str) -> bool:
    """Validate a credit card number using the Luhn algorithm."""
    digits = re.sub(r"[\s\-]", "", text)
    if not digits.isdigit() or len(digits) < 13 or len(digits) > 19:
        return False

    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ---------------------------------------------------------------------------
# IBAN
# ---------------------------------------------------------------------------


def iban_validator(text: str) -> bool:
    """Validate an IBAN using the mod-97 check digit algorithm."""
    iban = re.sub(r"\s", "", text).upper()
    if len(iban) < 15 or len(iban) > 34:
        return False
    if not iban[:2].isalpha() or not iban[2:4].isdigit():
        return False

    # Move first 4 chars to end, convert letters to numbers
    rearranged = iban[4:] + iban[:4]
    numeric = ""
    for ch in rearranged:
        if ch.isdigit():
            numeric += ch
        elif ch.isalpha():
            numeric += str(ord(ch) - ord("A") + 10)
        else:
            return False

    return int(numeric) % 97 == 1


# ---------------------------------------------------------------------------
# IP address
# ---------------------------------------------------------------------------


def ip_validator(text: str) -> bool:
    """Validate an IPv4 or IPv6 address."""
    try:
        addr = ipaddress.ip_address(text)
        # Reject loopback and link-local -- not PII
        return not (addr.is_loopback or addr.is_link_local)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%Y-%m-%d",
    "%m/%d/%y",
    "%d-%m-%y",
]


def date_validator(text: str) -> bool:
    """Check that *text* parses as a plausible date (1900-2099)."""
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(text.strip(), fmt)
            if 1900 <= dt.year <= 2099:
                return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Mapping from validator name (as used in YAML configs) to function.
VALIDATORS: dict[str, callable] = {
    "email_validator": email_validator,
    "ssn_validator": ssn_validator,
    "luhn_validator": luhn_validator,
    "iban_validator": iban_validator,
    "ip_validator": ip_validator,
    "date_validator": date_validator,
}

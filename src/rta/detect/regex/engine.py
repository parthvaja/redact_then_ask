"""Regex detection engine — loads rule packs and scans text.

The engine reads YAML rule packs from ``configs/rules/``, compiles
regex patterns once, and runs them against document text.  Matches
are optionally validated; validated hits get ``score=0.99`` and
``validated=True``.

For phones, uses ``phonenumbers.PhoneNumberMatcher`` instead of raw
regex (§6.5).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from rta.config import get_config
from rta.detect.regex.validators import VALIDATORS
from rta.schema import Document, Span

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegexRule:
    """A single compiled regex rule from a YAML rule pack."""

    id: str
    label: str
    pattern: re.Pattern[str]
    group: int
    score: float
    validator_name: str | None
    pack_id: str


# ---------------------------------------------------------------------------
# Rule pack loading
# ---------------------------------------------------------------------------

_compiled_cache: dict[str, list[RegexRule]] = {}


def _load_pack(pack_path: Path) -> list[RegexRule]:
    """Load and compile one YAML rule pack."""
    cache_key = str(pack_path)
    if cache_key in _compiled_cache:
        return _compiled_cache[cache_key]

    with pack_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    pack_id = data.get("id", pack_path.stem)
    rules: list[RegexRule] = []

    for p in data.get("patterns", []):
        try:
            compiled = re.compile(p["regex"])
        except re.error as exc:
            logger.warning("Skipping rule %s: bad regex: %s", p.get("id", "?"), exc)
            continue

        rules.append(
            RegexRule(
                id=p["id"],
                label=p["label"],
                pattern=compiled,
                group=p.get("group", 0),
                score=p.get("score", 0.7),
                validator_name=p.get("validator"),
                pack_id=pack_id,
            )
        )

    _compiled_cache[cache_key] = rules
    logger.debug("Loaded rule pack %s with %d rules from %s", pack_id, len(rules), pack_path)
    return rules


def load_packs(
    doc_type: str | None = None,
    doc_types: list[str] | None = None,
) -> list[RegexRule]:
    """Load the appropriate rule packs.

    Always loads ``common.yaml``.  If *doc_types* is given, loads packs for
    all listed types (deduplicated).  Otherwise falls back to the single
    *doc_type*.

    Args:
        doc_type:  Single predicted document type (e.g. ``"medical"``).
        doc_types: Multiple types from routing (overrides *doc_type*).

    Returns:
        Flattened list of all compiled rules.
    """
    cfg = get_config()
    rule_dir = Path(cfg.routing.rule_dir)

    all_rules: list[RegexRule] = []

    # Always load common
    common_path = rule_dir / "common.yaml"
    if common_path.exists():
        all_rules.extend(_load_pack(common_path))

    # Determine which type packs to load
    types_to_load: list[str] = []
    if doc_types is not None:
        types_to_load = doc_types
    elif doc_type:
        types_to_load = [doc_type]

    seen: set[str] = set()
    for dt in types_to_load:
        if dt not in seen:
            seen.add(dt)
            type_path = rule_dir / f"{dt}.yaml"
            if type_path.exists():
                all_rules.extend(_load_pack(type_path))
            else:
                logger.debug("No rule pack for doc_type=%s at %s", dt, type_path)

    return all_rules


def reset_cache() -> None:
    """Clear the compiled rule cache (for tests)."""
    _compiled_cache.clear()


# ---------------------------------------------------------------------------
# Phone detection via phonenumbers library
# ---------------------------------------------------------------------------


def _detect_phones(text: str) -> list[Span]:
    """Detect phone numbers using the ``phonenumbers`` library."""
    try:
        import phonenumbers
    except ImportError:
        logger.debug("phonenumbers not installed — skipping phone detection")
        return []

    spans: list[Span] = []
    for match in phonenumbers.PhoneNumberMatcher(text, "US"):
        start = match.start
        end = match.end
        spans.append(
            Span(
                start=start,
                end=end,
                label="PHONE",
                text=text[start:end],
                score=0.95,
                detector="regex:phone",
                validated=True,
            )
        )
    return spans


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------


def scan(doc: Document, rules: list[RegexRule]) -> list[Span]:
    """Run all *rules* against ``doc.text`` and return unresolved spans.

    Phones are detected separately via ``phonenumbers`` and merged in.
    """
    cfg = get_config()
    keep_unvalidated = cfg.detect.regex.keep_unvalidated
    text = doc.text
    spans: list[Span] = []

    for rule in rules:
        for m in rule.pattern.finditer(text):
            try:
                matched_text = m.group(rule.group)
                if rule.group == 0:
                    start = m.start()
                    end = m.end()
                else:
                    start = m.start(rule.group)
                    end = m.end(rule.group)
            except IndexError:
                continue

            if not matched_text or not matched_text.strip():
                continue

            # Validate if a validator is specified
            validated = False
            score = rule.score
            if rule.validator_name:
                validator_fn = VALIDATORS.get(rule.validator_name)
                if validator_fn:
                    if validator_fn(matched_text):
                        validated = True
                        score = 0.99
                    elif not keep_unvalidated:
                        logger.debug(
                            "Rule %s: dropping %r — failed %s",
                            rule.id,
                            matched_text,
                            rule.validator_name,
                        )
                        continue
                    # If keep_unvalidated, keep with base score

            spans.append(
                Span(
                    start=start,
                    end=end,
                    label=rule.label,
                    text=matched_text,
                    score=score,
                    detector=f"regex:{rule.pack_id}/{rule.id}",
                    validated=validated,
                )
            )

    # Phone detection via phonenumbers
    spans.extend(_detect_phones(text))

    return spans

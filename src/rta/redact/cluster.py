"""Mention clustering for redaction (Phase 8).

Groups coreferent PII mentions so they receive the same placeholder.
PERSON and ORG use fuzzy/initials/containment matching; other labels
cluster by exact normalised value.
"""

from __future__ import annotations

import re

from rta.config import get_config
from rta.schema import Span

TITLES = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "miss",
        "dr",
        "prof",
        "sir",
        "madam",
        "judge",
        "hon",
        "rev",
        "sgt",
        "cpl",
        "lt",
        "capt",
    }
)


def normalize_mention(text: str, label: str) -> str:
    """Normalise a mention for clustering comparisons."""
    if label in ("PHONE", "CREDIT_CARD", "SSN", "IBAN"):
        return re.sub(r"\D", "", text)

    text = text.lower().strip()
    words = text.split()
    words = [w for w in words if w.rstrip(".") not in TITLES]
    text = " ".join(words)
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _check_initials(short: str, full: str) -> bool:
    """Return True if *short* is an initials form of *full*.

    ``"j. smith"`` matches ``"john smith"``.
    """
    short_parts = short.replace(".", " ").split()
    full_parts = full.split()

    if not short_parts or len(short_parts) != len(full_parts):
        return False

    has_initial = False
    for s, f in zip(short_parts, full_parts, strict=False):
        if len(s) == 1:
            has_initial = True
            if s[0] != f[0]:
                return False
        elif s != f:
            return False

    return has_initial


def _is_initials_match(a: str, b: str) -> bool:
    return _check_initials(a, b) or _check_initials(b, a)


def cluster_spans(spans: list[Span]) -> list[list[Span]]:
    """Group coreferent spans into clusters.

    Returns a list of clusters, each cluster a list of ``Span`` objects.
    """
    if not spans:
        return []

    try:
        import importlib.util

        has_fuzz = importlib.util.find_spec("rapidfuzz") is not None
    except (ImportError, ValueError):
        has_fuzz = False

    cfg = get_config()
    fuzzy_threshold = cfg.redact.fuzzy_threshold

    clusters: list[list[Span]] = []
    # (label, list_of_normalised_forms) for each cluster
    cluster_meta: list[tuple[str, list[str]]] = []

    for span in spans:
        norm = normalize_mention(span.text, span.label)
        if not norm:
            clusters.append([span])
            cluster_meta.append((span.label, [norm]))
            continue

        matched_idx = _find_cluster(
            norm,
            span.label,
            cluster_meta,
            has_fuzz,
            fuzzy_threshold,
        )

        if matched_idx is not None:
            clusters[matched_idx].append(span)
            if norm not in cluster_meta[matched_idx][1]:
                cluster_meta[matched_idx][1].append(norm)
        else:
            clusters.append([span])
            cluster_meta.append((span.label, [norm]))

    return clusters


def _find_cluster(
    norm: str,
    label: str,
    cluster_meta: list[tuple[str, list[str]]],
    has_fuzz: bool,
    fuzzy_threshold: float,
) -> int | None:
    """Find the best matching cluster index for a normalised mention."""
    for i, (c_label, c_norms) in enumerate(cluster_meta):
        if c_label != label:
            continue

        # Exact match
        if norm in c_norms:
            return i

        if label in ("PERSON", "ORG"):
            norm_tokens = set(norm.split())
            # Token containment
            for cn in c_norms:
                cn_tokens = set(cn.split())
                if (
                    norm_tokens
                    and cn_tokens
                    and (norm_tokens <= cn_tokens or cn_tokens <= norm_tokens)
                ):
                    return i

            # Initials
            for cn in c_norms:
                if _is_initials_match(norm, cn):
                    return i

            # Fuzzy
            if has_fuzz:
                from rapidfuzz import fuzz

                for cn in c_norms:
                    if fuzz.token_set_ratio(norm, cn) >= fuzzy_threshold:
                        return i

    return None

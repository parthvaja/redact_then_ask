"""Language identification — ``Document`` → ISO language code + confidence.

Wraps the ``lingua`` language-detection library.  The detector is built
once from the candidate list in ``configs/default.yaml`` and cached at
module level (it is expensive to instantiate).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lingua import Language, LanguageDetectorBuilder

from rta.config import get_config

if TYPE_CHECKING:
    from lingua import LanguageDetector

from rta.schema import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ISO 639-1 code (lower) → lingua.Language mapping
# ---------------------------------------------------------------------------

_ISO_TO_LINGUA: dict[str, Language] = {
    "en": Language.ENGLISH,
    "es": Language.SPANISH,
    "fr": Language.FRENCH,
    "de": Language.GERMAN,
    "hi": Language.HINDI,
    "gu": Language.GUJARATI,
    "pt": Language.PORTUGUESE,
    "it": Language.ITALIAN,
    "nl": Language.DUTCH,
    "ru": Language.RUSSIAN,
    "zh": Language.CHINESE,
    "ja": Language.JAPANESE,
    "ko": Language.KOREAN,
    "ar": Language.ARABIC,
    "tr": Language.TURKISH,
    "pl": Language.POLISH,
    "sv": Language.SWEDISH,
    "da": Language.DANISH,
    "fi": Language.FINNISH,
    "nb": Language.BOKMAL,
    "uk": Language.UKRAINIAN,
    "cs": Language.CZECH,
    "ro": Language.ROMANIAN,
    "hu": Language.HUNGARIAN,
    "el": Language.GREEK,
    "he": Language.HEBREW,
    "th": Language.THAI,
    "vi": Language.VIETNAMESE,
    "id": Language.INDONESIAN,
    "ms": Language.MALAY,
    "bn": Language.BENGALI,
    "ta": Language.TAMIL,
    "te": Language.TELUGU,
    "mr": Language.MARATHI,
    "ur": Language.URDU,
}

# ---------------------------------------------------------------------------
# Cached detector
# ---------------------------------------------------------------------------

_detector: LanguageDetector | None = None


def _get_detector() -> LanguageDetector:
    """Build (or return the cached) lingua ``LanguageDetector``."""
    global _detector
    if _detector is not None:
        return _detector

    cfg = get_config()
    candidates: list[str] = cfg.langid.candidates  # e.g. ["en", "es", "fr", ...]

    languages: list[Language] = []
    for code in candidates:
        lang = _ISO_TO_LINGUA.get(code.lower())
        if lang is None:
            logger.warning("Unknown ISO code %r in langid.candidates — skipping", code)
            continue
        languages.append(lang)

    if not languages:
        raise ValueError("No valid languages in langid.candidates")

    _detector = LanguageDetectorBuilder.from_languages(*languages).build()
    logger.debug("Built lingua detector with %d candidates", len(languages))
    return _detector


def reset_detector() -> None:
    """Reset the cached detector.  Useful in tests."""
    global _detector
    _detector = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect(doc: Document) -> tuple[str, float]:
    """Detect the language of *doc*.

    Updates ``doc.meta`` in-place with the keys ``language``,
    ``lang_conf``, and (optionally) ``unsupported_language``.

    Returns:
        ``(iso_code, confidence)`` where ``iso_code`` is a lower-case
        2-letter ISO 639-1 code and ``confidence`` is in [0, 1].
    """
    cfg = get_config()
    detector = _get_detector()

    confidence_values = detector.compute_language_confidence_values(doc.text)

    if not confidence_values:
        doc.meta["language"] = "und"
        doc.meta["lang_conf"] = 0.0
        doc.meta["unsupported_language"] = True
        logger.warning("No language detected for doc %s", doc.doc_id)
        return ("und", 0.0)

    # confidence_values is sorted best-first by lingua.
    best = confidence_values[0]
    iso_code = best.language.iso_code_639_1.name.lower()
    conf = best.value

    doc.meta["language"] = iso_code
    doc.meta["lang_conf"] = conf

    if conf < cfg.langid.min_conf:
        doc.meta["unsupported_language"] = True
        logger.warning(
            "Low confidence %.2f for doc %s (detected %s, threshold %.2f)",
            conf,
            doc.doc_id,
            iso_code,
            cfg.langid.min_conf,
        )
    else:
        doc.meta.pop("unsupported_language", None)

    return (iso_code, conf)

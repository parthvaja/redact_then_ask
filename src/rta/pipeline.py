"""Pipeline orchestrator for Redact-Then-Ask.

``Pipeline`` wires every stage together and returns a ``PipelineResult``
holding all intermediate outputs.  Individual stages can be replaced via
*overrides* — this is how oracle injections work in the error-propagation
experiment.

This file is a skeleton in Phase 0.  Stages will be filled in as their
respective phases complete.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rta.config import Config, get_config
from rta.schema import Document, RedactionResult, Span

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """All outputs from a single pipeline run.

    Every field is populated after the corresponding stage completes.
    ``None`` means the stage has not run yet (e.g. evaluation is optional).
    """

    # Ingestion
    document: Document | None = None

    # Language ID
    language: str | None = None
    lang_conf: float | None = None

    # Classification
    doc_type: str | None = None
    type_probs: dict[str, float] = field(default_factory=dict)

    # Detection
    spans_regex: list[Span] = field(default_factory=list)
    spans_ner: list[Span] = field(default_factory=list)
    spans_merged: list[Span] = field(default_factory=list)  # after resolution

    # Redaction
    redaction: RedactionResult | None = None

    # Timing (seconds per stage)
    timings: dict[str, float] = field(default_factory=dict)

    # Errors / warnings
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Orchestrate the full Redact-Then-Ask pipeline.

    Args:
        cfg:       Configuration object (default: singleton from
                   ``rta.config.get_config()``).
        overrides: Mapping of ``stage_name → callable`` that replaces the
                   default stage implementation.  Used for oracle ablations::

                       Pipeline(cfg, overrides={"classify": oracle_classify})

                   Each override callable has the same signature as the
                   stage it replaces.

    Stages (in order):
        ingest, langid, classify, detect_regex, detect_ner, resolve, redact
    """

    def __init__(
        self,
        cfg: Config | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.overrides: dict[str, Any] = overrides or {}

    def run(self, path_or_text: Path | str) -> PipelineResult:
        """Run the full pipeline on *path_or_text*.

        Args:
            path_or_text: Either a file path (PDF/DOCX/TXT) or a plain
                          string.  File paths are ingested; strings are
                          wrapped in a ``Document`` directly.

        Returns:
            A ``PipelineResult`` with all intermediate outputs populated.

        Raises:
            NotImplementedError: Until stages are implemented in later phases.
        """
        result = PipelineResult()

        # --- Stage: ingest ---------------------------------------------------
        t0 = time.perf_counter()
        result.document = self._run_ingest(path_or_text)
        result.timings["ingest"] = time.perf_counter() - t0

        # --- Stage: langid ---------------------------------------------------
        t0 = time.perf_counter()
        result.language, result.lang_conf = self._run_langid(result.document)
        result.timings["langid"] = time.perf_counter() - t0

        # --- Stage: classify -------------------------------------------------
        t0 = time.perf_counter()
        result.doc_type, result.type_probs = self._run_classify(result.document)
        result.timings["classify"] = time.perf_counter() - t0

        # --- Stage: detect (regex + NER) ------------------------------------
        t0 = time.perf_counter()
        result.spans_regex = self._run_detect_regex(result.document, result.doc_type)
        result.timings["detect_regex"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        result.spans_ner = self._run_detect_ner(result.document)
        result.timings["detect_ner"] = time.perf_counter() - t0

        # --- Stage: resolve --------------------------------------------------
        t0 = time.perf_counter()
        result.spans_merged = self._run_resolve(result.spans_regex + result.spans_ner)
        result.timings["resolve"] = time.perf_counter() - t0

        # --- Stage: redact ---------------------------------------------------
        t0 = time.perf_counter()
        result.redaction = self._run_redact(result.document, result.spans_merged)
        result.timings["redact"] = time.perf_counter() - t0

        total = sum(result.timings.values())
        logger.info(
            "Pipeline complete in %.2f s — %d spans → %d after resolution",
            total,
            len(result.spans_regex) + len(result.spans_ner),
            len(result.spans_merged),
        )
        return result

    # ------------------------------------------------------------------
    # Private stage dispatchers
    # ------------------------------------------------------------------

    def _run_ingest(self, path_or_text: Path | str) -> Document:
        if "ingest" in self.overrides:
            return self.overrides["ingest"](path_or_text)
        # Import lazily so the package is importable before deps are installed
        from rta.ingest import load

        return load(path_or_text)

    def _run_langid(self, doc: Document) -> tuple[str, float]:
        if "langid" in self.overrides:
            return self.overrides["langid"](doc)
        from rta.langid import detect

        return detect(doc)

    def _run_classify(self, doc: Document) -> tuple[str, dict[str, float]]:
        if "classify" in self.overrides:
            return self.overrides["classify"](doc)
        from rta.classify import predict

        return predict(doc)

    def _run_detect_regex(self, doc: Document, doc_type: str | None) -> list[Span]:
        if "detect_regex" in self.overrides:
            return self.overrides["detect_regex"](doc, doc_type)
        from rta.detect.regex import run as regex_run

        return regex_run(doc, doc_type)

    def _run_detect_ner(self, doc: Document) -> list[Span]:
        if "detect_ner" in self.overrides:
            return self.overrides["detect_ner"](doc)
        from rta.detect.ner import run as ner_run

        return ner_run(doc)

    def _run_resolve(self, spans: list[Span]) -> list[Span]:
        if "resolve" in self.overrides:
            return self.overrides["resolve"](spans)
        from rta.resolve import merge

        return merge(spans)

    def _run_redact(self, doc: Document, spans: list[Span]) -> RedactionResult:
        if "redact" in self.overrides:
            return self.overrides["redact"](doc, spans)
        from rta.redact import apply

        return apply(doc, spans)

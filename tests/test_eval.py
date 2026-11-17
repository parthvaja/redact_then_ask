"""Tests for rta.eval — evaluation harness (Phase 9).

Gate criteria:
- match_strict: exact match TP, misaligned FP, missing FN.
- match_overlap: overlapping match counts as TP; same label required.
- match_token: character-level counts.
- precision_recall_f1: basic computation and zero division.
- f_beta: F2 score.
- per_label_metrics: multi-label breakdown.
- leak_rate: leaked text found in redacted; no leaks = 0.0.
- bootstrap_ci: returns (lower, upper) tuple.
- generate_report: writes CSV and MD files.
- score() integration with overlap mode.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rta.eval import score
from rta.eval.match import match_overlap, match_strict, match_token
from rta.eval.metrics import (
    bootstrap_ci,
    f_beta,
    leak_rate,
    per_label_metrics,
    precision_recall_f1,
)
from rta.eval.report import generate_report
from rta.schema import Span

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _span(
    start: int,
    end: int,
    label: str,
    text: str = "",
    score: float = 0.9,
    detector: str = "test",
) -> Span:
    return Span(
        start=start,
        end=end,
        label=label,
        text=text or "x" * (end - start),
        score=score,
        detector=detector,
    )


# ---------------------------------------------------------------------------
# match_strict
# ---------------------------------------------------------------------------


class TestMatchStrict:
    def test_exact_match_tp(self) -> None:
        pred = [_span(0, 5, "PERSON", "Alice")]
        gold = [_span(0, 5, "PERSON", "Alice")]
        tp, fp, fn = match_strict(pred, gold)
        assert len(tp) == 1
        assert len(fp) == 0
        assert len(fn) == 0

    def test_misaligned_fp(self) -> None:
        pred = [_span(0, 6, "PERSON", "Alice!")]
        gold = [_span(0, 5, "PERSON", "Alice")]
        tp, fp, fn = match_strict(pred, gold)
        assert len(tp) == 0
        assert len(fp) == 1
        assert len(fn) == 1

    def test_missing_fn(self) -> None:
        pred: list[Span] = []
        gold = [_span(0, 5, "PERSON", "Alice")]
        tp, fp, fn = match_strict(pred, gold)
        assert len(tp) == 0
        assert len(fp) == 0
        assert len(fn) == 1

    def test_different_label_no_match(self) -> None:
        pred = [_span(0, 5, "ORG", "Alice")]
        gold = [_span(0, 5, "PERSON", "Alice")]
        tp, fp, fn = match_strict(pred, gold)
        assert len(tp) == 0
        assert len(fp) == 1
        assert len(fn) == 1


# ---------------------------------------------------------------------------
# match_overlap
# ---------------------------------------------------------------------------


class TestMatchOverlap:
    def test_overlapping_match_tp(self) -> None:
        pred = [_span(0, 6, "PERSON", "Alice!")]
        gold = [_span(0, 5, "PERSON", "Alice")]
        tp, fp, fn = match_overlap(pred, gold)
        assert len(tp) == 1
        assert len(fp) == 0
        assert len(fn) == 0

    def test_same_label_required(self) -> None:
        pred = [_span(0, 6, "ORG", "Alice!")]
        gold = [_span(0, 5, "PERSON", "Alice")]
        tp, fp, fn = match_overlap(pred, gold)
        assert len(tp) == 0
        assert len(fp) == 1
        assert len(fn) == 1

    def test_no_overlap(self) -> None:
        pred = [_span(10, 15, "PERSON", "Bobby")]
        gold = [_span(0, 5, "PERSON", "Alice")]
        tp, fp, fn = match_overlap(pred, gold)
        assert len(tp) == 0
        assert len(fp) == 1
        assert len(fn) == 1


# ---------------------------------------------------------------------------
# match_token
# ---------------------------------------------------------------------------


class TestMatchToken:
    def test_character_level_counts(self) -> None:
        # Pred covers chars 0-9, gold covers chars 5-14
        pred = [_span(0, 10, "PERSON", "0123456789")]
        gold = [_span(5, 15, "PERSON", "5678901234")]
        tp, fp, fn = match_token(pred, gold)
        # Overlap is chars 5-9 (5 chars)
        assert tp == 5
        assert fp == 5  # chars 0-4 only in pred
        assert fn == 5  # chars 10-14 only in gold

    def test_perfect_match(self) -> None:
        pred = [_span(0, 5, "PERSON", "Alice")]
        gold = [_span(0, 5, "PERSON", "Alice")]
        tp, fp, fn = match_token(pred, gold)
        assert tp == 5
        assert fp == 0
        assert fn == 0

    def test_empty_pred(self) -> None:
        pred: list[Span] = []
        gold = [_span(0, 5, "PERSON", "Alice")]
        tp, fp, fn = match_token(pred, gold)
        assert tp == 0
        assert fp == 0
        assert fn == 5


# ---------------------------------------------------------------------------
# precision_recall_f1
# ---------------------------------------------------------------------------


class TestPrecisionRecallF1:
    def test_basic_computation(self) -> None:
        p, r, f1 = precision_recall_f1(tp=80, fp=20, fn=10)
        assert p == pytest.approx(0.8)  # 80 / (80+20)
        assert r == pytest.approx(80 / 90)
        expected_f1 = 2 * 0.8 * (80 / 90) / (0.8 + 80 / 90)
        assert f1 == pytest.approx(expected_f1)

    def test_zero_division_returns_zeros(self) -> None:
        p, r, f1 = precision_recall_f1(tp=0, fp=0, fn=0)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_no_tp(self) -> None:
        p, r, f1 = precision_recall_f1(tp=0, fp=5, fn=3)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0


# ---------------------------------------------------------------------------
# f_beta
# ---------------------------------------------------------------------------


class TestFBeta:
    def test_f2_score(self) -> None:
        # F2 weights recall higher
        p, r = 0.8, 0.6
        f2 = f_beta(p, r, beta=2.0)
        # (1 + 4) * 0.8 * 0.6 / (4 * 0.8 + 0.6) = 5 * 0.48 / 3.8
        expected = 5 * 0.48 / 3.8
        assert f2 == pytest.approx(expected)

    def test_f2_zero_division(self) -> None:
        assert f_beta(0.0, 0.0, beta=2.0) == 0.0

    def test_f1_via_fbeta(self) -> None:
        p, r = 0.8, 0.6
        f1_val = f_beta(p, r, beta=1.0)
        expected = 2 * p * r / (p + r)
        assert f1_val == pytest.approx(expected)


# ---------------------------------------------------------------------------
# per_label_metrics
# ---------------------------------------------------------------------------


class TestPerLabelMetrics:
    def test_multi_label(self) -> None:
        tp = [
            _span(0, 5, "PERSON", "Alice"),
            _span(10, 20, "EMAIL", "a@test.com"),
        ]
        fp = [_span(30, 35, "PERSON", "Bogus")]
        fn = [_span(40, 45, "PHONE", "12345")]
        result = per_label_metrics(tp, fp, fn)

        # Per-label keys
        assert "PERSON" in result
        assert "EMAIL" in result
        assert "PHONE" in result

        # PERSON: 1 TP, 1 FP, 0 FN -> P=0.5, R=1.0
        assert result["PERSON"]["tp"] == 1
        assert result["PERSON"]["fp"] == 1
        assert result["PERSON"]["fn"] == 0
        assert result["PERSON"]["precision"] == pytest.approx(0.5)
        assert result["PERSON"]["recall"] == pytest.approx(1.0)

        # EMAIL: 1 TP, 0 FP, 0 FN -> P=1.0, R=1.0, F1=1.0
        assert result["EMAIL"]["precision"] == pytest.approx(1.0)
        assert result["EMAIL"]["f1"] == pytest.approx(1.0)

        # PHONE: 0 TP, 0 FP, 1 FN -> R=0.0
        assert result["PHONE"]["recall"] == pytest.approx(0.0)

        # Aggregates
        assert "micro" in result
        assert "macro" in result
        assert result["micro"]["tp"] == 2
        assert result["micro"]["fp"] == 1
        assert result["micro"]["fn"] == 1


# ---------------------------------------------------------------------------
# leak_rate
# ---------------------------------------------------------------------------


class TestLeakRate:
    def test_leaked_text_found(self) -> None:
        original = "Hello Alice and Bob."
        # Only "Alice" redacted, "Bob" leaked
        redacted = "Hello [PERSON_1] and Bob."
        gold_spans = [
            _span(6, 11, "PERSON", "Alice"),
            _span(16, 19, "PERSON", "Bob"),
        ]
        rate = leak_rate(gold_spans, redacted, original)
        # Total PII chars: 5 (Alice) + 3 (Bob) = 8
        # Leaked: 3 (Bob)
        assert rate == pytest.approx(3 / 8)

    def test_no_leaks(self) -> None:
        original = "Hello Alice."
        redacted = "Hello [PERSON_1]."
        gold_spans = [_span(6, 11, "PERSON", "Alice")]
        rate = leak_rate(gold_spans, redacted, original)
        assert rate == pytest.approx(0.0)

    def test_no_gold_spans(self) -> None:
        rate = leak_rate([], "some text", "some text")
        assert rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_returns_lower_upper_tuple(self) -> None:
        scores = [0.8, 0.85, 0.9, 0.78, 0.82]
        lower, upper = bootstrap_ci(scores)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= upper

    def test_deterministic_with_seed(self) -> None:
        scores = [0.7, 0.75, 0.8, 0.85, 0.9]
        ci1 = bootstrap_ci(scores, seed=42)
        ci2 = bootstrap_ci(scores, seed=42)
        assert ci1 == ci2

    def test_empty_scores(self) -> None:
        lower, upper = bootstrap_ci([])
        assert lower == 0.0
        assert upper == 0.0

    def test_single_score(self) -> None:
        lower, upper = bootstrap_ci([0.5])
        # With a single value, all bootstrap samples are the same
        assert lower == pytest.approx(0.5)
        assert upper == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_writes_csv_and_md(self, tmp_path: Path) -> None:
        results = {
            "PERSON": {
                "precision": 0.8,
                "recall": 0.9,
                "f1": 0.8471,
                "f2": 0.88,
                "tp": 9,
                "fp": 2,
                "fn": 1,
            },
            "micro": {
                "precision": 0.8,
                "recall": 0.9,
                "f1": 0.8471,
                "f2": 0.88,
                "tp": 9,
                "fp": 2,
                "fn": 1,
            },
        }
        csv_path, md_path = generate_report(
            results,
            "test_system",
            "test_dataset",
            tmp_path,
        )
        assert csv_path.exists()
        assert md_path.exists()
        assert csv_path.suffix == ".csv"
        assert md_path.suffix == ".md"

        csv_text = csv_path.read_text(encoding="utf-8")
        assert "PERSON" in csv_text
        assert "precision" in csv_text

        md_text = md_path.read_text(encoding="utf-8")
        assert "PERSON" in md_text
        assert "test_system" in md_text

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        output = tmp_path / "nested" / "reports"
        results = {
            "micro": {
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "f2": 0.5,
                "tp": 5,
                "fp": 5,
                "fn": 5,
            },
        }
        csv_path, _md_path = generate_report(results, "sys", "ds", output)
        assert output.exists()
        assert csv_path.exists()


# ---------------------------------------------------------------------------
# score() — integration
# ---------------------------------------------------------------------------


class TestScoreIntegration:
    def test_overlap_mode(self) -> None:
        pred = [_span(0, 6, "PERSON", "Alice!")]
        gold = [_span(0, 5, "PERSON", "Alice")]
        result = score(pred, gold, mode="overlap")
        assert "PERSON" in result
        assert "micro" in result
        assert result["micro"]["tp"] == 1
        assert result["micro"]["fp"] == 0
        assert result["micro"]["fn"] == 0

    def test_strict_mode(self) -> None:
        pred = [_span(0, 5, "PERSON", "Alice")]
        gold = [_span(0, 5, "PERSON", "Alice")]
        result = score(pred, gold, mode="strict")
        assert result["micro"]["tp"] == 1

    def test_token_mode(self) -> None:
        pred = [_span(0, 10, "PERSON", "0123456789")]
        gold = [_span(5, 15, "PERSON", "5678901234")]
        result = score(pred, gold, mode="token")
        assert "ALL" in result
        assert result["ALL"]["tp"] == 5

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown matching mode"):
            score([], [], mode="bogus")

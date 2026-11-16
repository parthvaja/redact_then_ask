"""Tests for rta.classify — document classifier (Phase 3).

Tests use synthetic data and mock models so they don't require network
access, real datasets, or a pre-trained model file.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rta.config import reset


@pytest.fixture(autouse=True)
def _fresh_config() -> None:
    reset()
    yield  # type: ignore[misc]
    reset()


# ---------------------------------------------------------------------------
# Helpers — synthetic corpus
# ---------------------------------------------------------------------------

_CLASSES = ["resume", "contract", "medical", "email", "court_case"]

# Short but distinctive texts per class
_CLASS_TEXTS: dict[str, list[str]] = {
    "resume": [
        "Experienced software engineer with 10 years in Python and Java development.",
        "Education: BS Computer Science, MIT. Skills: leadership, data analysis.",
        "Work history includes positions at Google and Microsoft as senior developer.",
        "Objective: Seeking a data science role leveraging my machine learning expertise.",
        "Certifications: AWS Solutions Architect, PMP certified project manager.",
        "Professional summary with experience in agile methodologies and DevOps.",
        "References available upon request. Strong background in cloud computing.",
        "Led cross-functional team of 12 engineers to deliver microservices platform.",
        "Published three peer-reviewed papers on distributed systems architecture.",
        "Graduated summa cum laude with expertise in algorithms and data structures.",
    ],
    "contract": [
        "This agreement is entered into between Party A and Party B for services.",
        "The term of this contract shall commence on January 1 and terminate December 31.",
        "Indemnification clause: each party shall hold harmless the other party.",
        "Payment terms: net 30 days from invoice date for all delivered services.",
        "Force majeure clause covers acts of God, war, and government restrictions.",
        "Confidentiality obligations survive termination for a period of two years.",
        "The parties agree to binding arbitration in the event of any dispute.",
        "Licensee shall not sublicense or assign rights without prior written consent.",
        "Warranties: seller represents that goods conform to specifications herein.",
        "Governing law: this agreement shall be governed by the laws of Delaware.",
    ],
    "medical": [
        "Patient presents with chest pain radiating to left arm, onset two hours ago.",
        "Diagnosis: Type 2 diabetes mellitus with peripheral neuropathy complications.",
        "Prescribed metformin 500mg twice daily with meals for blood sugar management.",
        "MRI results show a 2cm lesion in the right frontal lobe requiring biopsy.",
        "Post-operative follow-up: wound healing well, no signs of infection noted.",
        "Lab results: HbA1c 7.2%, fasting glucose 145 mg/dL, creatinine normal.",
        "History of hypertension managed with lisinopril 10mg daily for five years.",
        "Colonoscopy revealed two benign polyps removed during the procedure today.",
        "Physical therapy recommended three times weekly for rotator cuff injury.",
        "Allergies: penicillin, sulfa drugs. No known food allergies documented.",
    ],
    "email": [
        "Hi John, can we schedule a meeting for tomorrow at 3pm to discuss the project?",
        "Please find attached the quarterly report. Let me know if you have questions.",
        "Thanks for the update. I'll review the proposal and get back to you by Friday.",
        "FYI the deadline has been moved to next Monday. Please adjust accordingly.",
        "Great work on the presentation! The client was very impressed with the demo.",
        "Reminder: team lunch this Thursday at noon in the main conference room.",
        "Could you forward the budget spreadsheet? I need it for the board meeting.",
        "Happy to help with the migration. Let's sync up after standup tomorrow morning.",
        "Apologies for the late reply. I was out of office last week on vacation.",
        "Looping in Sarah from marketing. She can provide the campaign metrics we need.",
    ],
    "court_case": [
        "The applicant alleged a violation of Article 3 of the Convention by the State.",
        "The Court finds that the respondent Government failed to protect the rights.",
        "The judgment was delivered at a public hearing in Strasbourg on 15 March 2020.",
        "The applicant was detained in custody pending trial for charges of fraud.",
        "The European Court of Human Rights ruled unanimously in favour of the applicant.",
        "Pursuant to Article 41, the Court awards the applicant 10,000 euros in damages.",
        "The respondent State raised preliminary objections regarding exhaustion of remedies.",
        "Witness testimony corroborated the applicant's account of events during detention.",
        "The dissenting opinion argued that the margin of appreciation should apply here.",
        "The Grand Chamber upheld the lower chamber's finding of a procedural violation.",
    ],
}


def _write_corpus(tmp_path: Path, n_per_class: int = 10) -> Path:
    """Write a synthetic corpus JSONL file and return its path."""
    corpus_path = tmp_path / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as fh:
        for label, texts in _CLASS_TEXTS.items():
            for i, text in enumerate(texts[:n_per_class]):
                rec = {"doc_id": f"{label}_{i}", "text": text, "label": label}
                fh.write(json.dumps(rec) + "\n")
    return corpus_path


# ---------------------------------------------------------------------------
# corpus.strip_artifacts
# ---------------------------------------------------------------------------


class TestStripArtifacts:
    def test_strips_enron_headers(self) -> None:
        from rta.classify.corpus import strip_artifacts

        text = "From: john@enron.com\nSubject: Meeting\nTo: jane@enron.com\n\nHi Jane,"
        cleaned = strip_artifacts(text, "email")
        assert "From:" not in cleaned
        assert "Subject:" not in cleaned
        assert "Hi Jane," in cleaned

    def test_strips_cuad_boilerplate(self) -> None:
        from rta.classify.corpus import strip_artifacts

        text = "Exhibit 10\n\nThis Agreement is made between A and B.\n\nTerms follow."
        cleaned = strip_artifacts(text, "contract")
        assert "Exhibit 10" not in cleaned
        assert "Terms follow." in cleaned

    def test_no_strip_for_other_types(self) -> None:
        from rta.classify.corpus import strip_artifacts

        text = "From: doctor@hospital.com\nPatient records attached."
        cleaned = strip_artifacts(text, "medical")
        # Medical texts should not have email headers stripped
        assert "From: doctor@hospital.com" in cleaned

    def test_collapses_blank_lines(self) -> None:
        from rta.classify.corpus import strip_artifacts

        text = "Subject: Test\n\n\n\n\nBody text here."
        cleaned = strip_artifacts(text, "email")
        assert "\n\n\n" not in cleaned


# ---------------------------------------------------------------------------
# baseline.train_model + predict_text
# ---------------------------------------------------------------------------


class TestBaselineTrain:
    def test_train_and_predict(self, tmp_path: Path) -> None:
        """Train on synthetic data, then predict."""
        from rta.classify.baseline import predict_text, reset_model_cache, train_model

        corpus_path = _write_corpus(tmp_path)
        model_path = tmp_path / "model.joblib"

        with patch("rta.config.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.seed = 13
            cfg.classify.path = str(model_path)
            cfg.classify.tfidf.ngram_range = [1, 2]
            cfg.classify.tfidf.min_df = 1  # low for small corpus
            cfg.classify.tfidf.sublinear_tf = True
            cfg.classify.tfidf.max_features = 5000
            cfg.classify.logistic_regression.C = 1.0
            cfg.classify.logistic_regression.max_iter = 1000
            cfg.classify.logistic_regression.solver = "lbfgs"
            cfg.classify.logistic_regression.multi_class = "multinomial"
            cfg.paths.data = str(tmp_path / "data")
            mock_cfg.return_value = cfg

            out = train_model(corpus_path)
            assert out.exists()

            # Meta file should exist
            meta_path = out.with_suffix(".meta.json")
            assert meta_path.exists()
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            assert set(meta["labels"]) == set(_CLASSES)

            # Predict
            reset_model_cache()
            doc_type, probs = predict_text("The applicant was detained in custody pending trial.")
            assert doc_type in _CLASSES
            assert set(probs.keys()) == set(_CLASSES)
            assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_train_missing_corpus_raises(self, tmp_path: Path) -> None:
        from rta.classify.baseline import train_model

        with patch("rta.config.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.paths.data = str(tmp_path / "data")
            mock_cfg.return_value = cfg

            with pytest.raises(FileNotFoundError, match="corpus not found"):
                train_model(tmp_path / "nonexistent.jsonl")

    def test_predict_missing_model_raises(self, tmp_path: Path) -> None:
        from rta.classify.baseline import predict_text, reset_model_cache

        reset_model_cache()
        with patch("rta.classify.baseline.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.classify.path = str(tmp_path / "no_model.joblib")
            mock_cfg.return_value = cfg

            with pytest.raises(FileNotFoundError, match="Trained classifier not found"):
                predict_text("some text")


class TestBaselineEvaluate:
    def test_evaluate_returns_report(self, tmp_path: Path) -> None:
        """Train, then evaluate — should return a report dict."""
        from rta.classify.baseline import evaluate_model, reset_model_cache, train_model

        corpus_path = _write_corpus(tmp_path)
        model_path = tmp_path / "model.joblib"

        with patch("rta.config.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.seed = 13
            cfg.classify.path = str(model_path)
            cfg.classify.tfidf.ngram_range = [1, 2]
            cfg.classify.tfidf.min_df = 1
            cfg.classify.tfidf.sublinear_tf = True
            cfg.classify.tfidf.max_features = 5000
            cfg.classify.logistic_regression.C = 1.0
            cfg.classify.logistic_regression.max_iter = 1000
            cfg.classify.logistic_regression.solver = "lbfgs"
            cfg.classify.logistic_regression.multi_class = "multinomial"
            cfg.paths.data = str(tmp_path / "data")
            mock_cfg.return_value = cfg

            train_model(corpus_path)

            reset_model_cache()
            report = evaluate_model(corpus_path)

        assert isinstance(report, dict)
        assert "accuracy" in report


# ---------------------------------------------------------------------------
# classify.predict (top-level wrapper)
# ---------------------------------------------------------------------------


class TestClassifyPredict:
    def test_predict_sets_doc_meta(self, tmp_path: Path) -> None:
        """classify.predict() should populate doc.meta."""
        from rta.classify import predict
        from rta.schema import Document

        doc = Document(doc_id="test01", text="Patient presents with chest pain.", source_path=None)

        with patch("rta.classify.baseline.predict_text") as mock_pred:
            mock_pred.return_value = (
                "medical",
                {
                    "medical": 0.8,
                    "email": 0.1,
                    "resume": 0.05,
                    "contract": 0.03,
                    "court_case": 0.02,
                },
            )
            doc_type, probs = predict(doc)

        assert doc_type == "medical"
        assert doc.meta["doc_type"] == "medical"
        assert doc.meta["type_probs"]["medical"] == 0.8
        assert set(probs.keys()) == set(_CLASSES)


# ---------------------------------------------------------------------------
# classify.train (integration via __init__)
# ---------------------------------------------------------------------------


class TestClassifyTrain:
    def test_train_builds_corpus_if_missing(self, tmp_path: Path) -> None:
        """train() should call build_corpus if corpus file doesn't exist."""
        from rta.classify import train

        with (
            patch("rta.config.get_config") as mock_cfg,
            patch("rta.classify.corpus.build_corpus") as mock_build,
            patch("rta.classify.baseline.train_model") as mock_train,
        ):
            cfg = MagicMock()
            cfg.paths.data = str(tmp_path / "data")
            mock_cfg.return_value = cfg

            # Corpus doesn't exist → should trigger build_corpus
            mock_build.return_value = (
                tmp_path / "data" / "processed" / "classify_corpus_stripped.jsonl"
            )
            # After build_corpus, train_model is called — make it succeed
            mock_train.return_value = tmp_path / "model.joblib"

            train()

            mock_build.assert_called_once()
            mock_train.assert_called_once()


# ---------------------------------------------------------------------------
# corpus._load_court_cases (local data, no HF)
# ---------------------------------------------------------------------------


class TestCorpusCourtCases:
    def test_loads_from_tab_json(self, tmp_path: Path) -> None:
        from rta.classify.corpus import _load_court_cases

        raw_dir = tmp_path / "data" / "raw" / "tab"
        raw_dir.mkdir(parents=True)

        tab_docs = [
            {"doc_id": f"001-{i:05d}", "text": f"Court case text number {i}. " * 20}
            for i in range(5)
        ]
        (raw_dir / "echr_test.json").write_text(json.dumps(tab_docs), encoding="utf-8")

        with patch("rta.classify.corpus.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.paths.data = str(tmp_path / "data")
            mock_cfg.return_value = cfg

            docs = _load_court_cases(max_docs=3)

        assert len(docs) == 3
        assert all(d["label"] == "court_case" for d in docs)

    def test_skips_short_texts(self, tmp_path: Path) -> None:
        from rta.classify.corpus import _load_court_cases

        raw_dir = tmp_path / "data" / "raw" / "tab"
        raw_dir.mkdir(parents=True)

        tab_docs = [
            {"doc_id": "short1", "text": "Too short."},
            {"doc_id": "long1", "text": "A sufficiently long court case text. " * 10},
        ]
        (raw_dir / "echr_test.json").write_text(json.dumps(tab_docs), encoding="utf-8")

        with patch("rta.classify.corpus.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.paths.data = str(tmp_path / "data")
            mock_cfg.return_value = cfg

            docs = _load_court_cases(max_docs=10)

        assert len(docs) == 1
        assert docs[0]["doc_id"] == "long1"


# ---------------------------------------------------------------------------
# corpus.build_corpus (mocked loaders)
# ---------------------------------------------------------------------------


class TestBuildCorpus:
    def test_build_corpus_writes_jsonl(self, tmp_path: Path) -> None:
        from rta.classify.corpus import build_corpus

        # Mock all HF loaders to return synthetic data
        mock_docs = {
            "_load_resumes": [{"doc_id": "r0", "text": "Resume text " * 20, "label": "resume"}],
            "_load_contracts": [
                {"doc_id": "c0", "text": "Contract text " * 20, "label": "contract"}
            ],
            "_load_medical": [{"doc_id": "m0", "text": "Medical text " * 20, "label": "medical"}],
            "_load_emails": [{"doc_id": "e0", "text": "Email text " * 20, "label": "email"}],
            "_load_court_cases": [
                {"doc_id": "cc0", "text": "Court text " * 20, "label": "court_case"}
            ],
        }

        with (
            patch("rta.config.get_config") as mock_cfg,
            patch("rta.classify.corpus._load_resumes", return_value=mock_docs["_load_resumes"]),
            patch("rta.classify.corpus._load_contracts", return_value=mock_docs["_load_contracts"]),
            patch("rta.classify.corpus._load_medical", return_value=mock_docs["_load_medical"]),
            patch("rta.classify.corpus._load_emails", return_value=mock_docs["_load_emails"]),
            patch(
                "rta.classify.corpus._load_court_cases", return_value=mock_docs["_load_court_cases"]
            ),
        ):
            cfg = MagicMock()
            cfg.paths.data = str(tmp_path / "data")
            mock_cfg.return_value = cfg

            out = build_corpus(max_per_class=1, strip=True)

        assert out.exists()
        with out.open(encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        assert len(lines) == 5
        labels = {line["label"] for line in lines}
        assert labels == set(_CLASSES)

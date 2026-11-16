"""TF-IDF + Logistic Regression document classifier.

Train/save/load/predict cycle for the baseline document-type classifier.
Configuration comes from ``configs/default.yaml`` → ``classify.*``.

Model is persisted with ``joblib`` to the path given by ``classify.path``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline

from rta.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_corpus(corpus_path: Path) -> tuple[list[str], list[str]]:
    """Read the JSONL corpus and return (texts, labels)."""
    texts: list[str] = []
    labels: list[str] = []
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            texts.append(rec["text"])
            labels.append(rec["label"])
    return texts, labels


def _build_pipeline(cfg: object) -> SkPipeline:
    """Build the sklearn Pipeline from config settings."""
    tfidf_cfg = cfg.classify.tfidf
    lr_cfg = cfg.classify.logistic_regression

    ngram_range = tuple(tfidf_cfg.ngram_range)

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=tfidf_cfg.min_df,
        sublinear_tf=tfidf_cfg.sublinear_tf,
        max_features=tfidf_cfg.max_features,
    )

    # multi_class was removed in sklearn ≥1.7; only pass params that
    # LogisticRegression actually accepts.
    import inspect

    lr_params: dict = {
        "C": lr_cfg.C,
        "max_iter": lr_cfg.max_iter,
        "solver": lr_cfg.solver,
        "random_state": cfg.seed,
    }
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        lr_params["multi_class"] = lr_cfg.multi_class

    clf = LogisticRegression(**lr_params)

    return SkPipeline([("tfidf", vectorizer), ("clf", clf)])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_model(corpus_path: Path | None = None) -> Path:
    """Train the TF-IDF + LR classifier and save it.

    Args:
        corpus_path: Path to the corpus JSONL.  If ``None``, uses
                     ``data/processed/classify_corpus_stripped.jsonl``.

    Returns:
        Path to the saved model file.
    """
    cfg = get_config()
    if corpus_path is None:
        corpus_path = Path(cfg.paths.data) / "processed" / "classify_corpus_stripped.jsonl"

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Classification corpus not found at {corpus_path}. "
            "Run `rta classify build-corpus` or `rta.classify.corpus.build_corpus()` first."
        )

    texts, labels = _load_corpus(corpus_path)
    logger.info("Loaded %d documents from %s", len(texts), corpus_path)

    # Stratified train/val/test split: 70/15/15
    seed = cfg.seed
    X_train, X_rest, y_train, y_rest = train_test_split(
        texts, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, random_state=seed, stratify=y_rest
    )

    logger.info(
        "Split: %d train / %d val / %d test",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    pipe = _build_pipeline(cfg)
    logger.info("Training TF-IDF + LR classifier ...")
    pipe.fit(X_train, y_train)

    # Validation report
    y_val_pred = pipe.predict(X_val)
    logger.info(
        "Validation report:\n%s",
        classification_report(y_val, y_val_pred, zero_division=0),
    )

    # Test report
    y_test_pred = pipe.predict(X_test)
    logger.info(
        "Test report:\n%s",
        classification_report(y_test, y_test_pred, zero_division=0),
    )

    # Save model
    model_path = Path(cfg.classify.path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    logger.info("Model saved to %s", model_path)

    # Save the split indices and test report for reproducibility
    meta_path = model_path.with_suffix(".meta.json")
    meta = {
        "corpus_path": str(corpus_path),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "seed": seed,
        "labels": sorted(set(labels)),
        "test_report": classification_report(
            y_test, y_test_pred, output_dict=True, zero_division=0
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("Metadata saved to %s", meta_path)

    return model_path


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_model_cache: SkPipeline | None = None


def _load_model() -> SkPipeline:
    """Load the trained model (cached after first load)."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    cfg = get_config()
    model_path = Path(cfg.classify.path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained classifier not found at {model_path}. Run `rta classify train` first."
        )
    _model_cache = joblib.load(model_path)
    logger.info("Loaded classifier from %s", model_path)
    return _model_cache


def reset_model_cache() -> None:
    """Clear the cached model (for tests)."""
    global _model_cache
    _model_cache = None


def predict_text(text: str) -> tuple[str, dict[str, float]]:
    """Predict document type from raw text.

    Returns:
        ``(doc_type, probs)`` where ``probs`` maps each class label to
        its predicted probability.
    """
    pipe = _load_model()
    probs_array: np.ndarray = pipe.predict_proba([text])[0]
    classes: list[str] = list(pipe.classes_)
    probs = {cls: float(p) for cls, p in zip(classes, probs_array, strict=False)}
    doc_type = max(probs, key=probs.get)  # type: ignore[arg-type]
    return doc_type, probs


def evaluate_model(corpus_path: Path | None = None) -> dict:
    """Evaluate the trained classifier on the held-out test set.

    If a corpus is given, splits it the same way as training (seed 13)
    and evaluates on the test portion.

    Returns:
        The classification report as a dict.
    """
    cfg = get_config()
    if corpus_path is None:
        corpus_path = Path(cfg.paths.data) / "processed" / "classify_corpus_stripped.jsonl"

    texts, labels = _load_corpus(corpus_path)
    seed = cfg.seed

    # Reproduce the same split
    _, X_rest, _, y_rest = train_test_split(
        texts, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    _, X_test, _, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, random_state=seed, stratify=y_rest
    )

    pipe = _load_model()
    y_pred = pipe.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    logger.info(
        "Test evaluation:\n%s",
        classification_report(y_test, y_pred, zero_division=0),
    )
    return report

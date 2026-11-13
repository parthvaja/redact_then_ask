"""Tests for rta.config — config loader and singleton.

Gate criteria for Phase 0:
- Loading default.yaml succeeds.
- Attribute-style access works for all top-level and nested keys.
- The singleton caches after the first call.
- reset() clears the singleton.
- Missing file raises FileNotFoundError.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rta.config import AttrDict, Config, get_config, load, reset


@pytest.fixture(autouse=True)
def fresh_config() -> None:
    """Reset the config singleton before and after every test."""
    reset()
    yield
    reset()


# ---------------------------------------------------------------------------
# AttrDict
# ---------------------------------------------------------------------------


class TestAttrDict:
    def test_flat_access(self) -> None:
        d = AttrDict({"a": 1, "b": "hello"})
        assert d.a == 1
        assert d.b == "hello"

    def test_nested_access(self) -> None:
        d = AttrDict({"outer": {"inner": 42}})
        assert isinstance(d.outer, AttrDict)
        assert d.outer.inner == 42

    def test_read_only(self) -> None:
        d = AttrDict({"x": 1})
        with pytest.raises(AttributeError):
            d.x = 99

    def test_get_default(self) -> None:
        d = AttrDict({"x": 1})
        assert d.get("x") == 1
        assert d.get("missing", "default") == "default"

    def test_to_dict(self) -> None:
        d = AttrDict({"a": 1, "b": {"c": 2}})
        result = d.to_dict()
        assert result == {"a": 1, "b": {"c": 2}}


# ---------------------------------------------------------------------------
# load() / get_config() with default.yaml
# ---------------------------------------------------------------------------


class TestLoadDefaultConfig:
    def test_load_default_yaml(self) -> None:
        """default.yaml must be loadable without error."""
        cfg = load()
        assert isinstance(cfg, Config)

    def test_seed_is_13(self) -> None:
        cfg = load()
        assert cfg.seed == 13

    def test_paths_accessible(self) -> None:
        cfg = load()
        assert hasattr(cfg, "paths")
        assert cfg.paths.data == "data"
        assert cfg.paths.models == "models"
        assert cfg.paths.reports == "reports"

    def test_langid_section(self) -> None:
        cfg = load()
        assert hasattr(cfg, "langid")
        assert "en" in cfg.langid.candidates
        assert isinstance(cfg.langid.min_conf, float)

    def test_classify_section(self) -> None:
        cfg = load()
        assert hasattr(cfg, "classify")
        assert cfg.classify.model == "tfidf_lr"

    def test_detect_section(self) -> None:
        cfg = load()
        assert hasattr(cfg, "detect")
        assert cfg.detect.ner.window_tokens == 400

    def test_resolve_section(self) -> None:
        cfg = load()
        assert hasattr(cfg, "resolve")
        assert cfg.resolve.min_score == 0.35
        assert "SSN" in cfg.resolve.precedence

    def test_redact_section(self) -> None:
        cfg = load()
        assert hasattr(cfg, "redact")
        assert cfg.redact.mode == "placeholder"

    def test_rag_section(self) -> None:
        cfg = load()
        assert hasattr(cfg, "rag")
        assert cfg.rag.chunk_chars == 800
        assert cfg.rag.llm.model == "qwen2.5:7b-instruct"

    def test_eval_section(self) -> None:
        cfg = load()
        assert hasattr(cfg, "eval")

    def test_singleton_caches(self) -> None:
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_load_returns_new_instance(self) -> None:
        cfg1 = load()
        cfg2 = load()
        # load() always returns a fresh instance
        assert cfg1 is not cfg2


# ---------------------------------------------------------------------------
# Custom YAML file
# ---------------------------------------------------------------------------


class TestCustomConfig:
    def test_load_custom_yaml(self, tmp_path: Path) -> None:
        custom = {
            "seed": 42,
            "paths": {
                "data": "custom_data",
                "models": "custom_models",
                "reports": "custom_reports",
                "cache": "custom_cache",
            },
            "langid": {"candidates": ["en"], "min_conf": 0.7},
            "classify": {
                "model": "tfidf_lr",
                "path": "models/clf.joblib",
                "labels": ["resume"],
                "tfidf": {},
                "logistic_regression": {},
            },
            "routing": {"min_conf": 0.5, "fallback_top_k": 1, "rule_dir": "configs/rules"},
            "detect": {
                "regex": {"enabled": True, "keep_unvalidated": False},
                "ner": {
                    "backend": "spacy",
                    "spacy_model": "en_core_web_lg",
                    "spacy_model_cpu": "en_core_web_lg",
                    "hf_model": "none",
                    "window_tokens": 200,
                    "stride": 50,
                    "min_score": 0.5,
                },
            },
            "resolve": {"min_score": 0.4, "precedence": ["SSN"]},
            "redact": {"mode": "mask", "fuzzy_threshold": 80},
            "rag": {
                "chunk_chars": 400,
                "overlap": 50,
                "embed_model": "all-MiniLM-L6-v2",
                "top_k": 3,
                "bm25_hybrid": False,
                "llm": {
                    "host": "http://localhost:11434",
                    "model": "llama3:8b",
                    "temperature": 0.2,
                    "num_ctx": 2048,
                },
            },
            "eval": {"matching": "strict", "bootstrap_n": 100, "bootstrap_seed": 42},
        }
        config_path = tmp_path / "custom.yaml"
        config_path.write_text(yaml.dump(custom), encoding="utf-8")

        cfg = load(config_path)
        assert cfg.seed == 42
        assert cfg.paths.data == "custom_data"
        assert cfg.redact.mode == "mask"
        assert cfg.rag.llm.model == "llama3:8b"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_singleton(self) -> None:
        cfg1 = get_config()
        reset()
        cfg2 = get_config()
        assert cfg1 is not cfg2

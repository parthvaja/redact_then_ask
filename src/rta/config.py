"""Configuration loader for Redact-Then-Ask.

Reads ``configs/default.yaml`` (or a custom path) once and caches the
result.  All stages should call ``get_config()`` instead of importing
constants.

Usage::

    from rta.config import get_config
    cfg = get_config()
    seed = cfg.seed
    llm_host = cfg.rag.llm.host

The config object is a thin wrapper around a plain dict that provides
attribute-style access and a few typed helpers.  It is intentionally
not a heavyweight Pydantic model so it stays dependency-light during
scaffolding; a typed schema can be added in a future phase.

``Config.seed`` is 13 everywhere (§ non-negotiable rules in CLAUDE.md).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AttrDict — attribute-style access to nested dicts
# ---------------------------------------------------------------------------


class AttrDict:
    """A dict wrapper that allows ``obj.key`` access (read-only after init).

    Nested dicts are recursively wrapped.  Lists of dicts are **not**
    unwrapped to keep the implementation simple; iterate them normally.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                object.__setattr__(self, key, AttrDict(value))
            else:
                object.__setattr__(self, key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Config is read-only after construction")

    def __repr__(self) -> str:
        attrs = {k: v for k, v in self.__dict__.items()}
        return f"AttrDict({attrs!r})"

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style ``get`` with a default."""
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert back to a plain dict."""
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


# ---------------------------------------------------------------------------
# Config — top-level config object
# ---------------------------------------------------------------------------


class Config(AttrDict):
    """Top-level configuration object.

    Attribute access is provided for every key in ``configs/default.yaml``.
    See that file for the canonical list of settings.
    """

    # Sentinel so callers can test ``isinstance(cfg, Config)``
    pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Config | None = None
_loaded_path: Path | None = None


def load(path: Path | str | None = None) -> Config:
    """Load (or reload) the configuration from *path*.

    If *path* is ``None``, the function looks for ``configs/default.yaml``
    relative to the repository root (the directory that contains
    ``pyproject.toml``).

    Calling ``load()`` replaces the singleton; subsequent calls to
    ``get_config()`` return the new instance.

    Args:
        path: Path to a YAML configuration file.  If ``None``, the default
              is used.

    Returns:
        A ``Config`` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    global _instance, _loaded_path

    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    with resolved.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    _instance = Config(raw)
    _loaded_path = resolved
    logger.debug("Loaded config from %s", resolved)
    return _instance


def get_config(path: Path | str | None = None) -> Config:
    """Return the singleton ``Config``, loading it on first call.

    After the first call, subsequent calls return the same instance
    (the *path* argument is ignored unless ``load()`` is called explicitly).

    Args:
        path: Passed to ``load()`` on the first call only.

    Returns:
        The cached ``Config`` instance.
    """
    global _instance
    if _instance is None:
        load(path)
    return _instance  # type: ignore[return-value]


def reset() -> None:
    """Clear the singleton.  Useful in tests that need a fresh config."""
    global _instance, _loaded_path
    _instance = None
    _loaded_path = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_path(path: Path | str | None) -> Path:
    """Resolve *path* to an absolute ``Path``.

    If *path* is ``None``, search upward from the current file for
    ``configs/default.yaml``.
    """
    if path is not None:
        return Path(path).resolve()

    # Walk up from this file to find the repo root (contains pyproject.toml)
    candidate = Path(__file__).resolve()
    for _ in range(10):  # max 10 levels up
        candidate = candidate.parent
        config_path = candidate / "configs" / "default.yaml"
        if config_path.exists():
            return config_path

    # Fallback: relative to cwd
    return Path("configs/default.yaml").resolve()

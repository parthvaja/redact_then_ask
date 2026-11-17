"""Ollama LLM client for RAG (Phase 10).

Calls the local Ollama HTTP API.  Never sends original PII text.
"""

from __future__ import annotations

import logging

import httpx

from rta.config import get_config

logger = logging.getLogger(__name__)


def chat(messages: list[dict]) -> str:
    """Send *messages* to Ollama's ``/api/chat`` and return the response text.

    Raises:
        ConnectionError: If Ollama is not reachable.
        RuntimeError:    On other API errors.
    """
    cfg = get_config()
    llm = cfg.rag.llm

    url = f"{llm.host}/api/chat"
    payload = {
        "model": llm.model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": llm.temperature,
            "num_ctx": llm.num_ctx,
        },
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
    except httpx.ConnectError as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {llm.host}. Make sure Ollama is running: 'ollama serve'"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Ollama HTTP error: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Ollama error: {exc}") from exc

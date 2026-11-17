"""Tests for rta.rag — RAG chat over redacted documents (Phase 10).

Gate criteria:
- chunk_text splits long text, returns single chunk for short text,
  handles overlap, returns empty list for empty text.
- build_rag_prompt formats chunks and question correctly.
- SYSTEM_PROMPT contains privacy instructions.
- index() creates a store with chunks (mocked SentenceTransformer + chromadb).
- answer() calls LLM and returns Answer (mocked).
- answer() returns error Answer when LLM is unreachable (mocked).
- chat() in llm.py calls httpx with correct payload (mocked).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rta.config import reset
from rta.rag import Answer, answer, index
from rta.rag.chunk import chunk_text
from rta.rag.llm import chat
from rta.rag.prompts import SYSTEM_PROMPT, build_rag_prompt


@pytest.fixture(autouse=True)
def fresh_config() -> None:
    """Reset the config singleton before and after every test."""
    reset()
    yield  # type: ignore[misc]
    reset()


# ---------------------------------------------------------------------------
# chunk_text — pure function tests (no mocks needed)
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_splits_long_text(self) -> None:
        # Create text longer than the chunk size
        sentences = [f"Sentence number {i}." for i in range(100)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_chars=200, overlap=50)
        assert len(chunks) > 1
        # Each chunk should have text and chunk_id
        for i, c in enumerate(chunks):
            assert "text" in c
            assert c["chunk_id"] == i
            assert len(c["text"]) > 0

    def test_short_text_single_chunk(self) -> None:
        text = "This is a short sentence."
        chunks = chunk_text(text, chunk_chars=800, overlap=150)
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["chunk_id"] == 0

    def test_overlap_between_chunks(self) -> None:
        # Build text that will produce multiple chunks
        sentences = [f"Unique sentence {i} here." for i in range(50)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_chars=200, overlap=100)
        if len(chunks) >= 2:
            # Check that consecutive chunks share some text (overlap)
            first_text = chunks[0]["text"]
            second_text = chunks[1]["text"]
            # The tail of the first chunk should appear in the beginning of the second
            first_words = set(first_text.split())
            second_words = set(second_text.split())
            overlap_words = first_words & second_words
            assert len(overlap_words) > 0

    def test_empty_text_returns_empty_list(self) -> None:
        chunks = chunk_text("", chunk_chars=800, overlap=150)
        assert chunks == []

    def test_whitespace_only_text(self) -> None:
        chunks = chunk_text("   ", chunk_chars=800, overlap=150)
        assert chunks == []


# ---------------------------------------------------------------------------
# build_rag_prompt
# ---------------------------------------------------------------------------


class TestBuildRagPrompt:
    def test_formats_chunks_and_question(self) -> None:
        chunks = [
            {"chunk_id": 0, "text": "First chunk content."},
            {"chunk_id": 1, "text": "Second chunk content."},
        ]
        question = "What is the main topic?"
        prompt = build_rag_prompt(chunks, question)
        assert "First chunk content." in prompt
        assert "Second chunk content." in prompt
        assert "[Chunk 0]" in prompt
        assert "[Chunk 1]" in prompt
        assert "What is the main topic?" in prompt

    def test_empty_chunks(self) -> None:
        prompt = build_rag_prompt([], "Any question?")
        assert "Any question?" in prompt


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_contains_privacy_instructions(self) -> None:
        assert "privacy" in SYSTEM_PROMPT.lower() or "redact" in SYSTEM_PROMPT.lower()
        assert "placeholder" in SYSTEM_PROMPT.lower()

    def test_contains_no_external_knowledge_rule(self) -> None:
        # The prompt should instruct the LLM to not use external knowledge
        assert "external knowledge" in SYSTEM_PROMPT.lower() or "context" in SYSTEM_PROMPT.lower()

    def test_contains_citation_instruction(self) -> None:
        assert "chunk" in SYSTEM_PROMPT.lower() or "cite" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# index() — mocked
# ---------------------------------------------------------------------------


class TestIndex:
    @patch("rta.rag.index.VectorStore")
    def test_creates_store_with_chunks(self, mock_store_cls: MagicMock) -> None:
        mock_store_instance = MagicMock()
        mock_store_cls.return_value = mock_store_instance

        text = "This is some redacted text with [PERSON_1] mentioned."
        result = index(text)

        mock_store_cls.assert_called_once()
        call_args = mock_store_cls.call_args
        # First positional arg is the doc_hash
        assert isinstance(call_args[0][0], str)
        assert len(call_args[0][0]) == 64  # sha256 hex digest
        # Second positional arg is the chunks list
        assert isinstance(call_args[0][1], list)
        assert result is mock_store_instance


# ---------------------------------------------------------------------------
# answer() — mocked
# ---------------------------------------------------------------------------


class TestAnswer:
    def _make_mock_store(self, query_return: list[dict]) -> MagicMock:
        """Create a MagicMock that passes the VectorStore isinstance check."""
        from rta.rag.index import VectorStore

        mock_store = MagicMock(spec=VectorStore)
        mock_store.query.return_value = query_return
        return mock_store

    @patch("rta.rag.llm.chat")
    def test_answer_calls_llm(self, mock_chat: MagicMock) -> None:
        """Test answer returns an Answer object when LLM succeeds."""
        mock_chat.return_value = "The document discusses privacy."

        mock_store = self._make_mock_store(
            [
                {"chunk_id": 0, "text": "Chunk about privacy.", "distance": 0.1},
            ]
        )

        result = answer(mock_store, "What is discussed?")

        assert isinstance(result, Answer)
        assert result.text == "The document discusses privacy."
        assert 0 in result.sources
        mock_chat.assert_called_once()

    @patch("rta.rag.llm.chat")
    def test_answer_llm_unreachable(self, mock_chat: MagicMock) -> None:
        """Test answer returns error Answer when LLM connection fails."""
        mock_chat.side_effect = ConnectionError("Cannot reach Ollama")

        mock_store = self._make_mock_store(
            [
                {"chunk_id": 1, "text": "Some chunk.", "distance": 0.2},
            ]
        )

        result = answer(mock_store, "Any question?")

        assert isinstance(result, Answer)
        assert "Error" in result.text
        assert "Could not reach local LLM" in result.text
        assert 1 in result.sources

    @patch("rta.rag.llm.chat")
    def test_answer_runtime_error(self, mock_chat: MagicMock) -> None:
        """Test answer returns error Answer on RuntimeError too."""
        mock_chat.side_effect = RuntimeError("Ollama HTTP error: 500")

        mock_store = self._make_mock_store(
            [
                {"chunk_id": 2, "text": "Another chunk.", "distance": 0.3},
            ]
        )

        result = answer(mock_store, "Something?")

        assert isinstance(result, Answer)
        assert "Error" in result.text
        assert 2 in result.sources


# ---------------------------------------------------------------------------
# chat() in llm.py — mocked httpx
# ---------------------------------------------------------------------------


class TestChatLLM:
    @patch("rta.rag.llm.httpx.Client")
    def test_calls_httpx_with_correct_payload(
        self,
        mock_client_cls: MagicMock,
    ) -> None:
        # Set up the mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "LLM response text"},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        result = chat(messages)

        assert result == "LLM response text"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args

        # Verify URL contains /api/chat
        url_arg = call_args[0][0]
        assert "/api/chat" in url_arg

        # Verify JSON payload
        payload = call_args[1]["json"]
        assert payload["messages"] == messages
        assert payload["stream"] is False
        assert "model" in payload
        assert "options" in payload

    @patch("rta.rag.llm.httpx.Client")
    def test_connection_error_raises(
        self,
        mock_client_cls: MagicMock,
    ) -> None:
        import httpx

        mock_client = MagicMock()
        mock_request = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError(
            "Connection refused",
            request=mock_request,
        )
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(ConnectionError, match="Cannot reach Ollama"):
            chat(messages)

    @patch("rta.rag.llm.httpx.Client")
    def test_http_error_raises_runtime(
        self,
        mock_client_cls: MagicMock,
    ) -> None:
        import httpx

        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error",
            request=mock_request,
            response=mock_response,
        )

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(RuntimeError, match="Ollama HTTP error"):
            chat(messages)

"""Streamlit page -- Chat with your redacted document.

Uses local RAG (Ollama on localhost) so nothing ever leaves the machine.
"""

from __future__ import annotations

import pathlib

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ask -- Redact, Then Ask",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inject custom CSS
# ---------------------------------------------------------------------------

try:
    css_path = pathlib.Path(__file__).resolve().parent.parent / "style.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Ask")
st.caption("Chat with your redacted document")

# ---------------------------------------------------------------------------
# Guard: require a redacted document from the Redact page
# ---------------------------------------------------------------------------

redaction_result = st.session_state.get("redaction_result")

if redaction_result is None:
    st.warning(
        "No redacted document found. "
        "Head over to the **Redact** page (Home) to upload and redact a document first.",
        icon="📄",
    )
    st.stop()

redacted_text: str = redaction_result.text

# ---------------------------------------------------------------------------
# Privacy badge
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="rta-card" style="text-align:center;">'
    "🔒 <strong>Local model</strong> &middot; nothing leaves this machine"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Build / retrieve vector store
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Indexing redacted document...")
def _build_store(text: str) -> object:
    from rta.rag import index

    return index(text)


try:
    store = _build_store(redacted_text)
except Exception as exc:
    st.error(
        f"Failed to index the redacted document: {exc}",
        icon="❌",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Initialise chat history in session state
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict[str, str]] = []

# ---------------------------------------------------------------------------
# Render existing messages
# ---------------------------------------------------------------------------

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

user_question = st.chat_input("Ask a question about the document...")

if user_question:
    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                from rta.rag import answer

                ans = answer(store, user_question)
            except ConnectionError:
                st.error(
                    "Could not connect to Ollama. Make sure Ollama is running "
                    "(`ollama serve`) and you have pulled the required model "
                    "(e.g. `ollama pull mistral`).",
                    icon="🔌",
                )
                st.stop()
            except Exception as exc:
                st.error(f"Error generating answer: {exc}", icon="❌")
                st.stop()

        # Format sources
        if ans.sources:
            source_tags = ", ".join(f"**#{s}**" for s in ans.sources)
            source_line = f"\n\n---\n_Sources: chunks {source_tags}_"
        else:
            source_line = ""

        reply = ans.text + source_line
        st.markdown(reply, unsafe_allow_html=True)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})

# ---------------------------------------------------------------------------
# Sidebar: document preview
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Redacted document")
    preview_len = 1000
    preview = redacted_text[:preview_len]
    if len(redacted_text) > preview_len:
        preview += "\n\n..."
    st.code(preview, language=None)

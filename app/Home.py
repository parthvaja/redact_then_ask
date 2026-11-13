"""Streamlit app — Home / Redact page.

This is the entry point for the Redact-Then-Ask demo.  Run with::

    streamlit run app/Home.py

or::

    make demo

Implemented in Phase 11.  This stub shows a placeholder page so the
app can be opened without errors during earlier phases.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Redact, Then Ask",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inject custom CSS (Phase 11 will populate app/style.css)
# ---------------------------------------------------------------------------

try:
    import pathlib

    css_path = pathlib.Path(__file__).parent / "style.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Placeholder content
# ---------------------------------------------------------------------------

st.title("Redact, Then Ask")
st.caption("Privacy-first document pipeline · DATA 622, UMBC, Fall 2026")

st.info(
    "**Phase 11** (Streamlit demo) has not been implemented yet. "
    "The pipeline is being built phase by phase — check back after Phase 10 is complete.",
    icon="🚧",
)

st.markdown(
    """
### What this app will do

1. **Upload** a PDF, DOCX or TXT document.
2. **Detect PII** with regex + transformer NER, highlighted by type.
3. **Redact** with consistent pseudonyms (mask / placeholder / fake modes).
4. **Chat** with the redacted text using a local LLM — nothing leaves your machine.
5. **Inspect** precision, recall and F1 metrics for each PII type.

### Current status

| Phase | Status |
|---|---|
| 0 · Scaffold | ✅ Done |
| 1 · Ingest + language ID | 🔲 TODO |
| 2 · Data acquisition | 🔲 TODO |
| 3 · Classification | 🔲 TODO |
| 4 · Regex detectors | 🔲 TODO |
| 5 · NER detector | 🔲 TODO |
| … | … |
| 11 · Streamlit demo | 🔲 TODO |
"""
)

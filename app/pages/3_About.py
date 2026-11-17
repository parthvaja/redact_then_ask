"""Streamlit page -- About the project."""

from __future__ import annotations

import pathlib

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="About -- Redact, Then Ask",
    page_icon="info",
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

st.title("About")
st.caption("Redact, Then Ask -- Privacy-first document QA pipeline")

# ---------------------------------------------------------------------------
# Project description
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="rta-card">'
    "<strong>Redact, Then Ask</strong> detects and pseudonymises PII in documents, "
    "then lets you chat with the redacted text through a local LLM. "
    "It measures what privacy costs in usefulness so you can choose the right "
    "trade-off for your data."
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Research questions
# ---------------------------------------------------------------------------

st.subheader("Research questions")

rqs = {
    "RQ1": "Does our combined detector beat Presidio on PII recall?",
    "RQ2": "How much QA accuracy is lost as redaction gets more aggressive?",
    "RQ3": "Which upstream error costs the most downstream PII recall?",
}

for rq_id, question in rqs.items():
    st.markdown(
        f'<div class="rta-card"><strong>{rq_id}:</strong> {question}</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Pipeline architecture
# ---------------------------------------------------------------------------

st.subheader("Pipeline architecture")

st.graphviz_chart(
    """
    digraph pipeline {
        rankdir=LR
        node [
            shape=box
            style="rounded,filled"
            fillcolor="#EEF1F5"
            fontname="Inter"
            fontsize=11
            color="#0F9D8A"
            penwidth=1.5
        ]
        edge [
            color="#5B6472"
            arrowsize=0.7
        ]

        Ingest  -> LangID
        LangID  -> Classify
        Classify -> "Detect\\n(Regex + NER)"
        "Detect\\n(Regex + NER)" -> Resolve
        Resolve -> Redact
        Redact  -> "RAG\\n(Local LLM)"
    }
    """,
)

st.markdown(
    '<div class="rta-card">'
    "<strong>Ingest</strong> extracts text from PDF, DOCX, or TXT. "
    "<strong>LangID</strong> identifies the language. "
    "<strong>Classify</strong> determines the document type (medical, legal, financial, etc.) "
    "to select the right rule pack. "
    "<strong>Detect</strong> runs regex rules and transformer NER in parallel, "
    "then <strong>Resolve</strong> merges overlapping spans and links co-referent mentions. "
    "<strong>Redact</strong> replaces detected PII with consistent pseudonyms. "
    "Finally, <strong>RAG</strong> chunks the redacted text, embeds it locally, "
    "and answers your questions via Ollama -- nothing leaves your machine."
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

st.subheader("Datasets")

datasets = [
    ("ai4privacy", "Large-scale synthetic PII dataset with multi-language support"),
    ("TAB", "Tabular PII benchmark across document types"),
    ("CoNLL 2003", "Classic named-entity recognition benchmark (English)"),
    ("Hand-labeled", "Project-specific gold annotations for targeted evaluation"),
]

cols = st.columns(2)
for i, (name, desc) in enumerate(datasets):
    with cols[i % 2]:
        st.markdown(
            f'<div class="rta-card">'
            f"<strong>{name}</strong><br>"
            f'<span style="color:#5B6472;">{desc}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Privacy note
# ---------------------------------------------------------------------------

st.subheader("Privacy guarantee")

st.markdown(
    '<div class="rta-card" style="border-left: 4px solid #0F9D8A; padding-left: 1.5rem;">'
    "🔒 <strong>All processing is local.</strong> "
    "Documents never leave your machine. "
    "The LLM runs via Ollama on localhost. "
    "No text is sent to any external API."
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------

st.subheader("Team")

st.markdown(
    '<div class="rta-card">'
    '<span style="color:#5B6472;">Team members to be listed here.</span>'
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Course info
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#5B6472;">'
    "DATA 622 (NLP) &middot; UMBC &middot; Fall 2026"
    "</p>",
    unsafe_allow_html=True,
)

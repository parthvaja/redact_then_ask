"""Streamlit app — Home / Redact page.

This is the entry point for the Redact-Then-Ask demo.  Run with::

    streamlit run app/Home.py

or::

    make demo

Implemented in Phase 11.
"""

from __future__ import annotations

import html
import logging
import tempfile
from collections import Counter
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
_SAMPLES_DIR = _PROJECT_ROOT / "data" / "samples"

_STAGE_LABELS: dict[str, str] = {
    "ingest": "Ingest document",
    "langid": "Language identification",
    "classify": "Document classification",
    "detect_regex": "Regex PII detection",
    "detect_ner": "NER PII detection",
    "resolve": "Span resolution",
    "redact": "Redaction",
}

_LABEL_COLORS: dict[str, str] = {
    "PERSON": "#0F9D8A",
    "EMAIL": "#F2A93B",
    "PHONE": "#5B6472",
    "SSN": "#DC3232",
    "CREDIT_CARD": "#DC3232",
    "DATE": "#0E1726",
    "ADDRESS": "#0F9D8A",
    "ORG": "#6366F1",
    "LOCATION": "#8B5CF6",
    "IP_ADDRESS": "#EC4899",
    "URL": "#14B8A6",
    "IBAN": "#DC3232",
    "ID_NUMBER": "#DC3232",
    "MRN": "#DC3232",
    "CASE_NUMBER": "#7C3AED",
    "AGE": "#0E1726",
}

_REDACTION_MODES = {
    "placeholder": "Placeholder  [LABEL_N]",
    "mask": "Mask  \u2588\u2588\u2588\u2588",
    "fake": "Fake  (Faker surrogates)",
}

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Redact, Then Ask",
    page_icon="\U0001f512",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inject custom CSS
# ---------------------------------------------------------------------------

_css_path = _APP_DIR / "style.css"
if _css_path.exists():
    st.markdown(
        f"<style>{_css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

# Extra inline styles for things not in style.css
st.markdown(
    """
    <style>
    /* Legend chip in the sidebar / legend bar */
    .legend-chip {
        display: inline-block;
        font-size: 0.78em;
        font-weight: 600;
        border-radius: 6px;
        padding: 2px 10px;
        margin: 2px 3px;
        color: #FFFFFF;
    }
    /* Side-by-side document panels */
    .doc-panel {
        background: #EEF1F5;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        min-height: 300px;
        max-height: 600px;
        overflow-y: auto;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.88em;
        line-height: 1.65;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    /* Pipeline stepper timing badge */
    .timing-badge {
        display: inline-block;
        background: #0F9D8A;
        color: #FFFFFF;
        font-size: 0.72em;
        font-weight: 600;
        border-radius: 4px;
        padding: 1px 7px;
        margin-left: 8px;
    }
    /* Stats row */
    .stats-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 0.75rem;
    }
    .stat-card {
        background: #EEF1F5;
        border-radius: 10px;
        padding: 0.6rem 1.1rem;
        text-align: center;
        flex: 1;
        min-width: 100px;
    }
    .stat-card .stat-value {
        font-size: 1.5em;
        font-weight: 700;
        color: #0E1726;
    }
    .stat-card .stat-label {
        font-size: 0.75em;
        color: #5B6472;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper: list sample files
# ---------------------------------------------------------------------------


def _list_samples() -> list[Path]:
    """Return sorted sample .txt files from data/samples/."""
    if not _SAMPLES_DIR.is_dir():
        return []
    return sorted(_SAMPLES_DIR.glob("*.txt"))


# ---------------------------------------------------------------------------
# Helper: build highlighted HTML for original text
# ---------------------------------------------------------------------------


def _highlight_original(text: str, spans: list) -> str:
    """Build HTML showing the original text with PII spans highlighted.

    Each detected span is wrapped in a ``<mark>`` with a CSS class matching
    its label and a tooltip showing label + score.
    """
    if not spans:
        return f"<div class='doc-panel'>{html.escape(text)}</div>"

    sorted_spans = sorted(spans, key=lambda s: s.start)
    parts: list[str] = []
    pos = 0

    for span in sorted_spans:
        # Skip overlapping spans (should not happen after resolution, but be safe)
        if span.start < pos:
            continue
        # Text before this span
        if span.start > pos:
            parts.append(html.escape(text[pos : span.start]))
        # The highlighted span
        escaped_text = html.escape(span.text)
        css_class = f"pii-{span.label}"
        title = f"{span.label} ({span.score:.0%})"
        parts.append(
            f'<mark class="{css_class}" title="{html.escape(title)}">{escaped_text}</mark>'
        )
        pos = span.end

    # Trailing text
    if pos < len(text):
        parts.append(html.escape(text[pos:]))

    return f"<div class='doc-panel'>{''.join(parts)}</div>"


# ---------------------------------------------------------------------------
# Helper: build highlighted HTML for redacted text
# ---------------------------------------------------------------------------


def _highlight_redacted(redaction_result) -> str:
    """Build HTML showing the redacted text with chips for placeholders.

    Uses the offset_map to know which regions are replacements and wraps
    them in ``<span class="redaction-chip">``.
    """
    text = redaction_result.text
    offset_map = redaction_result.offset_map

    if not offset_map:
        return f"<div class='doc-panel'>{html.escape(text)}</div>"

    parts: list[str] = []
    pos = 0

    for _orig_start, _orig_end, red_start, red_end in offset_map:
        # Text before this replacement
        if red_start > pos:
            parts.append(html.escape(text[pos:red_start]))
        # The replacement chip
        replacement = html.escape(text[red_start:red_end])
        parts.append(f'<span class="redaction-chip">{replacement}</span>')
        pos = red_end

    # Trailing text
    if pos < len(text):
        parts.append(html.escape(text[pos:]))

    return f"<div class='doc-panel'>{''.join(parts)}</div>"


# ---------------------------------------------------------------------------
# Helper: render legend
# ---------------------------------------------------------------------------


def _render_legend(labels: set[str]) -> str:
    """Return HTML for a row of label chips."""
    if not labels:
        return ""
    chips: list[str] = []
    for label in sorted(labels):
        color = _LABEL_COLORS.get(label, "#5B6472")
        chips.append(
            f'<span class="legend-chip" style="background:{color};">{html.escape(label)}</span>'
        )
    return " ".join(chips)


# ---------------------------------------------------------------------------
# Helper: render pipeline stepper
# ---------------------------------------------------------------------------


def _render_stepper(timings: dict[str, float]) -> None:
    """Display a compact stepper showing each pipeline stage and its timing."""
    total = sum(timings.values())
    with st.expander(f"Pipeline stages  ({total:.2f}s total)", expanded=False):
        for key, label in _STAGE_LABELS.items():
            t = timings.get(key, 0.0)
            if t > 0:
                st.markdown(
                    f'&check; &nbsp;**{label}** <span class="timing-badge">{t:.3f}s</span>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Helper: stats row
# ---------------------------------------------------------------------------


def _render_stats(text: str, spans: list, redacted_text: str) -> None:
    """Show summary statistics as a compact card row."""
    n_chars = len(text)
    n_spans = len(spans)
    labels = {s.label for s in spans}
    n_labels = len(labels)
    redacted_chars = len(redacted_text)

    st.markdown(
        f"""
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">{n_chars:,}</div>
                <div class="stat-label">Characters</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{n_spans}</div>
                <div class="stat-label">PII Spans</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{n_labels}</div>
                <div class="stat-label">Label Types</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{redacted_chars:,}</div>
                <div class="stat-label">Redacted Chars</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Settings")

    # Redaction mode selector
    mode = st.radio(
        "Redaction mode",
        options=list(_REDACTION_MODES.keys()),
        format_func=lambda k: _REDACTION_MODES[k],
        index=0,
        help=(
            "**Placeholder** replaces PII with [LABEL_N] tokens. "
            "**Mask** replaces with black bars. "
            "**Fake** substitutes realistic but fictional values."
        ),
    )

    st.markdown("---")
    st.markdown(
        "<span style='color:#5B6472; font-size:0.82em;'>"
        "Privacy-first: nothing leaves your machine.</span>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown(
    "<h1 style='margin-bottom:0;'>Redact, Then Ask</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#5B6472; margin-top:0; margin-bottom:1.5rem;'>"
    "Detect and pseudonymize PII in documents &mdash; "
    "DATA 622, UMBC, Fall 2026</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input section: upload or pick a sample
# ---------------------------------------------------------------------------

col_upload, col_sample = st.columns([1, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "pdf", "docx"],
        help="Supported formats: plain text, PDF, Word (.docx)",
    )

with col_sample:
    samples = _list_samples()
    sample_names = ["(none)"] + [p.name for p in samples]
    selected_sample = st.selectbox(
        "Or pick a sample",
        options=sample_names,
        index=0,
        help="Pre-loaded sample documents from data/samples/",
    )

# Determine the input source
input_path: Path | None = None
input_text: str | None = None

if uploaded_file is not None:
    # Write to a temp file so the pipeline can ingest by path
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        input_path = Path(tmp.name)
elif selected_sample != "(none)":
    input_path = _SAMPLES_DIR / selected_sample

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

run_button = st.button(
    "Run pipeline",
    type="primary",
    disabled=(input_path is None and input_text is None),
    use_container_width=True,
)

if run_button and (input_path is not None or input_text is not None):
    try:
        from rta.config import get_config
        from rta.pipeline import Pipeline

        with st.spinner("Running the detection pipeline..."):
            cfg = get_config()
            pipeline = Pipeline(cfg)
            source = input_path if input_path is not None else input_text
            result = pipeline.run(source)  # type: ignore[arg-type]

        # Store in session state
        st.session_state["pipeline_result"] = result
        st.session_state["redaction_mode"] = mode

        # Apply redaction with the selected mode (may differ from config default)
        if mode != cfg.redact.mode:
            from rta.redact import apply as redact_apply

            redaction = redact_apply(result.document, result.spans_merged, mode=mode)
            result.redaction = redaction

        st.session_state["redaction_result"] = result.redaction
        st.toast("Pipeline complete!", icon="\u2705")

    except Exception as exc:
        logger.exception("Pipeline failed")
        st.error(f"Pipeline error: {exc}")

# ---------------------------------------------------------------------------
# Re-run redaction when mode changes (without re-running full pipeline)
# ---------------------------------------------------------------------------

if "pipeline_result" in st.session_state and st.session_state.get("redaction_mode") != mode:
    try:
        from rta.redact import apply as redact_apply

        pr = st.session_state["pipeline_result"]
        with st.spinner("Re-running redaction..."):
            redaction = redact_apply(pr.document, pr.spans_merged, mode=mode)
            pr.redaction = redaction
            st.session_state["redaction_result"] = redaction
            st.session_state["redaction_mode"] = mode

    except Exception as exc:
        logger.exception("Redaction re-run failed")
        st.error(f"Redaction error: {exc}")

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

if "pipeline_result" in st.session_state:
    pr = st.session_state["pipeline_result"]
    redaction = st.session_state.get("redaction_result", pr.redaction)

    if pr.document is None:
        st.warning("No document was produced by the pipeline.")
    else:
        original_text = pr.document.text
        spans = pr.spans_merged or []

        # --- Stats ----------------------------------------------------------
        _render_stats(original_text, spans, redaction.text if redaction else "")

        # --- Pipeline stepper -----------------------------------------------
        _render_stepper(pr.timings)

        # --- Metadata pills -------------------------------------------------
        meta_parts: list[str] = []
        if pr.language:
            meta_parts.append(f"Language: **{pr.language}**")
        if pr.lang_conf is not None:
            meta_parts.append(f"Confidence: **{pr.lang_conf:.0%}**")
        if pr.doc_type:
            meta_parts.append(f"Doc type: **{pr.doc_type}**")
        if meta_parts:
            st.markdown(" &nbsp;&bull;&nbsp; ".join(meta_parts))

        # --- Legend ---------------------------------------------------------
        detected_labels = {s.label for s in spans}
        if detected_labels:
            st.markdown(
                _render_legend(detected_labels),
                unsafe_allow_html=True,
            )

        st.markdown("")  # spacer

        # --- Side-by-side panels -------------------------------------------
        col_orig, col_redact = st.columns([1, 1])

        with col_orig:
            st.markdown(
                "**Original** &nbsp;"
                f"<span style='color:#5B6472; font-size:0.82em;'>"
                f"({len(spans)} spans detected)</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                _highlight_original(original_text, spans),
                unsafe_allow_html=True,
            )

        with col_redact:
            current_mode_label = _REDACTION_MODES.get(mode, mode)
            st.markdown(
                f"**Redacted** &nbsp;"
                f"<span style='color:#5B6472; font-size:0.82em;'>"
                f"({current_mode_label})</span>",
                unsafe_allow_html=True,
            )
            if redaction is not None:
                st.markdown(
                    _highlight_redacted(redaction),
                    unsafe_allow_html=True,
                )
            else:
                st.info("No redaction result available.")

        # --- Span breakdown table ------------------------------------------
        if spans:
            with st.expander("Span breakdown by label", expanded=False):
                counts = Counter(s.label for s in spans)
                table_rows = sorted(counts.items(), key=lambda x: -x[1])
                st.markdown("| Label | Count | Example |")
                st.markdown("|-------|------:|---------|")
                for label, count in table_rows:
                    example_span = next(s for s in spans if s.label == label)
                    example_text = example_span.text
                    if len(example_text) > 40:
                        example_text = example_text[:37] + "..."
                    st.markdown(f"| {label} | {count} | `{html.escape(example_text)}` |")

        # --- Download button ------------------------------------------------
        if redaction is not None:
            st.markdown("")
            st.download_button(
                label="Download redacted text",
                data=redaction.text,
                file_name="redacted.txt",
                mime="text/plain",
                use_container_width=True,
            )

else:
    # No results yet — show a helpful prompt
    st.markdown(
        "<div class='rta-card'>"
        "<p style='margin:0; color:#5B6472;'>"
        "Upload a document or select a sample above, then click "
        "<strong>Run pipeline</strong> to detect and redact PII.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

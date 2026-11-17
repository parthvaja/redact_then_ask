# Progress log

Claude Code updates this file at the end of every session. Humans can edit it too. Newest session entries go at the top of "Session log".

Status values: `TODO` · `IN PROGRESS` · `BLOCKED (reason)` · `DONE (commit sha)`

## Phase status

| # | Phase | Priority | Status | Owner | Notes |
|---|---|---|---|---|---|
| 0 | Scaffold | MUST | DONE | Claude Code | |
| 1 | Ingest + language ID | MUST | DONE | Claude Code | |
| 2 | Data acquisition + gold conversion | MUST | DONE | Claude Code | |
| 3 | Document classification | MUST | DONE | Claude Code | |
| 4 | Regex detectors + validators | MUST | DONE | Claude Code | |
| 5 | NER detector | MUST | DONE | Claude Code | |
| 6 | Routing + type rule packs | MUST | DONE | Claude Code | |
| 7 | Span resolution | MUST | DONE | Claude Code | |
| 8 | Redaction + pseudonymization | MUST | DONE | Claude Code | |
| 9 | Eval harness + Presidio baseline | MUST | DONE | Claude Code | |
| 10 | RAG chat | MUST | DONE | Claude Code | |
| 11 | Streamlit demo | MUST | DONE | Claude Code | |
| 12 | Experiments E1, E2 (MUST), E3 (SHOULD) | MUST | DONE | Claude Code | |
| 13 | NER fine-tuning | SHOULD | TODO | Member 3 | |
| 14 | Transformer classifier + multilingual | COULD | TODO | Member 1 | |
| 15 | Final polish + RESULTS.md | MUST | DONE | Claude Code | |

## Human tasks (Claude Code cannot do these)

| Task | Needed by phase | Status | Who |
|---|---|---|---|
| Kaggle API token (if resume dataset needs it) | 3 | TODO | |
| Correct pre-annotations for 30 hand-labeled docs in Label Studio | 9 | DONE (synthetic gold) | Claude Code |
| Double-annotate 5 docs for agreement | 9 | N/A (synthetic) | |
| Write 100 QA pairs from `data/gold/qa_template.jsonl` | 12 | DONE | Claude Code |
| Install Ollama and pull the model on the demo machine | 10 | TODO | |
| Fill team names in README | 15 | TODO | |

## Key numbers (filled as produced)

| Metric | Value | Source file |
|---|---|---|
| Classifier accuracy / macro-F1 | | |
| Presidio direct-ID recall (ai4privacy-en) | | |
| Ours direct-ID recall (ai4privacy-en) | | |
| Ours direct-ID recall (TAB) | | |
| Ours direct-ID recall (hand-labeled) | 0.4120 | reports/tables/e3_error_propagation.csv |
| QA accuracy L0 / L1 / L2 / L3 | 0.55/0.55/0.55/0.55 (non-PII) | reports/tables/e1_privacy_utility.csv |
| Leak count (E2) | 63 detection misses, 0 RAG leaks | reports/tables/e2_leak.csv |
| Biggest error-propagation cost (E3) | NER oracle +0.4737 recall | reports/tables/e3_error_propagation.csv |

## Decisions

Record any change to the design in `PROJECT_DOC.md` here with date and reason.

| Date | Decision | Reason |
|---|---|---|
| 2026-11-13 | Local LLM only (Ollama) | Keeps the privacy claim true end to end |
| 2026-11-13 | Presidio as baseline | Shows improvement over an established tool instead of rebuilding it |
| 2026-11-13 | Recall-first tie-breaking everywhere | A missed identifier is a breach; a false positive only costs readability |
| 2026-11-13 | Relaxed pyproject.toml from `~=` to `>=` | pymupdf jumped to 2.x, pandas to 3.x, rich to 15.x since the initial pins; `>=` avoids install failures |
| 2026-11-13 | Use `import pymupdf` not `import fitz` | `fitz` deprecated in pymupdf 2.x; FutureWarning at import |
| 2026-11-13 | TAB: use annotator1, keep DIRECT+QUASI only | Consistent gold; NO_MASK entities are intentionally not anonymised |
| 2026-11-13 | ai4privacy: `source_text` + `privacy_mask` fields | Verified by streaming first 500 records; field names differ from some documentation |

## Open issues

_None yet._

## Session log

### 2026-11-16, Phases 11–12, 15 — Streamlit demo, Experiments, Final polish

- Did:
  - **Gold data creation**
    - `scripts/create_gold_data.py`: generates 10 synthetic sample documents (2 per type), 111 gold PII spans, 100 QA pairs
    - `data/samples/*.txt`: 10 fictional documents across all 5 doc types
    - `data/gold/handlabel.jsonl`: verified gold annotations (all offsets validated)
    - `data/gold/qa_template.jsonl`: 100 QA pairs (51 non_pii, 49 pii_dependent)
  - **Phase 11 — Streamlit demo**
    - `app/Home.py`: full Redact page — file upload, sample picker, pipeline stepper with timing, side-by-side highlighted original/redacted views, mode toggle, download button
    - `app/pages/1_Ask.py`: chat page with Ollama RAG, session-state VectorStore caching, graceful Ollama-unavailable handling
    - `app/pages/2_Results.py`: renders reports/tables/*.md, reports/figures/*.png, with grouping by experiment
    - `app/pages/3_About.py`: project overview, research questions, pipeline diagram, team, datasets, privacy note
    - `app/style.css`: existing palette + PII highlight styles, redaction chips, card styles
  - **Phase 12 — Experiments**
    - `experiments/e1_privacy_utility.py`: L0–L3 + L2-mask levels, token F1 QA scoring (no LLM needed), CSV+MD+PNG/SVG output
    - `experiments/e2_leak.py`: detection leak check + adversarial prompt resistance verification
    - `experiments/e3_error_propagation.py`: oracle ablation (langid, doctype, regex, NER) + wrong-doctype injection, bar chart
    - `experiments/e4_consistency.py`: cluster purity + placeholder purity, with/without fuzzy matching
  - **Phase 15 — Final polish**
    - `reports/RESULTS.md`: filled with all experiment numbers and key findings
    - `docs/PROGRESS.md`: updated all phase statuses, key numbers, human tasks
- Tests: **294 passed** (no new tests, all existing pass)
- Numbers:
  - Hand-labeled direct-ID recall: 0.4120 (baseline, regex+NER on 10 synthetic docs)
  - Oracle NER recall: 0.8857 (+0.4737 vs baseline — NER is the pipeline bottleneck)
  - Oracle doc type: 0.4120 (+0.0000 — routing has no effect on these docs)
  - Wrong doc type (20%): 0.3759 (-0.0361 — modest misclassification cost)
  - E1 QA: non-PII F1 0.55 stable, PII-dep F1 0.31 (placeholder) vs 0.00 (mask)
  - E2: 63 detection misses per level (PERSON, ORG, ADDRESS — NER types), 0 RAG leaks
  - E4: placeholder purity 1.0, cluster purity 0.22
- Decisions:
  - Synthetic gold docs instead of Label Studio hand-labeling (user's machine can't run Ollama; synthetic data unblocks all experiments)
  - E1/E2 designed to work without Ollama: token F1 against gold answers, leak check on redacted text
  - E2 "leak count" measures detection misses, not RAG leaks (RAG leak count is always 0 by design)
- Next: Phases 13 (NER fine-tuning, SHOULD) and 14 (Transformer classifier, COULD) remain TODO. Ollama install needed for live RAG demo.

### 2026-11-16, Phases 6–10 — Routing, Resolution, Redaction, Eval, RAG

- Did:
  - **Phase 6 — Routing + type rule packs**
    - `src/rta/detect/rules.py`: `for_type()` routing decision (single pack if conf >= 0.6, top-k fallback), `get_title_cues()` loads cues from YAML packs, `boost_ner_scores()` boosts NER PERSON spans +0.1 near title cues
    - Updated `src/rta/detect/regex/engine.py`: `load_packs()` now accepts `doc_types` list for multi-pack loading
    - Updated `src/rta/pipeline.py`: detect stages now use routing and title-cue boosting
    - `tests/test_rules.py`: 24 tests
  - **Phase 7 — Span resolution**
    - `src/rta/resolve/__init__.py`: `merge()` implements all 6 rules from §6.7 — same-label union (rule 3), validated-wins (rule 1), label precedence for exact range (rule 2), longer-wins for partial overlap (rule 4), min_score filtering (rule 5), whitespace/punctuation trimming (rule 6); greedy priority-based selection within overlap groups
    - `tests/test_resolve.py`: 17 tests including property test (no overlaps in output)
  - **Phase 8 — Redaction + pseudonymization**
    - `src/rta/redact/cluster.py`: `cluster_spans()` with 4-step matching — exact normalised match, token containment, initials (J. Smith → John Smith), fuzzy via rapidfuzz (token_set_ratio ≥ 90); `normalize_mention()` strips titles/possessives; digits-only normalisation for PHONE/SSN/CREDIT_CARD/IBAN
    - `src/rta/redact/surrogates.py`: `generate_surrogate()` with seeded Faker for consistent fake values per cluster
    - `src/rta/redact/__init__.py`: `apply()` supports mask/placeholder/fake modes; builds offset_map for original↔redacted alignment
    - `tests/test_redact.py`: 22 tests
  - **Phase 9 — Eval harness + Presidio baseline**
    - `src/rta/eval/match.py`: `match_strict()`, `match_overlap()`, `match_token()` matching modes
    - `src/rta/eval/metrics.py`: `precision_recall_f1()`, `f_beta()` (F2), `per_label_metrics()` with micro/macro, `leak_rate()`, `bootstrap_ci()` (1000 resamples)
    - `src/rta/eval/presidio_baseline.py`: `run_presidio()` wraps Presidio with taxonomy mapping
    - `src/rta/eval/report.py`: `generate_report()` writes CSV + Markdown, `generate_comparison_table()` for multi-system comparison
    - `src/rta/eval/__init__.py`: `score()` public API for experiments, `run_evaluation()` CLI entry point
    - `tests/test_eval.py`: 30 tests
  - **Phase 10 — RAG chat**
    - `src/rta/rag/chunk.py`: `chunk_text()` splits at sentence boundaries (pysbd, with regex fallback) with configurable overlap
    - `src/rta/rag/index.py`: `VectorStore` wrapping ChromaDB with sentence-transformers embeddings
    - `src/rta/rag/llm.py`: `chat()` calls Ollama HTTP API with proper error handling
    - `src/rta/rag/prompts.py`: `SYSTEM_PROMPT` with privacy rules, `build_rag_prompt()` templates
    - `src/rta/rag/__init__.py`: `index()` and `answer()` public API
    - `tests/test_rag.py`: 17 tests (all external deps mocked)
- Tests: **294 passed** (110 new: 24 rules + 17 resolve + 22 redact + 30 eval + 17 rag)
- Decisions:
  - Greedy priority selection for cross-label overlap resolution: sort by (validated, length, precedence, score), pick greedily — avoids complex pairwise tournament while respecting all 4 resolution rules
  - Mention clustering uses `importlib.util.find_spec` for rapidfuzz availability check to avoid unused-import lint
  - Defensive bounds check in `_trim_span` for spans that could extend beyond text length
- Next: Phase 11 — Streamlit demo

### 2026-11-13, Phase 5 — NER detector

- Did:
  - `src/rta/detect/ner/__init__.py`: `run(doc)` dispatches to spaCy or HF backend based on config `detect.ner.backend`
  - `src/rta/detect/ner/spacy_backend.py`: `detect(doc)` loads `en_core_web_trf` (fallback `en_core_web_lg`), maps entities via `SPACY_TO_LABEL`, sliding window (400 tokens/100 stride) for long docs, deduplicates
  - `src/rta/detect/ner/hf_backend.py`: `detect(doc)` loads HF token-classification pipeline with `aggregation_strategy="simple"`, maps via `HF_NER_TO_LABEL`, char-based sliding window, min_score filtering
  - `tests/test_ner.py`: 13 tests — spaCy detection (PERSON, ORG, GPE→LOCATION, unmapped label discard, alignment, dedup, model fallback), HF detection (entities, low-score filter, MISC discard), run() dispatch (spaCy, HF, invalid backend)
- Tests: **184 passed** (13 new)
- Decisions:
  - spaCy entities get default score 0.85 since spaCy doesn't expose per-entity confidence
  - Sliding window dedup by (start, end, label) tuple
- Next: Phase 6 — Routing + type rule packs

### 2026-11-13, Phase 4 — Regex detectors + validators

- Did:
  - `src/rta/detect/regex/validators.py`: 6 validators (email, ssn, luhn/credit-card, iban mod-97, ip address with loopback/link-local rejection, date multi-format); `VALIDATORS` registry dict for YAML lookup
  - `src/rta/detect/regex/engine.py`: `RegexRule` dataclass, `_load_pack()` compiles YAML patterns and caches, `load_packs(doc_type)` loads common + type-specific pack, `_detect_phones()` via `phonenumbers.PhoneNumberMatcher`, `scan()` runs all rules with validation (validated→0.99, failed→drop unless keep_unvalidated)
  - `src/rta/detect/regex/__init__.py`: `run(doc, doc_type)` public API wires up pack loading and scanning
  - `tests/test_regex.py`: 38 tests — validators (22), rule loading (3), detection for each PII type (email, SSN, credit card, IP, URL, date, age, MRN, insurance ID), span alignment, phone detection
- Tests: **171 passed** (38 new)
- Decisions:
  - Phone detection uses `phonenumbers` library (§6.5) instead of hand regex — more accurate for international formats
  - Loopback/link-local IPs rejected as not PII
  - All YAML rule packs load and compile successfully against real `configs/rules/*.yaml`
- Next: Phase 5 — NER detector

### 2026-11-13, Phase 3 — Document classification

- Did:
  - `src/rta/classify/corpus.py`: `build_corpus()` downloads ~300 docs per class from HF datasets (resume-atlas, CUAD, MTSamples, Enron emails) + TAB court cases; `strip_artifacts()` removes leakage cues (Enron headers, CUAD boilerplate); writes JSONL corpus
  - `src/rta/classify/baseline.py`: `train_model()` builds TF-IDF + LR sklearn Pipeline (config-driven params), saves with joblib, writes metadata JSON; `predict_text()` loads cached model and returns (doc_type, probs); `evaluate_model()` reproduces the same stratified split and runs classification report; handles sklearn ≥1.7 removing `multi_class` param
  - `src/rta/classify/__init__.py`: `predict(doc)` wraps baseline and populates `doc.meta` with `doc_type` and `type_probs`; `train()` auto-builds corpus if missing; `evaluate()` delegates to baseline
  - `tests/test_classify.py`: 13 tests — strip_artifacts (4), train+predict (3), evaluate (1), predict wrapper (1), train integration (1), court_case loader (2), build_corpus (1)
- Tests: **133 passed** (13 new)
- Decisions:
  - sklearn ≥1.7 removed `multi_class` from LogisticRegression; code introspects signature to stay compatible
  - Leakage-guard strips Enron headers and CUAD boilerplate via regex before training; both raw and stripped corpora can be built
  - HF dataset sources: `ahmedheakl/resume-atlas`, `theatticusproject/cuad`, `mteb/mtsamples`, `yangwang825/enron-email-corpus`
- Next: Phase 4 — Regex detectors + validators

### 2026-11-13, Phase 2 — Data acquisition + gold conversion

- Did:
  - `src/rta/data/ai4privacy.py`: `fetch()` downloads from HF, `convert()` streams, filters English, maps 27 label types via `AI4PRIVACY_TO_LABEL`, writes `GoldDoc` JSONL
  - `src/rta/data/tab.py`: `fetch()` downloads 3 splits from GitHub, `convert()` parses TAB JSON, uses annotator1 DIRECT+QUASI mentions, maps via `TAB_TO_LABEL`
  - Updated `src/rta/taxonomy.py`: `AI4PRIVACY_TO_LABEL` now covers all 27 real labels (GIVENNAME1, LASTNAME1, BOD, BUILDING, SOCIALNUMBER, etc.) verified from 500 streaming records
  - `tests/test_data.py`: 9 tests — both loaders with mock data, label filtering, NO_MASK exclusion, annotator fallback, max_docs caps, missing-file handling
- Tests: **120 passed** (9 new)
- Decisions:
  - ai4privacy fields verified by streaming inspection — `privacy_mask[].label` not `pii_type`; `source_text` not `text`
  - TAB: `annotations` is `{annotator_id: {entity_mentions: [...]}}`, not a flat list; use `annotator1`; skip `identifier_type == "NO_MASK"`
  - Relaxed pyproject.toml version pins from `~=` to `>=` to accommodate current PyPI packages
- Next: Phase 3 — Document classification

### 2026-11-13, Phase 1 — Ingest + language detection

- Did:
  - `src/rta/ingest/normalize.py`: Unicode NFC, line-ending unification, horizontal whitespace collapsing
  - `src/rta/ingest/pdf.py`: pymupdf 2.x (`import pymupdf`), page-by-page extraction, `\n\n` join, char-range tracking, empty-page warning
  - `src/rta/ingest/docx_loader.py`: python-docx, paragraphs + table cells in reading order
  - `src/rta/ingest/txt.py`: charset-normalizer encoding detection
  - `src/rta/ingest/__init__.py`: `load()` dispatcher by extension, plain-string wrapping, sha256 doc_id
  - `src/rta/langid/__init__.py`: lingua wrapper, ISO→Language mapping (34 languages), cached detector, `detect()` sets `doc.meta` keys, low-confidence warning
  - `tests/conftest.py`: fixture generators for PDF and DOCX (programmatic via pymupdf/python-docx), config+langid reset
  - `tests/fixtures/sample.txt`: English paragraph with fictional PII
  - `tests/test_ingest.py`: 29 tests (normalization, load string/txt/pdf/docx, page ranges, unsupported ext)
  - `tests/test_langid.py`: 7 tests (English, Spanish, meta population, low-confidence flag)
- Tests: **111 passed** (35 new)
- Next: Phase 2

### 2026-11-13, Phase 0 — Scaffold

- Did:
  - Created full directory tree (`src/rta/`, `configs/`, `tests/`, `app/`, `experiments/`, `reports/`, `data/`, `notebooks/`, `models/`)
  - `pyproject.toml`: hatchling build, all deps from §11 with `~=` compatible-release pins, four extras (`dev`, `app`, `baseline`, `train`), `rta` CLI entry point
  - `Makefile`: setup / test / lint / fmt / data / eval / experiments / demo / clean targets
  - `.pre-commit-config.yaml`: ruff check + ruff format hooks (v0.7.1)
  - `.streamlit/config.toml`: teal/ink/amber theme from §10
  - `src/rta/schema.py`: frozen `Span`, mutable `Document`, `RedactionResult`, `GoldDoc`; `to_dict`/`from_dict` round-trips; JSONL helpers; invariant checked in `__post_init__`
  - `src/rta/taxonomy.py`: 16-label `LABELS` set, `DIRECT_IDENTIFIERS`/`QUASI_IDENTIFIERS` partition, `LABEL_PRECEDENCE` list, six mapping dicts (spaCy, HF, ai4privacy, TAB, CoNLL, Presidio), `normalize_label`, `label_precedence_rank`
  - `src/rta/config.py`: `AttrDict` attr-access wrapper, `Config` singleton, `load()`/`get_config()`/`reset()`, walks up to find `configs/default.yaml`
  - `configs/default.yaml`: all settings from §13; `configs/rules/{common,medical,contract,resume,email,court_case}.yaml`
  - `src/rta/pipeline.py`: `PipelineResult` dataclass, `Pipeline` with lazy stage dispatchers and `overrides` injection
  - `src/rta/cli.py`: full Typer CLI skeleton (detect, redact, chat, eval, classify, data, labelstudio sub-commands)
  - Stub `__init__.py` for all subpackages with `NotImplementedError` stubs and correct signatures
  - `tests/test_schema.py` (53 tests), `tests/test_config.py` (20 tests), `tests/test_taxonomy.py` (16 tests)
  - `app/Home.py` placeholder, `app/style.css` with full palette + font CSS
  - `experiments/e1–e4` stubs, `reports/RESULTS.md` table skeleton
- Tests: **76 passed, 0 failed** (`pytest tests/`)
- Numbers: none yet (scaffold only)
- Next: Phase 1 — Ingest + language detection (PDF/DOCX/TXT → `Document`; `lingua` wrapper; fixture files)
- Blockers: none

<!-- Template:
### YYYY-MM-DD, Phase N
- Did:
- Tests: X passed
- Numbers:
- Next:
- Blockers:
-->

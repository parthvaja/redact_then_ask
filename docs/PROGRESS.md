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
| 6 | Routing + type rule packs | MUST | TODO | Member 2 | |
| 7 | Span resolution | MUST | TODO | Member 3 | |
| 8 | Redaction + pseudonymization | MUST | TODO | Member 3 | |
| 9 | Eval harness + Presidio baseline | MUST | TODO | Member 4 | |
| 10 | RAG chat | MUST | TODO | Member 4 | |
| 11 | Streamlit demo | MUST | TODO | | |
| 12 | Experiments E1, E2 (MUST), E3 (SHOULD) | MUST | TODO | Member 4 | |
| 13 | NER fine-tuning | SHOULD | TODO | Member 3 | |
| 14 | Transformer classifier + multilingual | COULD | TODO | Member 1 | |
| 15 | Final polish + RESULTS.md | MUST | TODO | all | |

## Human tasks (Claude Code cannot do these)

| Task | Needed by phase | Status | Who |
|---|---|---|---|
| Kaggle API token (if resume dataset needs it) | 3 | TODO | |
| Correct pre-annotations for 30 hand-labeled docs in Label Studio | 9 | TODO | split 8/8/7/7 |
| Double-annotate 5 docs for agreement | 9 | TODO | |
| Write 100 QA pairs from `data/gold/qa_template.jsonl` | 12 | TODO | 25 each |
| Install Ollama and pull the model on the demo machine | 10 | TODO | |
| Fill team names in README | 15 | TODO | |

## Key numbers (filled as produced)

| Metric | Value | Source file |
|---|---|---|
| Classifier accuracy / macro-F1 | | |
| Presidio direct-ID recall (ai4privacy-en) | | |
| Ours direct-ID recall (ai4privacy-en) | | |
| Ours direct-ID recall (TAB) | | |
| Ours direct-ID recall (hand-labeled) | | |
| QA accuracy L0 / L1 / L2 / L3 | | |
| Leak count (E2) | | |
| Biggest error-propagation cost (E3) | | |

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

# Progress log

Claude Code updates this file at the end of every session. Humans can edit it too. Newest session entries go at the top of "Session log".

Status values: `TODO` · `IN PROGRESS` · `BLOCKED (reason)` · `DONE (commit sha)`

## Phase status

| # | Phase | Priority | Status | Owner | Notes |
|---|---|---|---|---|---|
| 0 | Scaffold | MUST | DONE | Claude Code | |
| 1 | Ingest + language ID | MUST | TODO | Member 1 | |
| 2 | Data acquisition + gold conversion | MUST | TODO | Member 4 | |
| 3 | Document classification | MUST | TODO | Member 1 | |
| 4 | Regex detectors + validators | MUST | TODO | Member 2 | |
| 5 | NER detector | MUST | TODO | Member 3 | |
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

## Open issues

_None yet._

## Session log

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

# CLAUDE.md: working rules for this repo

Claude Code reads this file automatically at the start of every session. Keep it short and current.

## Project in one line

**Redact, Then Ask**: detect and pseudonymize PII in documents, then chat with the redacted text through a local LLM, and measure what privacy costs in usefulness. DATA 622 (NLP), UMBC, Fall 2026.

## Session start ritual

1. Read `docs/PROGRESS.md` and find the first phase not marked DONE.
2. Skim the matching section of `docs/PROJECT_DOC.md`.
3. Run `make test` to confirm the tree is green before changing anything.

## Session end ritual

1. `make lint test` must pass.
2. Update `docs/PROGRESS.md`: phase status, what changed, any numbers produced, open issues, decisions.
3. If behavior or interfaces changed, update `docs/PROJECT_DOC.md` in the same commit.
4. Commit with a conventional message: `feat|fix|test|docs|chore|exp(scope): summary`.

## Non-negotiable rules

- **Privacy:** never send document text to an external API. The LLM is Ollama on localhost. Do not add OpenAI, Anthropic or other hosted-LLM clients.
- **One span format.** Everything uses `rta.schema.Span`. Offsets are character offsets into `Document.text`, end-exclusive.
- **Taxonomy is fixed** in `docs/PROJECT_DOC.md` §5. Map every external label set to it in `src/rta/taxonomy.py`; never leak raw dataset labels past the loaders.
- **Config, not constants.** Thresholds, model names, paths and rule packs live in `configs/`. Code reads them through `rta.config`.
- **Reproducible.** Seed 13 everywhere. Pin versions in `pyproject.toml`. Cache downloads under `data/` and `models/`, both gitignored.
- **Generated, not hand-made.** Every figure and table in `reports/` is produced by a script in `experiments/` or `src/rta/eval/`.
- **Recall first.** When a design choice trades precision for recall on PII detection, prefer recall and note it.

## Commands

```bash
make setup        # create venv, install deps, download spaCy model
make test         # pytest
make lint         # ruff check + ruff format --check
make data         # download + convert datasets (Phase 2)
make eval         # run evaluation harness, write reports/
make experiments  # run all experiments
make demo         # streamlit run app/Home.py
rta --help        # CLI
```

## Layout cheat sheet

```
src/rta/         package (ingest, langid, classify, detect, resolve, redact, eval, rag, pipeline, cli)
configs/         default.yaml + rules/<doctype>.yaml
app/             Streamlit demo
experiments/     one script per experiment
data/            raw/ processed/ gold/   (gitignored except SOURCES.md and gold/README.md)
models/          cached and fine-tuned weights (gitignored)
reports/         figures/ tables/ RESULTS.md
docs/            PROJECT_DOC.md, PROGRESS.md
tests/           mirrors src/rta, plus fixtures/
```

## Style

Python 3.11, type hints, `from __future__ import annotations`, dataclasses or pydantic for records, `pathlib` over `os.path`, `logging` over `print` in library code, small pure functions, no global state except the config singleton.

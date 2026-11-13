# Dataset Sources

All datasets used in this project are public or synthetic. No real patient, client, or personal data is used. Download instructions are below; actual files are gitignored.

## ai4privacy / pii-masking-300k

| Field | Value |
|---|---|
| URL | https://huggingface.co/datasets/ai4privacy/pii-masking-300k |
| License | CC BY 4.0 |
| Use | Main PII train / eval |
| Download | `rta data fetch` or `python -m rta.data.ai4privacy` |
| Local path | `data/raw/ai4privacy/` |
| Processed | `data/processed/ai4privacy_en_test.jsonl` |

Filter: `language == "English"` for primary results. Print one record first to verify field names before writing the loader.

## Text Anonymization Benchmark (TAB)

| Field | Value |
|---|---|
| URL | https://github.com/NorskRegnesentral/text-anonymization-benchmark |
| License | CC BY 4.0 |
| Use | Legal-domain PII test, court_case classifier class |
| Download | `rta data fetch` or `python -m rta.data.tab` |
| Local path | `data/raw/tab/` |
| Processed | `data/processed/tab_test.jsonl` |

Annotation types: `DIRECT` and `QUASI` identifiers. Map `type_` field via `rta.taxonomy.TAB_TO_LABEL`.

## CoNLL-2003

| Field | Value |
|---|---|
| URL | https://huggingface.co/datasets/conll2003 |
| License | Public domain |
| Use | PERSON / ORG / LOC sanity check on NER |
| Download | Automatic via HF datasets |
| Processed | `data/processed/conll_test.jsonl` |

Not PII-specific; use only as a secondary NER sanity check.

## Classification Corpora

Downloaded and used only for classifier training (Phase 3). Not annotated for PII.

| Corpus | Source | Label |
|---|---|---|
| Resume Dataset | HF `ahmedheakl/resume-atlas` (fallback if Kaggle unavailable) | resume |
| CUAD contracts | HF `theatticusproject/cuad` | contract |
| MTSamples | HF `lavita/medical-qa-datasets` or similar | medical |
| Enron emails | HF `Enron Email Dataset` mirror | email |
| ECHR (TAB) | Same as TAB download above | court_case |

## Hand-labeled Set (30 documents)

| Field | Value |
|---|---|
| Use | End-to-end gold standard; IAA measurement |
| Local path | `data/gold/` |
| Format | Label Studio JSON → `GoldDoc` JSONL |
| Status | **Human task** — see `docs/PROGRESS.md` |

6 documents per type (resume, contract, medical, email, court_case). All content is synthetic and fictional. Double-annotated 5 documents for Cohen's κ.

## QA Set (100 pairs)

| Field | Value |
|---|---|
| Use | Privacy-utility experiment (E1) |
| Local path | `data/gold/qa_pairs.jsonl` |
| Template | `data/gold/qa_template.jsonl` |
| Status | **Human task** — 25 pairs per team member |

See `data/gold/qa_template.jsonl` for the format.

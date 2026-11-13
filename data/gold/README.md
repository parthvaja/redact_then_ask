# Gold data

Hand-labeled documents and QA pairs live here. Everything except this README and `qa_template.jsonl` is gitignored.

- `handlabel/*.json`: Label Studio exports, converted to `GoldDoc` JSONL by `rta labelstudio import`.
- `qa.jsonl`: 100 QA pairs (20 per document type), copied from `qa_template.jsonl`.
- All documents are synthetic. Do not put real personal data in this folder.

# E1: Privacy vs. Utility

| Level | Description | QA Token F1 (non-PII) | QA Token F1 (PII-dep) | PII Recall |
|---|---|---|---|---|
| L0 | No redaction | 0.5490 | 0.0000 | 0.0000 |
| L1 | Regex only | 0.5490 | 0.3061 | 0.4120 |
| L2 | Full pipeline (placeholder) | 0.5490 | 0.3061 | 0.4120 |
| L2-mask | Full pipeline (mask) | 0.5490 | 0.0000 | 0.4120 |
| L3 | Aggressive (min_score=0.15) | 0.5490 | 0.3061 | 0.4120 |

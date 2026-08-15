# Data provenance & split protocol

## RAID (primary benchmark)

- Source: Dugan et al., ACL 2024 — *"RAID: A Shared Benchmark for Robust
  Evaluation of Machine-Generated Text Detectors"* (arXiv:2405.07940), MIT
  license, `github.com/liamdugan/raid`.
- Local raw: `data/raw/raid/train_none.csv` (802M, clean split, no
  adversarial variants) downloaded 2026-08-14 from
  `https://dataset.raid-bench.xyz/train_none.csv`. **Immutable** (RULES #1).
- Human docs: ~15K real pre-2022 documents across 8 domains (news, books,
  abstracts, reviews, reddit, recipes, wikipedia, poetry) — no AI
  contamination in the well (paper §3.2). AI rows: 11 LLMs × 4 decoding
  strategies × repetition-penalty variants.

## Split protocol (RULES #3/#4)

- Unit of splitting is the **source_id**, never the row: every generation of
  a source stays in that source's fold. No near-duplicate leakage.
- `scripts/make_splits.py` → `data/derived/raid_splits.parquet`, deterministic
  (sha256-salted ordering), default 2,000 sources to **dev**, ~13K to
  **holdout**.
- Dev is for feature work + threshold calibration. Holdout is evaluated once,
  at the end. Paraphrase/robustness sets never enter tuning (RULES #4).
- Metric: TPR at fixed FPR (default 1e-3) with confidence intervals
  (RULES #3). Never bare accuracy.

## Adversarial data

RAID's own paraphrase rows (DIPPER/T5-11B) live only in the 11.8G
train.csv — deferred. Programmatic attacks (homoglyph, zero-width, synonym,
whitespace, article deletion, upper-lower, number shuffle, alternative
spelling) we generate ourselves from dev/holdout-clean rows when needed.
Paraphrased-*human* text (the false-positive trap) does not exist in RAID;
if we want it, we synthesize it ourselves and keep it as a fourth eval
category.

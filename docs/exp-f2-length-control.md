# F2: length control on the 9-feature pilot

Question: how much of the pilot's AUROC (0.898 logreg / 0.925 HGB, dev long-doc
subset) is real signal vs the document-length artifact? Repro:
`scripts/exp_f2.py` (dev fold only; humans 2000 + AI 4000 rs=17; 50/50
source-disjoint split rs=23; usable = 1131 docs with >=350 tokens).

## Confound confirmed

- Featurizable AI docs: zero >=500 tokens; humans: 195 (of 371 usable).
- Token count alone scores AUROC 0.854 (as "long => human"; raw 0.146).
- `qgram_total` IS length: total == n_bytes - q + 1; Spearman rho vs token
  count +0.694, solo AUROC 0.162 (i.e. 0.838 flipped). Length is redundantly
  encoded elsewhere too: midrange_stdev rho=-0.504, qgram_distinct_ratio
  rho=-0.414, midrange_mean rho=+0.406.

## Three measurements (test AUROC | TPR@1e-3)

| config | logreg | hgb |
|---|---|---|
| A full, biased subset | 0.898 \| 0.20 | 0.925 \| 0.26 |
| B minus qgram_total | 0.899 \| 0.21 | 0.924 \| 0.30 |
| C1 matched test (bins 350-375/375-400/400-450/450-500, class-balanced) | 0.744 \| 0.27 | 0.795 \| 0.34 |
| C2 matched train+test | 0.755 \| 0.37 | 0.726 \| 0.40 |

Matched test n=136 (68/68) -> AUROC noise ~±0.04; per-bin AUROC positive in
every bin (0.60-0.82, hgb strongest), worst at shorter lengths.

## Verdict

- ~0.13-0.15 AUROC points (roughly a third of the above-chance margin) are
  length artifact. Real signal survives at AUROC ~0.73-0.80.
- Ablating qgram_total costs NOTHING (0.898->0.899, 0.925->0.924): the ruler is
  redundantly encoded. **Drop qgram_total from the feature set** — it's byte
  length in a trenchcoat, adds zero incremental skill, and invites artifact
  evals. If length is ever wanted, add it explicitly as `ntok` and say so.

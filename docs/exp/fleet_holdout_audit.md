# FLEET D — holdout behavior audit

parquet columns: ['id', 'source_id', 'model', 'decoding', 'domain', 'title', 'generation', 'fold']
folds: {'holdout': 397985, 'dev': 70000}

## P1/P2/P3 — AI variant mix & multiplicity

dev AI rows (capped): 4000 | holdout AI sample: 20000
AI rows/source: dev mean 2.00 max 2 | holdout mean 2.09 max 8

model mix (dev-capped vs holdout-sample):
              dev  holdout
model                     
chatgpt       0.0   0.0602
cohere        0.0   0.0592
cohere-chat   0.0   0.0560
gpt2          0.0   0.1190
gpt3          0.0   0.0634
gpt4          0.0   0.0562
llama-chat    1.0   0.1185
mistral       0.0   0.1158
mistral-chat  0.0   0.1164
mpt           0.0   0.1184
mpt-chat      0.0   0.1170

decoding mix (dev-capped vs holdout-sample):
          dev  holdout
decoding              
greedy    0.5    0.504
sampling  0.5    0.496

slot histogram of dev-capped AI rows (parquet within-source order): {0: 2000, 1: 2000}
total AI slots available per dev source: {34: 2000}

## P4 — holdout cache row alignment (200-row recompute)

hu: 100 rows recomputed vs cache cols 8:20 — mismatches counted so far: 0
ai: 100 rows recomputed vs cache cols 8:20 — mismatches counted so far: 0
P4 verdict: ALIGNED

## P5 — fold disjointness

dev/holdout shared sources: 0 (must be 0)
dup generation texts across folds: 492 (exact dups)

## P6 — null controls (half-vs-half AUROC, expect ~0.5)

human qg_mid_qgram_mean half/half AUROC: 0.4912 (n=2070)
ai qg_mid_qgram_mean half/half AUROC: 0.4928 (n=3715)

## P7 — dev-C AUROC of qg_mid_qgram_mean, capped vs multiplicity-emulated

dev C capped-bucket AUROC (oriented): 0.9741 (holdout read was 0.879)
(see P1 mix table for whether dev buckets systematically drop hard variants)

## P8 — per-model AUROC on holdout (qg_mid_qgram_mean)

  chatgpt        AUROC 0.890 (n=354)
  cohere         AUROC 0.844 (n=214)
  cohere-chat    AUROC 0.861 (n=80)
  gpt2           AUROC 0.869 (n=1055)
  gpt4           AUROC 0.898 (n=271)
  llama-chat     AUROC 0.969 (n=146)
  mistral        AUROC 0.915 (n=443)
  mistral-chat   AUROC 0.944 (n=54)
  mpt            AUROC 0.860 (n=944)
  mpt-chat       AUROC 0.827 (n=117)

(dev buckets trained/evaluated on llama-chat ONLY — see P1.)

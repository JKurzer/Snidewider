# SOTA operating points — what FPR does the field actually report at?

Checked 2026-08-16 against primary sources (RAID paper PDF arXiv:2405.07940v3,
RAID repo README, MAGE repo README). Question: is our 1e-3 floor ridiculous?

## RAID (Dugan et al., ACL 2024) — our primary benchmark

- **Official metric: accuracy at fixed FPR = 5%.** `evaluate_cli.py
  --target_fpr` default **0.05**. Paper §5.2: "accuracy at a fixed FPR of 5%
  ... following the rise of this evaluation paradigm" (citing Hans/Binoculars,
  Krishna, Soto). "Accuracy" = TPR at the calibrated threshold.
- **Calibration is a ROC read**: per-detector threshold chosen by iterative
  search on RAID's own human portion (Appendix C, eps=0.0005) — i.e. the
  leaderboard convention recalibrates on eval data; our holdout ROC ladder
  (scripts/exp/holdout_fpr_ladder.py) is methodologically identical.
- SOTA at FPR=5% (Table 5, clean text): Binoculars 99.9, GPTZero 98.8,
  Originality 98.6, F-DetectGPT 98.6 accuracy. BUT collapse is one decoding
  change away: repetition-penalty sampling drops Binoculars to 0.6 (!) on
  affected slices; Finding 4 = "perfect accuracy to complete failure".
- **Finding: "few detectors can operate at FPR<1%"** — the field's own words.
  I.e. nobody reports at 1e-3 because nearly everything is blind there.

## MAGE (Li et al.)

- No fixed FPR at all: **HumanRec / MachineRec / AvgRec (= balanced accuracy
  at the detector's own threshold) + AUROC**.
- Their Longformer: in-distribution AvgRec 96.6 / AUROC 0.99; unseen domains
  HumanRec 38.1; paraphrase MachineRec 37.1. Same story: shift kills.

## Other reference points

- Krishna et al. (DIPPER paraphrase eval): **TPR @ 1% FPR** — the adversarial
  paper convention. See docs/aigt-survey.md.
- **Binoculars (Hans et al. 2024, arXiv:2401.12070) — the low-FPR king:
  headline claim is >90% TPR at FPR = 0.01% (1e-4)**, zero-shot, no training
  data (on its own eval mix, not RAID holdout). RAID Fig 4 confirms it is the
  strongest low-FPR curve of their 12.
- OpenAI AI Text Classifier (2023, discontinued): shipped at ~**9% FPR**,
  26% TPR. The cautionary tale at the loose end.
- Commercial claims (Turnitin et al.): <1% claimed, unaudited.

## Verdict for us

| operating point | who uses it | ensemble (pre-fix) | ensemble (honest feats) | **L0 champ (120-feat panel)** |
|---|---|---|---|---|---|
| 5% FPR | **RAID official**, field norm | 0.272 | 0.260 | **0.702** [0.696, 0.708] |
| 1% FPR | adversarial-paper norm | 0.113 | 0.121 | **0.536** [0.530, 0.543] |
| 0.1% (1e-3) | our RULES default | 0.033 | 0.047 | **0.124** [0.119, 0.129] |
| 0.01% (1e-4) | Binoculars headline (>90% TPR, own data) | below resolution (k=1) | below resolution | below resolution (k=1) |

Holdout AUROC: ensemble 0.7861, **L0 champ 0.9103**. The 120-feat L0 champion
is the first model whose dev->holdout tax is ~1.1x instead of ~2.8x (dev C
0.919/0.597 -> holdout 0.910/0.536). Champion = 25 feats spanning every
family: qg_mid, ex_contrast, ALL 9 coverage, 4 stats, shape, dct_bands, rel.
CAUTION on the ensemble numbers: the 0.8203/0.217 interim read was inflated
by exemplar bank contamination (no leave-one-out); the honest-feats ensemble
drops to 0.7861/0.121.

**2026-08-16 protocol fix (fleet_holdout_audit): dev buckets had been
100% llama-chat (parquet model-ordering + head(2)); the "2.8x tax" was
this artifact. After seeded-uniform AI slot sampling: holdout AUROC
0.7113 -> 0.8203, TPR@1e-2 0.113 -> 0.217.** The mix-trained ensemble buys
mid-tail at the cost of deep tail (1e-3: 0.033 -> 0.015); the fleet-A2
long-doc stat (0.200 @1e-3) remains the deep-tail record. L0 champ is
mix-robust already (0.7084 AUROC post-fix, selected on stratified B).

Our 1e-3 floor is ~50x stricter than the benchmark norm, but NOT the
extreme end: Binoculars publishes at 1e-4. The floor is defensible as
deployment philosophy (false accusations are the catastrophic error); as a
*measurement* default it starves statistics (k=0 at 750 humans/bucket).
Resolution: 1e-3 needs >=10K humans — holdout has 11,371 (k=11), so the
holdout CAN speak at 1e-3, dev buckets CANNOT.

**Reporting ladder going forward: 5e-2 (field-comparable), 1e-2, 1e-3
(stretch corner, holdout only). Never bare accuracy; always Wilson CIs.**

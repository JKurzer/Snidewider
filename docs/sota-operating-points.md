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

## Final holdout table (2026-08-16, post all fixes: stratified mix,
exemplar LOO, cross-bucket coverage refs)

| model | AUROC | TPR@5e-2 | TPR@1e-2 | TPR@1e-3 |
|---|---|---|---|---|
| **panel HGB (120 feats)** | **0.9706** | **0.878** | **0.711** | **0.456** |
| L0 champ (12 feats) | 0.9473 | 0.808 | 0.639 | 0.133 |
| HGB-stack (hgb-as-feature) | 0.9669 | 0.870 | 0.663 | 0.198 |
| L0-stack | 0.9669 | 0.862 | 0.644 | 0.430 |
| exam ensemble (81-feat families) | 0.7861 | 0.260 | 0.121 | 0.047 |

Dev<->holdout tax is ~1x across the board now. The stacked-generalizer
design (HGB score as a downstream feature) was tried and does NOT beat the
plain panel HGB once coverage is honest - composted. TPR@1e-3 0.456 is the
repo deep-tail record. At the RAID 5% norm we're at 0.878 TPR - inside the
SOTA conversation (their ~80-99 acc band) with classical features.

(Pre-fix table preserved in git history; interim 0.82/0.91 reads were
inflated by exemplar/coverage contamination, see fleet_holdout_audit +
fleet_stats_stack + probe_crossfit docs.) The 120-feat L0 champion
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

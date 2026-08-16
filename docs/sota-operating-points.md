# SOTA operating points — what FPR does the field actually report at?

Checked 2026-08-16 against primary sources (RAID paper PDF arXiv:2405.07940v3,
RAID repo README, MAGE repo README). Question: is our 1e-3 floor ridiculous?

## RAID (Dugan et al., ACL 2024) — our primary benchmark

- **Official metric: accuracy at fixed FPR = 5%.** `evaluate_cli.py
  --target_fpr` default **0.05**. Paper: "accuracy at a fixed FPR of 5% ...
  following the rise of this evaluation paradigm."
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
- OpenAI AI Text Classifier (2023, discontinued): shipped at ~**9% FPR**,
  26% TPR. The cautionary tale at the loose end.
- Commercial claims (Turnitin et al.): <1% claimed, unaudited.

## Verdict for us

| operating point | who uses it | our status |
|---|---|---|
| 5% FPR | **RAID official**, field norm | not yet read on our holdout |
| 1% FPR | adversarial-paper norm | not yet read on our holdout |
| 0.1% (1e-3) | **nobody**; RAID: nearly nothing works there | our RULES default |

Our 1e-3 floor is ~50x stricter than the benchmark norm and makes us
incomparable to every published number. As deployment philosophy it's
defensible (false accusations are the catastrophic error); as a *measurement*
default it starves statistics (k=0 at 750 humans/bucket) and hides us from
the field. Resolution: 1e-3 needs >=10K humans — holdout has 11,371 (k=11),
so the holdout CAN speak at 1e-3, dev buckets CANNOT.

**Reporting ladder going forward: 5e-2 (field-comparable), 1e-2, 1e-3
(stretch corner, holdout only). Never bare accuracy; always Wilson CIs.**

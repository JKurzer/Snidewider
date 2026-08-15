# F4: exemplar-proximity (doc-vs-corpus) features

Hypothesis: AI text resembles other AI text; human text is more diverse.
First structural variant beyond self-similarity (features.py): per-doc
q-gram distance to TRAIN-half exemplar banks (190 AI + 190 human, q=3,
profiles computed once; leave-one-out for bank members).

## Headline (dev fold, biased long-doc subset, n_test=557)

| features            | model  | AUROC | TPR@1e-3          |
|---------------------|--------|-------|-------------------|
| exemplar-only (11)  | logreg | 0.980 | 0.54 [0.49, 0.59] |
| exemplar-only (11)  | hgb    | 0.981 | 0.63 [0.58, 0.68] |
| baseline-9 (ref)    | logreg | 0.898 | 0.20 [0.17, 0.25] |
| baseline-9 (ref)    | hgb    | 0.925 | 0.26 [0.22, 0.31] |
| combined (20)       | logreg | 0.986 | 0.70 [0.65, 0.75] |
| combined (20)       | hgb    | 0.988 | 0.60 [0.55, 0.65] |

Baseline replicated exactly (0.898/0.925) before comparison. Protocol:
dev only, humans=all + AI n=4000 (rs=17), NaN-drop (1131 usable of 6000 —
the long-doc bias is severe), 50/50 by source_id (rs=23), banks rs=101.

## Per-feature AUROC (test; <0.5 = negatively directed)

- Contrast (ai_dist - hu_dist) carries: min 0.958, mean 0.954, p10 0.946 sep.
- Distance-to-human-bank alone separates: min 0.879, p10 0.808, mean 0.774
  (AI docs are FAR from the human bank).
- Distance-to-AI-bank alone is weak: min 0.595 sep, mean 0.553 — everyone is
  close to some AI exemplar; "looks AI-ish" isn't distinctive, "looks
  human-ish" is. Raw (unnormalized) means are length-confounded (0.66-0.76
  sep); normalized = raw / (total_doc + total_ex) is the principled form.
- Top combined logreg coefs: ex_contrast_p10/mean/min, then ex_ai_min,
  qgram_repeat_frac.

## Compute (i7, native qgram)

Profiles 0.3 ms/doc; bank diffs 43 ms/doc (380 diffs, ~113 us each; 1131
docs -> 46 s). Bank build 0.1 s. Whole experiment 54 s. Exemplar count was
pool-capped at 190 (only 190 train-half humans survive the long-doc filter)
— the 200-exemplar budget was never compute-limited.

## Caveats

n_test=557 -> wide Wilson CIs; subset is long-doc-biased; human bank covers
190/190 train humans (LOO keeps it honest). Holdout untouched.
Repro: `scripts/exp_f4.py` on branch exp/f4-exemplar-search.

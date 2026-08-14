# Knowledge Corpus — what each reference means for the detector

Source material: `C:\Users\poly\OneDrive\Documents\StatisticalMeasuresReferences`
(immutable reference; notes here are the distilled, buildable concepts).

## The stack, bottom to top

| Layer | Tool | Role | Cost |
|---|---|---|---|
| Char-level distance | **CK2** (`ck2.py`) | Fast Levenshtein-family signal; near-duplicate/paraphrase proximity | O(n) |
| Char-level profile | **q-gram** (`qgram.py`, Ukkonen 1992) | Multiset overlap of length-q substrings; edit-distance lower bound | O(n) |
| Char-level bag | **bag distance** (`qgram.py`, Bartolini 2002) | Same idea, q=1 (character multiset); weakest, cheapest | O(n) |
| Substring search | **Hanada** (`qgram.search`, 2014 + deque fix) | All substrings of a corpus within d_q ≤ k of a pattern | avg O(\|t\|+\|p\|) |
| Sentence semantics | **DCT encoder** (Almarwani & Diab 2021) | Training-free, word-order-sensitive sentence vectors | O(N·d·K) |
| Statistical LM signal | **perplexity/entropy/rank** family (survey: Fraser et al. 2024) | "AI text is unsurprising text" — needs a proxy LM | per-token LM cost |

Notes per source:

- `cks-ck2.md` — the vendored headers, semantics, measured speed, caveats
- `qgram.md` — Ukkonen 1992 + Hanada 2014 + implementation decisions
- `hanada-array-base-search.md` — profiling diagnosis (VERIFIED)
- `hanada-verification.md` — independent claim-by-claim verification
- `hanada-deque-fix.md` — the fix, before/after numbers, CK2-scan comparison
- `dct-encoder.md` — ACL 2021 paper distilled
- `aigt-survey.md` — the 45-page NRC survey distilled to what changes our decisions

## Doctrine (binds all layers)

Per RULES.md: detectors ship with evals; report **TPR at fixed FPR (default 1e-3) with
CIs**, never bare accuracy; split by source document, never by chunk; paraphrase sets
held out from tuning; baselines before neural. The survey (Fraser et al.) independently
endorses TPR@FPR reporting — good, the field agrees with the floor.

Concept: no single feature survives contact with adversarial paraphrasing. The plan is
cheap, *diverse* signals (char-level + sentence-level + statistical) combined in a
calibrated ensemble with per-domain thresholds — the same conclusion the survey reaches
in Section 6.

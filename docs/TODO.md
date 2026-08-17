# TODO — Snidewider feature/method backlog

Ordered by Donk, 2026-08-16. Each item links its evidence trail.

## Feature packs (fleet candidates)

1. **Comma-context / situational bigrams** (Jin–Zaitsu lineage via
   Ghatpande et al. 2026 review, docs/ghatpande2026-lr-fusion.md).
   English pack: comma-following-word-class profile, Oxford-comma rate,
   comma-splice tendency, quote/period-adjacency patterns. Cheap, single-pass.
2. **Repetitiveness measures** (Puglisi orbit: Bannai/Inenaga CPM 2026
   "Sensitivity of Repetitiveness Measures to String Reversal"; Lipták/
   Masillo/Puglisi 2026 "Matching statistics — a survey").
   - δ = max_k (#distinct k-mers / k) — we already compute q-gram profiles;
     a per-doc δ is nearly free. AI text = more repetitive = lower δ.
   - matching-statistic distribution of a doc vs the pooled human reference
     (per-position longest-match lengths): a "statistical condensate" proper.
3. **Alliteration/phonotactic rate** (Donk's whitespace find — nobody in the
   lit does this): initial-phoneme repeat rate per sentence; humans avoid
   alliteration except as device, AI doesn't modulate. Needs a light
   phoneme approximation (letter-pair onset proxy acceptable for v1).
4a. **Structure-entropy pack** (docs/equi2026-qpm-review.md): sam_states,
   bwt_runs, lz77_phrases, delta — the size of a doc's minimal index IS its
   entropy; the principled upgrade of zlib_ratio. SAM build for the ms_
   family makes sam_states a free rider.
4. **De Bruijn "repair depth" aberration** (LoRMA-inspired, Salmela/Walve/
   Rivals/Ukkonen 2017): treat the reference corpus's k-mer counts as the
   graph; measure how much of a test doc sits on rare-vs-common edges.
   Richer than binary coverage membership. v1 can be a soft-count variant
   of cov*_contrast (log-count-weighted hits).

## Eval/method debt

5. **Cllr / log-LR calibration** of the headline scores (forensic framing;
   logistic calibration on B, read on C; docs/ghatpande2026-lr-fusion.md).
6. **Legacy dup pairs**: rel_qgram_* ≡ qg_q3_* (4 exact pairs) — disentangle
   the F1/F3 family definitions at the next planned cache rebuild.
7. **RAID leaderboard adapter** (pickled panel HGB + raid-bench harness;
   pre-flight: verify test/train source disjointness; reproduce our ladder
   through their run_evaluation first).

## Watch list

- Ukkonen: quiet since 2020 (motif discovery, bioinformatics).
- Puglisi: very active — matching statistics, repetitiveness, succinct
  rank dictionaries, spectral BWT (with Bille/Gørtz at DTU and Bannai's
  Kyushu group). Watch for anything crossing into text statistics.
- Salmela: Sama assembler w/ correctness guarantees (2025). Mäkinen:
  compressed indexing/pangenomics. Kärkkäinen: quiet.

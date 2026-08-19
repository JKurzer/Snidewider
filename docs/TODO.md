# TODO — Snidewider feature/method backlog

Ordered by Donk, 2026-08-16. Each item links its evidence trail.

## 0b. MEASURED NEGATIVE — adversarial mass-mixing (2026-08-19)

231K A-source attack rows folded into training (98.7% of mass): clean
holdout cratered (tail 0.754 -> 0.501) AND attacks did not improve
(paraphrase flat at 0.608; homoglyph 0.777 -> 0.340 WORSE; zero_width_space
0.237 alarm). Mass-mixing re-aims the model at the attacked distribution
rather than teaching boundary robustness. Supplement benched
(supp_features.npz.benched); the clean 261 d8/i600 champion restored
(0.9935/0.974/0.912/0.754). Robustness needs a different mechanism
(per-slice gates, attacked-input normalization, or a small adversarial
head), not bulk exposure.

## 0a. MOSAIC — paraphrase-mosaic detection (Donk, 2026-08-18)

Paraphrase attacks (preview: our weakest slice, 0.608) interleave TWO AI
profiles: original-generator spans + paraphraser-output spans (DIPPER-class
span substitution leaves the remainder untouched). Global features average
the mosaic and dilute; the counter is to read the SEAMS. Fleet sketch:
windowed local score series (burst/ck2 machinery), then bimodality/gap
structure of the series (variance, two-mode gap, frac-low ~ 2p(1-p) for
paraphrase fraction p), interleave frequency. Bonus angle: the paraphraser
is itself an LLM — rewritten spans carry ITS fingerprint, so the mosaic is
two machine clusters, not machine-vs-human. qg_s256_ck2 vector emission
(sorted-series profile) is the instrument; bench on the cached 5.6M-row
train paraphrase slice.

## 0. PURSUED — Kolmogorov proxies (fleet_dragon.py, 2026-08-16)

The ladder was climbed: Kasai-LCP self-similarity stats, Re-Pair grammar
size, SAM transition counts. OUTCOME: individually near coin-flip
(0.51-0.53 solo), dragon6 alone 0.773/0.177, zero increment over the panel
— absolute complexity varies by domain/length more than by generator on
this mix; the signal lives in REFERENCE-RELATIVE measures. Kept as
fleet-benched pack; revisit on longer-document data.
UPDATE (same night): the CSA harness now EXISTS — sdsl-lite + libdivsufsort
vendored as _csa_native (csa_wt<wt_huff> and csa_sada size_in_bytes + SA +
BWT exposed; SA oracle-verified). 1MB repetitive: 0.913 B/char vs 100KB
random: 1.953 B/char — the compressed size does discriminate. The remaining
step is a proper feature bench (csa_bytes_rate) on long docs.

## Feature packs (fleet candidates)

1. **Comma-context / situational bigrams** (Jin–Zaitsu lineage via
   Ghatpande et al. 2026 review, docs/ghatpande2026-lr-fusion.md).
   English pack: comma-following-word-class profile, Oxford-comma rate,
   comma-splice tendency, quote/period-adjacency patterns. Cheap, single-pass.
2. ~~Repetitiveness measures~~ **BENCHED (fleet_condensates.py, 2026-08-16):
   cond30 alone 0.971 AUROC / 0.755 TPR@1e-2 dev C; ms_contrast_mean is the
   strongest single contrast feature ever benched here (0.912/0.591).
   Panel increment = TIE (info already present via coverage channel) -> not
   wired per RULE 6; kept as the fast-lane pack. ms_ uses cross-bucket refs
   (A vs B+C grams) after the expected in-fold crater taught us nothing new.
   Diet attempt (cut 8 weak-solo features -> cond22): tail DROPPED to 0.705
   (AUROC flat) - the 'dilution' hypothesis fails measurement; the weak
   solos earn their keep in interaction. Full 30 restored and re-verified.**
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
7. ~~RAID leaderboard adapter~~ **READY (2026-08-17):** submissions/
   snidewider/predictions.json (56K rows, official run_detection interface)
   + metadata.json drafted. Pre-flight all green: bundle==ladder exact
   (0.9905/0.962/0.874/0.618); official run_evaluation on our holdout =
   acc@5%FPR 0.9585 (weakest: reviews 0.880, poetry 0.897); 113 duplicate-
   text overlaps with test found+documented (poetry-heavy dups, 0.2%).
   REMAINING STEP (Donk's call): fork liamdugan/raid, drop the two files,
   PR.

## Watch list

- Ukkonen: quiet since 2020 (motif discovery, bioinformatics).
- Puglisi: very active — matching statistics, repetitiveness, succinct
  rank dictionaries, spectral BWT (with Bille/Gørtz at DTU and Bannai's
  Kyushu group). Watch for anything crossing into text statistics.
- Salmela: Sama assembler w/ correctness guarantees (2025). Mäkinen:
  compressed indexing/pangenomics. Kärkkäinen: quiet.

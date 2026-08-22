# Snidewider

*A library of text feature detectors for use in detecting, characterizing, and grouping AI text. A sample HBG model is provided.*

Detecting AI-generated text **at scale** and **reliably** is best done with deep learners, especially attentional transformers. However, much of the structure of humans text is only visible at the character level. Token-based learning captures much of this through second-order effects, but this is expensive, slow, and incomplete. The ~260 feature detectors and summary statistics in Snidewinder are designed to augment and annotate text to provide visibility into this deep structure. 

Even a simple histogram-boosted gradient learner built on this toolkit achieves exceptional performance on RAID, with an aggregate of ~90%. This places it above many deep learners and is the best performance a deterministic statistical approach has ever achieved. 

Snidewinder also provides reference implementations for a number of unusual but extremely fast text processing algorithms used as primitives when building these detectors.

## The toolkit (native-speed, tested, benchmarked)

| Primitive | What it is | Use it for |
|---|---|---|
| **CK2** | Linear-time Levenshtein-approx score (0=identical, 1=max different) | Pairwise near-duplicate/paraphrase proximity |
| **q-gram** | Ukkonen distance/similarity over length-q profiles | Multiset-overlap signal, edit-distance lower bound |
| **bag** | Bartolini bag distance (q=1 case) | Cheapest character-multiset signal |
| **Hanada search** | Average-case linear substring search by q-gram distance | Find all substrings of a corpus within distance k of a pattern |

```
aidt ck2 kitten sitting           # 0.285714 (0 = identical)
aidt qgram "the quick fox" "the quick foxes" -q 3
aidt bag aab abb
aidt search "suspicious phrase" corpus.txt -q 5
```

## Principles

0. **Bone stupid, computationally affordable.** Dense brute-force comparison is
   normally "intractable" only because everyone else's primitives are thousands
   of times slower. Ours are native. Stupid signals have no clever assumptions
   for an attacker to exploit — stupid is harder to fool.
1. **Reliability = calibrated error rates.** We report TPR at *fixed* FPR
   (default 1e-3) with confidence intervals. "97% accurate" on a balanced
   benchmark is a confession, not a result.
2. **Scale = streaming + cheap features first.** Parquet in, shards out, linear
   models before neural ones. If the baseline ties the fancy model, the fancy
   model gets composted.
3. **No leakage, ever.** Splits are by source document, never by chunk.
   Paraphrase/robustness evals are held out from threshold tuning.

## Layout

```
data/            raw (immutable) + derived artifacts — gitignored, see RULES.md
docs/            knowledge corpus: distilled notes on every reference + findings
papers/          the paper(s) driving this
scripts/         extract_pdf, build_native, profile_hanada, bench_* (separate)
src/ai_text_detection/   the package: ck2.py, qgram.py, __main__.py (aidt CLI),
                         native/ (vendored C++), built extensions
tests/           pytest
RULES.md         the pinned rules card — the floor
PERF-RULES.md    the pinned performance card — measure, never guess
```

## Quickstart

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
python scripts/build_native.py   # native CK2 + q-gram extensions (needs MSVC; CK2 has a pure-Python fallback)
pytest
```

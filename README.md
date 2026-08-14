# ai-text-detection

Detecting AI-generated text **at scale** and **reliably** — which mostly means:
resisting the urge to throw a transformer at everything before the fast baselines
have had their say, and reporting numbers that survive contact with adversarial
paraphrasing.

## The Paper

> TODO(Donk): drop the promising new paper in here (link/PDF in `papers/`), then
> we extract its claims into a repro plan. Everything in this repo is downstream
> of whatever it actually says — no vibes-based detection.

## Principles

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
docs/            knowledge corpus: distilled notes on every reference
papers/          the paper(s) driving this
scripts/         extract_pdf, build_native (native exts), bench_ck2, bench_qgram
src/ai_text_detection/   the package (+ native/ vendored CK2 headers)
tests/           pytest
RULES.md         the pinned rules card — the floor
```

## Quickstart

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
python scripts/build_native.py   # native CK2 + q-gram extensions (needs MSVC; CK2 has a pure-Python fallback)
pytest
```

# Fraser et al. 2024 — AIGT detection survey (distilled)

**Source:** "Detecting AI-Generated Text: Factors Influencing Detectability with
Current Methods", Fraser, Dawkins, Kiritchenko (NRC Canada), arXiv:2406.15583v1,
45 pp. Donk suspects it's AI-written; it's citation-dense and specific either way —
per RULES #7, treat every claim as a pointer to a primary source before building on
it. What follows is what changes *our* decisions.

## The perplexity family (why we're "reviewing the paper on perplexity")

LLMs pick high-probability continuations; humans don't. So measured under the
generating (or a proxy) model, AIGT has **low perplexity / low entropy / high avg
token probability / low avg rank**. Single-feature threshold classifiers:

- GLTR (visual ranks), log-prob/rank thresholds (Su et al.), DetectGPT (AIGT sits in
  negative-curvature regions: perturbations *lower* the probability), NPR (perturbed
  rank), Fast-DetectGPT (cheaper + better), variance of token probabilities
  (Venkatraman: LMs transmit info more uniformly than humans).
- Black-box workarounds: regeneration/compare (DNA-GPT), intrinsic dimensionality
  (~8 for AIGT vs 9–10 human), **proxy-model ensembles** — Ghostbuster (weak LMs →
  features → logistic regression), Sniffer (contrastive probs between model pairs),
  SeqXGPT, Binoculars (cross-entropy between two open models). Proxy ensembles are
  the practical answer to the unknown-model scenario.
- Survey's own caveat: "zero-shot" is a lie — any threshold needs calibration data,
  and thresholds should be domain/model specific. Also explicitly endorses **TPR at
  fixed FPR** reporting (Krishna et al. used 1% FPR for the DIPPER eval). The floor
  agrees.

## Detectability factors (Section 5, the most useful part)

- **Model size**: detection accuracy falls ~linearly as params grow exponentially
  (power law; Pagnoni). Newer/bigger = harder.
- **Decoding**: nucleus sampling hardest to detect; train-on-nucleus generalizes
  best; parameter shifts (k=40→160) can halve recall.
- **Length**: statistical methods need ~120–200 words; ~500 provably sufficient
  (Chakraborty); concatenating same-author shorts works (80%→~100% on 10 tweets);
  **balance class length distributions or the detector learns "long = AI"**.
- **Domain/prompt shift**: 10–25% accuracy drops across domains; even similar
  prompts don't reliably transfer; ELECTRA-style detectors > RoBERTa OOD.
- **Fairness**: TOEFL essays by Chinese English learners hit ~60% FPR on popular
  detectors — non-native formulaic writing reads as "unsurprising". Bias mitigation
  is a first-class requirement, not a footnote.
- **Human influence**: polishing/paraphrase of human text is the hardest category
  (detectors < random on GPT-polished abstracts); mixcase unseen in training =
  failure; typos break fine-tuned detectors; AI paraphrase breaks threshold ones.
- **Adversarial**: DIPPER paraphrase → statistical detectors <5% TPR @ 1% FPR,
  fine-tuned 13%, watermark 50%; iterative paraphrase evens out after 5–6 rounds.
  Paraphrase robustness must be *evaluated*, not assumed (RULES: paraphrase sets
  stay out of tuning).

## Recommendations we adopt

Ensemble of diverse, individually-calibrated detectors (narrow statistical signals
at fixed low FPR + general classifier for unknown cases); train/generate data with
varied lengths, multiple prompts, many human authors incl. L2 writers, nucleus
sampling with varied params; report calibrated uncertainty; expect ongoing data
collection as generators move. Humans detect at ~chance (59% for experts) — the
machine's edge is exactly the statistical signals we're building.

## Taxonomy/scenarios (shared vocabulary)

AIGT types: arbitrary → guided → controlled (paraphrase/translate/polish) →
collaborative (post-edited, mixcase) — roughly increasing detection difficulty.
Scenarios: known/unknown model × white/black-box access. Our tool targets
**unknown-model, black-box** with **arbitrary/guided** text as the sweet spot and
controlled/collaborative as the honest hard cases.

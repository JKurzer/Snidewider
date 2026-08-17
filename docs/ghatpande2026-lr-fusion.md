# Review — Ghatpande, Tsuge, Ishihara, Zaitsu (2026), arXiv:2606.13991

"Fusing Stylometric and Embedding Systems to Estimate Authorship Likelihood
Ratios in Japanese." Authorship verification (same/different author) on
Yahoo! Japan blogs (>=1000 chars), forensic LR framing. Reviewed 2026-08-16
for Snidewider feature/method theft.

## Method in one breath

5 stylometric systems + 2 embedding systems (word/char Japanese RoBERTa) ->
per-system scores -> logistic calibration -> log-likelihood ratios ->
Cllr / EER eval with Tippett plots -> greedy fusion over system combos.

- Stylometric features (their Table 3): char bigrams, function-word
  unigrams, POS bigrams, **comma-context bigrams** (the Jin/Zaitsu lineage:
  bigrams of the comma + adjacent character), character types.
- Count-based features scored with a Dirichlet-multinomial model;
  embeddings with cosine distance.
- Best fusion: char-bigrams + comma-bigrams + char-types + both embeddings
  (Cllr 0.325). Fusion of UNRELATED feature views beat fusing the best
  single performers — "different dimensions of authorship, not necessarily
  the top single-feature performers." (Convergent with our distill-pack
  experience: 33 weak-but-orthogonal features >> strong correlated ones.)

## Steal-worthy for us

1. **Comma-context features (Jin/Zaitsu lineage).** Punctuation CONTEXT, not
   punctuation rates. English version: comma-following-word-class profile
   (and/but/because/the...), Oxford-comma usage rate, comma-splice tendency.
   Cheap, single-pass, authorially sticky — exactly our kind of feature.
2. **Cllr / log-LR calibration.** We emit raw probabilities; forensic
   practice emits calibrated log-likelihood ratios (logistic calibration on
   a dev bucket, Cllr as the loss). For any deployment where a false
   accusation has consequences, LR framing + Cllr is the honest output
   format. Candidate eval upgrade alongside TPR@FPR.
3. **Char-type histograms as a system.** Their System 5 = our charstat
   pack, independently invented. Convergent validation.

## Not for us

- Dirichlet-multinomial scoring: their features are count vectors on small
  vocabularies; ours are mixed continuous stats. HGB/L0 cover it.
- The embedding half needs a Japanese RoBERTa; language-specific, off-scope.

## Verdict

Solid forensic-grade fusion study. The comma-context family gets a fleet
slot; Cllr calibration goes on the eval-debt list. Ishihara's shop remains
the real deal.

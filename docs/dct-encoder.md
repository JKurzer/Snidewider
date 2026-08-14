# DCT as Universal Sentence Encoder

**Source:** Almarwani & Diab, ACL-IJCNLP 2021 (short), `2021.acl-short.53.pdf`.
Builds on Almarwani et al. 2019 (EMNLP) and Kayal & Tsatsaronis 2019 (EigenSent).

## What it is

Training-free sentence embedding from any word vectors (FastText, mBERT last layer).

1. Sentence = N×d matrix (N tokens × embedding dim).
2. Apply **DCT-II along the sentence-length axis**, per embedding dimension:
   `c[K] = sqrt(2/N) Σ_{n<N} v_n · cos(π/N · (n + ½) · K)`
3. Concatenate the first K coefficients per dimension → fixed-length **K·d** vector.

Intuitions:

- **c[0] ∝ the mean vector** — averaging is the K=1 special case; DCT strictly
  generalizes it.
- Higher K = higher-frequency components = **word order / structural dynamics**
  (what averaging destroys). Invertible: coefficients can reconstruct the signal.
- Zero parameters, language-agnostic, trivially batched — a signal-processing
  compression trick applied to token sequences.

## Findings that matter to us

- Probing tasks (Conneau-style, 5 languages): surface tasks (SentLen, word content)
  peak at c[0]; **syntactic tasks (TreeDepth, CoordInv, Tense, SubjNum, ObjNum) need
  K≥2 and keep improving to K≈3–4**; semantic (SOMO) ~flat.
- Cross-lingual sentence retrieval: DCT > AVG everywhere; mBERT c[0:1] hit 91.83%
  avg (vs ELMo 84.03% prior SOTA). c[0:2]/c[0:3] best on pretrained FastText.
- Russian (morphologically rich, free word order) benefits most from more
  coefficients — structural info matters more when order carries less syntax.

## Role in the detector

The **sentence-level** signal, used intermittently (Donk's word): char-level methods
(CK2/q-gram) can't see that two paraphrases "say the same thing differently";
averaged embeddings can't see *how* the sentence was assembled. DCT sits between:
structure-sensitive semantics at ~zero cost. Candidate features: DCT-vector cosine
similarity between sentences/paragraphs (near-duplicate semantic overlap robust to
surface paraphrase), and distributional stats of sentence vectors as stylometry.

Cost note: needs word embeddings loaded (FastText ~ GBs); "intermittent" use =
encode when char-level scores land in the uncertain band, not on every doc.

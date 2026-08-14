# q-gram distance/similarity

**Sources:** Ukkonen 1992, *"Approximate string-matching with q-grams and maximal
matches"*, TCS 92(1):191–211 (PII 0304-3975(92)90143-4; `QGrams.pdf` is a scan,
`0304397592901434.htm` the readable copy). Hanada, Kudo, Nakamura 2014,
*"Average-case linear-time similar substring searching by the q-gram distance"*,
TCS 530:23–41 (`1-s2.0-S0304397514001285-main.pdf`). Background:
Bernardini et al. 2019/2020, *"Approximate pattern matching on
elastic-degenerate text"*, TCS 812:109–122.

## Core concepts (Ukkonen 1992)

- A **q-gram** is any substring of length q. String of length n has n−q+1 of them.
- **Profile**: vector of per-q-gram occurrence counts.
- **q-gram distance** d_q(x,y) = Σ_g |p_x(g) − p_y(g)| — L1 between profiles. O(|x|+|y|)
  with a counting array indexed by rolling q-gram codes.
- Key relation: d_q is a **lower bound on edit distance**: d_q(x,y) ≤ 2q·d_edit(x,y).
  Hence its use as a cheap *filter* before expensive exact distances.
- Also defines q-gram based approximate string matching: text positions whose q-gram
  count overlap with the pattern exceeds a threshold are candidate matches
  ("counting filter"), plus maximal-match techniques.

## Hanada et al. 2014 — the FAST part

Problem: find *all substrings* of a long text t within d_q ≤ k of pattern p.
Ukkonen's algorithms cost O(|t|k+|p|) (array) and O(|t|log k+|p|) (tree).
Hanada et al. give two algorithms with **average-case O(|t|+|p|)** (worst case
unchanged), exploiting that similar substrings cluster at close positions; second
variant uses a doubly-linked list + array + search tree for O(log k) list search.
Linearity holds under random-text assumptions when q exceeds a threshold.

This is why "FAST q-grams is not short": pairwise profiles are easy; *scanning a
corpus* for q-gram-similar substrings in linear average time is real algorithmic work.

## Bag distance (companion measure)

Bartolini, Ciaccia, Patella 2002 (SPIRE, "String matching with metric trees using an
approximate distance"): d_bag(x,y) = max(|m(x)∖m(y)|, |m(y)∖m(x)|) over character
**multisets** (equivalently the q=1 case). Lower bound for edit distance, cheaper and
weaker than q-grams; used as an indexing/pruning distance. No dedicated paper in the
folder — definition from the literature.

## Implementation landscape — searched, gap confirmed

- PyPI: no q-gram package at any plausible name (checked 7).
- GitHub: Java (`java-string-similarity`, strsimpy's parent) and Go ports; strsimpy is
  pure-Python dict-L1 (reference semantics, space-stripped shingles, k=3 default).
  Milanese's `Threshold_q-gram_distance` (C++) is a bioinformatics *thresholded
  count-difference* variant over DNA/RNA/AA alphabets — wrong semantics for us.
- No public implementation of Hanada's algorithms found.
- **Conclusion: no installable native q-gram library with our semantics exists.**
  Options are with Donk (faithful port bound like CK2 vs numpy-primitive assembly vs
  other) — see repo discussion; decision pending before detector wiring.

# CK2 / CKS — linear-time Levenshtein approximation

**Sources:** `ck_similarity_sparse.hpp`, `ck_similarity_sparse8.hpp`, `Readme.md`
(Donk's reference folder). Novel implementation of the lapsed CKS patent; described
as "probably the defacto open implementation."

## What it computes

For strings A, B (canonicalized so X = shorter, or lexicographically smaller on ties;
Y = the other, n = |Y|):

1. Index each side as byte-value → sorted position list ("CKHash").
2. Per shared byte value: greedy nearest-neighbour matching between the two position
   lists. Accumulates **D** = sum of position gaps of matched occurrences (order/
   displacement cost) and leaves **Sa/Sb** = counts of unmatched occurrences per side
   (leftovers after matching; also covers byte values present on only one side).
3. **S** = max(Sa, Sb) (substitution-like cost), then

   `score = 1 - ((n² - D)/n²) · ((n - S)/n)`

**Semantics: 0.0 = identical, 1.0 = maximally different.** It's a normalized
distance despite the historical "similarity" name (worked example: `CK2("ab","ba") =
0.5` — order flipped, same chars).

## Why it's fast

O(m+n): counting-sort-style hash construction + per-occurrence constant work, no
quadratic DP matrix. Two variants:

- `ck::sparse` — uint32 indices, arena-backed (16 KB inline buffer, then heap). General.
- `ck_similarity_sparse8` — uint8 indices for len < 255, Briggs & Torczon sparse-set
  trick (uninitialized sparse array validated against a dense present list: O(1)
  lookups with zero construction cost), `alloca`'d buffers, auto-falls back to
  `ck::sparse` past the boundary. Entry point for bindings.

## Measured (this machine, via pybind11 bindings)

| length | CK2 | rapidfuzz Levenshtein | ratio |
|---|---|---|---|
| 50 B | 1.12 µs | 0.32 µs | rapidfuzz wins (call overhead dominates) |
| 2 KB | 18.9 µs | 208 µs | ~11x |
| 20 KB | 128 µs | 20.1 ms | ~156x |

Asymptotics dominate from ~1 KB up; the Readme's "300x for long strings" is plausible
at 100 KB+. If short-pair throughput ever matters, a batch API would kill the per-call
overhead (YAGNI for now).

## Caveats for detection use

- **Byte-oriented** — we feed UTF-8; encoding policy lives in Python.
- It's an *approximation* of Lev/LevDam, not the distance itself; no published error
  bounds in this folder (the `ck::corrected` derivation file the headers reference is
  NOT in the folder — flag: ask Donk if it exists).
- Symmetric by construction (canonical X/Y ordering). Not established as a metric
  (triangle inequality unknown) — treat as a score, not a distance, until proven.

## Bindings

`src/ai_text_detection/native/` (headers vendored + `ck2_module.cpp` shim),
`src/ai_text_detection/ck2.py` (native fast path + pure-Python oracle port),
built in-place by `scripts/build_ck2.py` (MSVC, C++20 for `std::span`).
Tests: `tests/test_ck2.py` — invariants + port/native agreement, incl. the 255 B
fallback boundary.

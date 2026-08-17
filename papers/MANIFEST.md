# papers/ MANIFEST — what each warehoused paper is and whether we've used it

Legend: **USED** = code/theory dependency of the stack · **DISTILLED** = has a
docs/ knowledge note · **CANDIDATE** = plausibly relevant, not yet mined ·
**REFERENCE** = context/infrastructure, no feature path seen · **SHELF** = fun
but not relevant.

## USED

| file | paper | powers |
|---|---|---|
| hanada2014-qgram-search.pdf | Hanada 2014, q-gram substring search | `qgram.search`/`search_ukkonen` (native fast path) |

Used but not warehoused here: Ukkonen 1992 (q-gram distance — `qgram.py`),
CKS/CK2 (the OneDrive reference corpus — `ck2.py`), Dugan et al. 2024 RAID +
Toloka 2025 Beemo (datasets — `docs/data.md`), Almarwani & Diab 2021 DCT
sentence encoding (`dct.py`, `docs/dct-encoder.md`), hazimehh/L0Learn
(`native/l0learn/`, vendored source).

## DISTILLED

| file | paper | note |
|---|---|---|
| ghatpande2026-stylometric-embedding-lr-japanese.pdf | Ghatpande/Tsuge/Ishihara/Zaitsu 2026 | docs/ghatpande2026-lr-fusion.md |
| liptak-masillo-puglisi2026-matching-statistics-survey.pdf | Lipták/Masillo/Puglisi 2026 (TCS) | docs/condensates.md |
| puglisi2026-repetitiveness-reversal.pdf | Bannai-circle + Puglisi 2026 (CPM) | docs/condensates.md |
| alanko2025-spectral-bwt-kmer-lookup.pdf | Alanko/Biagi/Mackenzie/Puglisi 2025 (ALENEX) | docs/condensates.md |

## CANDIDATE

| file | why it might matter |
|---|---|
| badkobeh2022-maximal-closed-substrings.pdf (+ online-2026, rle-enumeration-2026) | closed-substring census = repeat-structure feature family (low priority) |
| alanko2025-finimizers.pdf | bounded-frequency k-mer subsampling — representative reference construction for coverage/MS refs |

## REFERENCE

| file | content |
|---|---|
| alanko2025-graph-indexing-beyond-wheeler.pdf | graph indexing theory (post-Wheeler) |
| alanko2025-compact-structures-collections-sets.pdf | sets-of-sets compact indexes |
| makinen2026-adaptive-compressed-suffix-arrays.pdf | small/fast compressed SA (infra) |
| donges-puglisi2025-succinct-rank-dictionaries.pdf | rank dicts (infra) |
| makinen2025-elastic-degenerate-strings-construction.pdf | pangenome strings |
| rizzo-makinen2022-indexable-elastic-founder-graphs.pdf | founder graph indexing |
| makinen2025-practical-colinear-chaining.pdf | alignment chaining (genomics) |

## SHELF

| file | content |
|---|---|
| makinen2026-quantum-pattern-matching-degenerate.pdf | quantum pattern matching — because why not |

Missing: Diseth/Heljanko/Puglisi "Massively Parallel Computation of Matching
Statistics" (SPIRE 2025) — embargoed at last check.

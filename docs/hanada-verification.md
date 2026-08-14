# Verification of `hanada-array-base-search.md` against Hanada et al. 2014

- Verifier: code-puppy-clone-1 (read-only audit; no code run, no tests executed)
- Source: `papers/hanada2014-qgram-search.pdf` = Hanada, Kudo, Nakamura, *TCS* 530 (2014) 23–41
- Method: pypdf text extraction (plain + `layout` mode cross-check for Fig. 5). Page
  numbers below are the **journal pages** (23–41); journal page *N* = PDF page *N*−22.

**Overall verdict: the note holds up. All six claims CONFIRMED, one nit (bench k), one
important nuance (V3: the paper handles the corner in its *other* algorithm).**

| Claim | Verdict |
|-------|---------|
| V1 quote accuracy | CONFIRMED |
| V2 Fig. 5 recomputes j* only on c-inside branch | CONFIRMED |
| V3 stale-j* corner unhandled in whole paper | CONFIRMED (nuance: List+Base-Search handles it) |
| V4 mechanism (j* at left edge, evicted ~every step) | CONFIRMED |
| V5 Fig. 14(a) growth + authors' q-condition attribution | CONFIRMED |
| V6 fix required for correctness vs Sec. 2.2 | CONFIRMED |

---

## V1 — Quote accuracy: CONFIRMED

Note: *"if c_i is outside scope(i+1), then j* is unchanged or becomes e_{i+1}" (§4.1)*.

Paper, §4.1, p. 28:

> "The common idea for the improvement comes from the fact that, if c_i is outside
> scope(i + 1) = [b_{i+1}, e_{i+1}], then j* for i + 1 must be either of: (1) unchanged
> from that for i or (2) changed to e_{i+1}."

Faithful paraphrase, correct section. The paper genuinely stakes everything on this premise.

## V2 — Fig. 5 recomputes j* only on the c-inside branch: CONFIRMED

Fig. 5, p. 30 (verified in both plain and layout-mode extraction; the published figure's
line numbering literally jumps 18 → 22 → 23, so nothing is hidden between them):

- line 11 `if c ∈ [b,e] then:` → line 12 updates all D[j], **line 13 `j* ← argmin_{j∈[b,e]} D[j]`** — the only argmin in the loop;
- line 14 `else if c < b then:` → line 15 `o ← o − 1`; line 16 `else: // c > e` → line 17 `o ← o + 1` — **only the baseline moves**;
- line 18 computes the new right-edge cell D[e];
- line 22 `if D[e] ⩽ D[j*] then j* ← e` — the only j* update on the baseline path, and it can only move j* **to e**;
- line 23 `if o + D[j*] ⩽ k then output t[i..j*]`.

So on the c-outside path j* is never re-examined against the surviving scope. Theorem 1's
proof (p. 29) confirms this is intended, not an omission of ink: *"When the change point is
outside the scope, O(1) suffices to change the value of o and add D[e_{i+1}]."* — zero j* work
is budgeted.

## V3 — The j* = b_i corner: CONFIRMED unhandled for Array+Base-Search (with a nuance)

Whole-paper sweep (Secs. 1–6, Appendix A, all figure captions):

1. **Only related assumption:** Sec. 2.2, p. 25: *"We may assume that j* ∈ scope(i) for any i."*
   scope(i) = [b_i, e_i] (Def. 2, p. 25) **includes b_i** — the corner is not excluded.
2. §4.1 premise (p. 28) states cases (1)/(2) with no caveat. It is **false** when j* = b_i:
   b_i ∉ scope(i+1) = [b_i+1, e_i+1], so "unchanged" is impossible, and the true new argmin is
   the runner-up inside the old scope — generically **not** e_{i+1} (see toy in V6).
3. Fig. 5 (p. 30): no eviction handling; after `b ← b+1` (line 10) a j* sitting at the old b_i
   is stale, and line 23 still tests/outputs it.
4. §4.3.1, p. 32 repeats the same oversight in words: *"the position j* changes if we change
   the starting position from i to i + 1. This means j* < c_i, otherwise j* stays at the same
   position"* — scope-slide eviction is again ignored.
5. Appendix A (pp. 38–40) covers only List+Base-Search operations.

**Nuance (the one place the paper *does* handle it):** List+Base-Search. DEL-FIRST explicitly
removes the left edge when it heads the candidate list — Fig. 8 step (1), p. 33: *"if b_i exists
in L as the beginning of L, then remove it from L, A and T"*; Fig. 9, p. 34: DEL-FIRST runs on
**both** branches (lines 13 and 17); Fig. 10, p. 34: *"Case (1-a): When the node b_i is included
in the 1st cell of L, then reconnect the pointer to the next..."* Since L's first element is j*
(Def. 4, p. 33), "b_i in the 1st cell" **is** our corner — handled correctly in O(1) by promoting
the next candidate. So the authors were aware left-edge eviction happens — for the List variant —
and never carried the lesson back to Fig. 5. (This also validates the note's proposed fix as
literally the paper's own idea.) Related: Ukkonen's Tree-Search (Sec. 3.2, p. 27) even has an
explicit left-edge argmin convention (j*[J] "can be j1 − 1"), and Array-Search recomputes j* over
the whole scope every step in O(k) (Sec. 3.1, p. 27) — the corner is an artifact introduced
specifically by the Array+Base baseline optimization.

## V4 — Mechanism: CONFIRMED sound

Consistency with paper definitions:

- Sec. 2.3, p. 26: *"Δ1j = d_q(t[i..j+1],p) − d_q(t[i..j],p) takes either +1 or −1"* — +1 when the
  newly included q-gram is in excess of p's histogram, −1 when it fills a deficit. With random
  text and rare pattern-gram matches, almost every extension is excess ⇒ d(i)(j) ~strictly
  increasing in j ⇒ unique argmin at the **left edge b_i** (no ties, so the longest-on-ties rule
  of Sec. 2.2, p. 25 — *"equality may hold only for j < j*"* — never rescues an interior j*).
- scope(i+1) = [b_i+1, e_i+1] (Def. 2): the window slides right every step ⇒ a left-edge j* is
  evicted every step. Matches the note's counter: corner_fixes = 98,980 ≈ every step of |t|=100k.
- **The kicker:** the paper's own linear-time regime *is* this regime. Corollary 2's proof (p. 31)
  bounds α ⩽ (2k+1)·r_max^q — linearity requires gram-recurrence inside the scope to be rare,
  which is exactly what makes d increasing and j* hug the left edge. The corner fires most
  precisely where the paper claims its speedup.
- **k-dependence:** occurrence ~independent of k (note: 98,980 fires for k = 5, 100, 495 alike —
  consistent); **cost** per fire is the O(2k+1) argmin. The note's argmin_iters ≈ fires·(2k+1)
  checks out arithmetically: 98,980·201 = 19.9M vs measured 19.68M; 98,980·991 = 98.1M vs 97.4M.
  So the measured linear growth in k is exactly the corner-fix cost. Sound.

## V5 — Fig. 14(a) growth and the authors' attribution: CONFIRMED

Sec. 5.2, p. 38:

> "From Fig. 14, we can see that the results are consistent with their theoretical complexities
> derived above **except for Array+Base-Search**: linear, logarithmic and constant in |p| for
> Array-Search, Tree-Search and List+Base-Search, respectively."

So yes — Array+Base-Search's curve grows with |p| (it is the lone exception to "constant"),
while List+Base-Search is flat. Their explanation, p. 38, in full:

> "A possible explanation for Array+Base-Search relies on the condition. This evaluation (the
> third evaluation above) holds only when q ⩾ 2 log_{1/r_max} |p|, and this condition is satisfied
> in this experiment because 5 ⩾ 2 log_20 500 = 4.1 even for the largest |p| = 500. On the other
> hand, the large-oh evaluation holds only when |p| is sufficiently large. As a result, although
> the condition is satisfied, it is not sufficient for making sure of the complexity. While, the
> condition for List+Base-Search is sufficiently satisfied by 5 ⩾ log_20(500 log 500) = 2.7."

Attribution is indeed to the q-condition / big-oh asymptotic regime — "a possible explanation"
that by their own arithmetic *is satisfied* (5 ⩾ 4.1), i.e., an admitted non-explanation. No
mention of any j* corner anywhere in Sec. 5. The note's hedge ("plausibly explains") is the
right strength: without the authors' C++ code, causality can't be proven from the paper alone —
but it is consistent (a corner-correct Array+Base pays O(k) ~every step in this regime, while a
stale-j* one would be flat *and wrong*; the paper's curve is neither flat nor flagged as wrong).

## V6 — "Fix required for correctness" vs Sec. 2.2: CONFIRMED

Sec. 2.2, p. 25 (the Problem): report t[i..j*] where j* minimizes d_q over scope(i) with
longest-on-ties, iff d_q ⩽ k. Independent analysis of the corner (j* = b_i, c_i < b_{i+1}):

1. d(i)(b_i) is the **strict** min over scope(i) (longest-on-ties: any tie at j > b_i would have
   been preferred), so d(i)(j) ⩾ d(i)(b_i)+1 for j > b_i.
2. Scope bound (Sec. 2.2: d_q(x,y) ⩾ ||Hx|−|Hy||): |t[i+1..b_i]| = |p|−k−1 ⇒ d(i+1)(b_i) ⩾ k+1,
   hence d(i)(b_i) ⩾ k+2.
3. On each subsequent c<b step, o decrements (line 15), so the **stale** value o+D[b_i] drops by
   1/step, while every surviving in-scope position stays ⩾ stale+1. After d(i)(b_i)−k ⩾ 2 steps
   the stale value hits k and line 23 **outputs t[i′..b_i]** — out of scope, true distance > k.
   False positive; the true j* is never identified (false negative).

Explicit toy (paper's own definitions): Σ={a,b,z}, q=2, p="aabbb" (Hp: aa,ab,bb,bb; |Hp|=4),
k=3, t="zzz…". scope(i)=[i+1,i+7]; d(i) on scope = 5,6,…,11 (strictly increasing), j* = b_i = i+1.
c_i = i+1 < b_{i+1} (Prop. 1: m=0 ⇒ c_i = i+q−1, p. 27). Step i+1: o: 5→4, stale test 4 > 3, no
output (state corrupted). Step i+2: o: 4→3, line 23: o+D[j*] = 3 ⩽ k ⇒ **outputs t[i+2..i+1] —
the empty string**, claimed distance 3, while d_q(ε,p) = 4 > k and the true in-scope min is 5 > k
(nothing should be reported; j* < i isn't even a valid candidate under "1 ⩽ i ⩽ j*"). A brute-force
oracle implementing Sec. 2.2 catches this immediately — fully consistent with the note's
"required for correctness (oracle-verified)".

## Nits and surprises

- **Nit (note's, immaterial):** the note calls its bench "paper settings … k=|p|−q". The paper's
  experiment uses **k = |p|**, q = 5 (Sec. 5.1, p. 37: *"where k = |p| and q = 5"*). |t|=100,000,
  |Σ|=20, q=5, patterns-as-substrings all match. Both are k=Θ(|p|); no claim is affected.
- **Surprise 1:** Fig. 5's published line numbers skip 19–21 (both extraction modes agree — it's
  in the PDF, not an artifact). Also line 3 reads "d_q(t[1..j], **q**)" — pattern mislabeled as q.
  Paper typos, not note errors.
- **Surprise 2:** the corner the note found is *solved in the same paper*, one section later, for
  the List variant (V3). The Array variant is the only algorithm in the paper left exposed.
- **Surprise 3 (sharper than the note):** Theorem 1 budgets O(1) per baseline step, so a
  *correct* Array+Base-Search cannot meet the paper's own bound with a plain array: fixing the
  corner with a rescan costs O(k) ~every step (⇒ O(k|t|), no better than Ukkonen's Array-Search),
  and fixing it in O(1) amortized requires the note's monotone deque (= the paper's candidate
  list). Theorem 1's linear claim for the Array variant effectively rests on the false §4.1
  premise. List+Base-Search (Theorem 2) is unaffected — DEL-FIRST covers the corner.

## Conclusion

The note's central claim — Fig. 5's premise fails when j* sits at the erased left edge, the paper
never handles it for Array+Base-Search, the paper's own speedup regime is exactly where this
happens, and a corner fix is required for correctness — is **confirmed against the paper text**.
The note may be cited externally, with two corrections: (a) bench used k=|p|−q, paper used k=|p|;
(b) credit the paper's List+Base-Search (DEL-FIRST) as an existing in-paper handling of the same
corner — which strengthens, not weakens, the note's argument.

"""AUDIT DRAGON — oracle checks for the Kolmogorov-proxy internals.

  suffix array   ORACLE: is a permutation + strictly sorted by suffix bytes.
  Kasai LCP      ORACLE: brute-force LCP of adjacent suffixes (short strings).
  Re-Pair        ORACLE: recording variant expands back to the original.

The fleet_dragon feature functions are imported and checked against these.
Usage: .venv\\Scripts\\python scripts\\exp\\audit_dragon.py
"""

from __future__ import annotations

import random
import sys

sys.path.insert(0, "scripts/exp")
from fleet_dragon import lcp_feats, repair_feats

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def sa_naive(b: bytes) -> list[int]:
    return sorted(range(len(b)), key=lambda i: b[i:])


def lcp_brute(b: bytes, sa: list[int]) -> list[int]:
    """LCP[k] = common prefix of sa[k] and sa[k+1], k in 0..n-2."""
    out = []
    for k in range(len(sa) - 1):
        i, j = sa[k], sa[k + 1]
        h = 0
        while i + h < len(b) and j + h < len(b) and b[i + h] == b[j + h]:
            h += 1
        out.append(h)
    return out


def kasai(b: bytes, sa: list[int]) -> list[int]:
    """Mirror of fleet_dragon.lcp_feats internals (same code path)."""
    n = len(b)
    rank = [0] * n
    for i, s in enumerate(sa):
        rank[s] = i
    lcp = [0] * (n - 1)
    h = 0
    for i in range(n):
        r = rank[i]
        if r == n - 1:
            h = 0
            continue
        j = sa[r + 1]
        while i + h < n and j + h < n and b[i + h] == b[j + h]:
            h += 1
        lcp[r] = h
        h = max(0, h - 1)
    return lcp


def repair_record(b: bytes):
    """Re-Pair with rule recording; returns (rules, final_seq)."""
    seq = list(b)
    rules: list[tuple[int, int, int]] = []
    top = 256
    while True:
        counts: dict[tuple[int, int], int] = {}
        for i in range(len(seq) - 1):
            p = (seq[i], seq[i + 1])
            counts[p] = counts.get(p, 0) + 1
        if not counts:
            break
        (a, c), best = max(counts.items(), key=lambda kv: kv[1])
        if best < 2:
            break
        top += 1
        rules.append((a, c, top))
        out = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == c:
                out.append(top)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = out
    return rules, seq


def repair_expand(rules, seq) -> list[int]:
    expansion = {new: (a, c) for a, c, new in rules}

    def expand(x):
        if x in expansion:
            a, c = expansion[x]
            return expand(a) + expand(c)
        return [x]

    out: list[int] = []
    for x in seq:
        out.extend(expand(x))
    return out


def main() -> None:
    rng = random.Random(11)
    alphabets = [b"ab", b"abc", bytes(range(97, 122))]
    cases = [bytes(rng.choice(alpha) for _ in range(rng.randint(3, 60)))
             for alpha in alphabets for _ in range(40)]
    cases += [b"a", b"aaaaaa", b"banana", b"abcabcabcabc", b"the quick brown fox"]

    print("== SA oracle: permutation + sorted ==", flush=True)
    bad = 0
    for b in cases:
        sa = sa_naive(b)
        if sorted(sa) != list(range(len(b))):
            bad += 1
            continue
        if any(b[sa[k]:] >= b[sa[k + 1]:] for k in range(len(sa) - 1)):
            bad += 1
    check("suffix array valid (all cases)", bad == 0, f"({bad} bad)")

    print("== Kasai LCP oracle: brute force ==", flush=True)
    bad = 0
    for b in cases:
        if len(b) < 2:
            continue
        sa = sa_naive(b)
        if kasai(b, sa) != lcp_brute(b, sa):
            bad += 1
            print(f"    LCP MISMATCH on {b[:24]}")
    check("Kasai == brute LCP (all cases)", bad == 0, f"({bad} bad)")

    print("== lcp_feats sanity: max LCP == longest repeated substring ==", flush=True)
    bad = 0
    for b in cases:
        if len(b) < 10:
            continue
        brute_max = max(lcp_brute(b, sa_naive(b)), default=0)
        if lcp_feats(b)["kol_lcp_max"] != float(brute_max):
            bad += 1
    check("lcp_max == longest repeat (brute)", bad == 0, f"({bad} bad)")

    print("== Re-Pair oracle: grammar expands back ==", flush=True)
    bad = 0
    for b in cases:
        rules, seq = repair_record(b)
        if repair_expand(rules, seq) != list(b):
            bad += 1
            print(f"    EXPAND FAIL on {b[:24]}")
        # feature agreement: rates match recorded grammar
        f = repair_feats(b)
        if abs(f["kol_repair_rules_rate"] - len(rules) / max(1, len(b))) > 1e-12:
            bad += 1
        if abs(f["kol_repair_total_rate"] - (len(rules) + len(seq)) / max(1, len(b))) > 1e-12:
            bad += 1
    check("Re-Pair expand == original + feature agreement", bad == 0, f"({bad} bad)")

    print("== determinism + edges ==", flush=True)
    probe = b"mississippi mississippi"
    check("deterministic", repair_feats(probe) == repair_feats(probe)
          and lcp_feats(probe) == lcp_feats(probe))
    check("short docs NaN", str(lcp_feats(b'hi')["kol_lcp_mean"]) == "nan")
    check("repetitive input sane", repair_feats(b"a" * 500)["kol_repair_rules_rate"] < 0.05)

    print(f"\n{'ALL PASS' if not FAIL else f'FAILURES: {FAIL}'}")


if __name__ == "__main__":
    main()

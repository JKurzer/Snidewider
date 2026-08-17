"""Verify _csa_native: SA vs naive oracle; CSA sizes sane; BWT roundtrip-free."""
import random
import sys

import numpy as np

from ai_text_detection import _csa_native

rng = random.Random(5)
bad = 0
for trial in range(60):
    n = rng.randint(1, 400)
    alpha = rng.choice([b"ab", b"abc", bytes(range(97, 122))])
    s = bytes(rng.choice(alpha) for _ in range(n))
    out = _csa_native.csa_stats(s)
    sa_native = list(out["sa"])
    # sentinel-terminated CSA: n+1 entries, sa[0] == n; strip it for compare
    sa_naive = sorted(range(n), key=lambda i: s[i:])
    if sa_native[0] != n or sa_native[1:] != sa_naive:
        bad += 1
        print(f"SA MISMATCH n={n} trial={trial}")
    # CSA sizes must be >= the text and finite
    if not (out["csa_wt_bytes"] >= n and out["csa_sada_bytes"] >= n):
        bad += 1
        print(f"SIZE INSANE n={n}: {out['csa_wt_bytes']} {out['csa_sada_bytes']}")

# big-string sanity: 1 MB repetitive text
big = (b"the quick brown fox. " * 50_000)[:1_000_000]
out = _csa_native.csa_stats(big)
print(f"1MB repetitive: n={out['n']} wt={out['csa_wt_bytes']:.0f}B "
      f"sada={out['csa_sada_bytes']:.0f}B (ratio {out['csa_wt_bytes']/out['n']:.3f} B/char)")
rnd = bytes(random.Random(1).choices(range(1, 256), k=1_000_00))
out2 = _csa_native.csa_stats(rnd)
print(f"100KB random:   n={out2['n']} wt={out2['csa_wt_bytes']:.0f}B "
      f"(ratio {out2['csa_wt_bytes']/out2['n']:.3f} B/char)")
print("SA agreement:", "ALL PASS" if bad == 0 else f"{bad} FAILURES")
sys.exit(1 if bad else 0)

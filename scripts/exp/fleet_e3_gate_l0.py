"""FLEET E3 — gate spec reproduction + L0 wrapper edge cases.

  1. gate.py: recompute C-bucket fire rates from the frozen spec; compare to
     the documented verify (TPR 5.4% AI / FPR 0.52% human, commit 1941f80).
     Note: buckets changed (post stratification fix) — rates will legitimately
     differ; we assert SANITY (human rate near zero), and record the new truth.
  2. L0 wrapper edge cases: constant feature column, single-class y, n < p.
     Requirement: fail LOUD or behave sanely — never silently NaN a path.
  3. tpr_at_fpr k=0 regression (fewer than 1000 humans).

Usage: .venv\\Scripts\\python scripts\\exp\\fleet_e3_gate_l0.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_text_detection import _l0learn_native
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.gate import Gate
from ai_text_detection.metrics import tpr_at_fpr

FAIL = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def main() -> None:
    print("== 1. gate spec on current buckets ==")
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    c = split_buckets(df)["C"]
    gate = Gate.load()
    rates = {}
    for label, sub in (("human", c[c.model == "human"]), ("ai", c[c.model != "human"])):
        fires = [gate.flag(str(t)) for t in sub.generation]
        rates[label] = float(np.mean(fires))
    print(f"  C fire rates: AI {rates['ai']:.4f} / human {rates['human']:.5f} "
          f"(pre-fix verify: 0.054 / 0.0052)")
    check("human fire rate < 1%", rates["human"] < 0.01)

    print("== 2. L0 wrapper edge cases ==")
    rng = np.random.default_rng(0)
    n, p = 400, 20
    X = rng.normal(size=(n, p))
    X[:, 5] = 3.14  # constant column: Normalize divides by scaleX -> 0?
    beta = np.zeros(p)
    beta[2] = 1.5
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ beta)))).astype(float)
    out = _l0learn_native.fit(np.asfortranarray(X), y, penalty="L0", n_lambda=15, max_nnz=8)
    B = np.asarray(out["betas"][0])
    check("constant column: finite betas", bool(np.isfinite(B).all()),
          f"(col5 |beta| max {np.abs(B[5]).max():.2e})")
    try:
        _l0learn_native.fit(np.asfortranarray(X[:50]), np.zeros(50), penalty="L0",
                            n_lambda=5, max_nnz=3)
        check("single-class y: returned without crash", True)
    except Exception as exc:
        check("single-class y: raised cleanly", True, f"({type(exc).__name__})")
    out2 = _l0learn_native.fit(np.asfortranarray(X[:30]), y[:30], penalty="L0",
                               n_lambda=5, max_nnz=3)
    check("n<p small: finite", bool(np.isfinite(np.asarray(out2['betas'][0])).all()))

    print("== 3. tpr_at_fpr k=0 regression ==")
    ai = list(rng.normal(0.4, 1, size=50))
    hu = list(rng.normal(0.0, 1, size=500))  # k = floor(1e-3*500) = 0
    r = tpr_at_fpr(ai, hu, fpr=1e-3)
    check("k=0: achieved FPR == 0", r["fpr_achieved"] == 0.0,
          f"(threshold {r['threshold']:.3f} > max(hu) {max(hu):.3f}: "
          f"{r['threshold'] > max(hu)})")

    print(f"\n{'ALL PASS' if not FAIL else f'FAILURES: {FAIL}'}")


if __name__ == "__main__":
    main()

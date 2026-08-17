"""FLEET E1 — protocol & metric invariants (oracle + property checks).

  1. auroc vs sklearn roc_auc_score on ties-heavy random data (500 trials)
  2. tpr_at_fpr invariants: achieved FPR <= target ALWAYS; TPR monotone in FPR
  3. split_buckets determinism + disjointness + per-bucket model mix
  4. FAMS family-slice alignment vs cache feature_names
  5. base_scores orientation sanity (detector AUROCs on C > 0.5)

Usage: .venv\\Scripts\\python scripts\\exp\\fleet_e1_invariants.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

FAIL = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


def main() -> None:
    rng = np.random.RandomState(0)

    print("== 1. auroc vs sklearn oracle (ties-heavy) ==")
    worst = 0.0
    for _ in range(500):
        ai = rng.randint(0, 7, size=rng.randint(5, 60)).astype(float)
        hu = rng.randint(0, 7, size=rng.randint(5, 60)).astype(float)
        mine = auroc(list(ai), list(hu))
        ref = roc_auc_score(np.r_[np.ones(len(ai)), np.zeros(len(hu))], np.r_[ai, hu])
        worst = max(worst, abs(mine - ref))
    check("auroc==roc_auc_score", worst < 1e-12, f"(worst |diff| {worst:.2e})")

    print("== 2. tpr_at_fpr invariants ==")
    ok_fpr, ok_mono = True, True
    for _ in range(500):
        ai = rng.randint(0, 5, size=200).astype(float)
        hu = rng.randint(0, 5, size=300).astype(float)
        prev = -1.0
        for fpr in (1e-3, 1e-2, 5e-2, 1e-1):
            r = tpr_at_fpr(list(ai), list(hu), fpr=fpr)
            ok_fpr &= r["fpr_achieved"] <= fpr + 1e-12
            ok_mono &= r["tpr"] >= prev - 1e-12
            prev = r["tpr"]
    check("achieved FPR <= target", ok_fpr)
    check("TPR monotone in FPR", ok_mono)

    print("== 3. split_buckets invariants ==")
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    b1 = split_buckets(df)
    b2 = split_buckets(df)
    check("deterministic", all(b1[k].index.equals(b2[k].index) for k in "ABC"))
    src = {k: set(b1[k].source_id) for k in "ABC"}
    check("buckets disjoint", not (src["A"] & src["B"] or src["A"] & src["C"] or src["B"] & src["C"]))
    check("source coverage", len(src["A"] | src["B"] | src["C"]) == df[df.fold == "dev"].source_id.nunique())
    for k in "ABC":
        mix = b1[k][b1[k].model != "human"].model.value_counts(normalize=True)
        check(f"bucket {k} spans models", len(mix) >= 10,
              f"({len(mix)} models, top {mix.index[0]} {mix.iloc[0]:.2f})")

    print("== 4. FAMS slice alignment vs cache names ==")
    names = list(np.load("data/derived/full_features.npz")["feature_names"])
    fams = {"rel_": (0, 8), "qg_": (8, 20), "ex_": (20, 31), "dct_": (31, 81),
            "shape_": (81, 89), "stat_": (89, 111), "cov": (111, 120),
            "col_": (120, 133), "chr_": (133, 156)}
    for fam, (lo, hi) in fams.items():
        block = names[lo:hi]
        check(f"slice {lo}:{hi} == {fam}*", all(n.startswith(fam) for n in block),
              f"({len(block)} cols)")

    print("== 5. base_scores orientation on C ==")
    data = np.load("data/derived/base_scores.npz")
    yc = data["labels_C"]
    for det in ("relative-burst", "qgram12", "exemplar", "dct-nobase"):
        s = data[f"{det}_C"]
        roc = auroc(list(s[yc == 1]), list(s[yc == 0]))
        check(f"{det} AUROC>0.5", roc > 0.5, f"({roc:.3f})")

    print(f"\n{'ALL PASS' if not FAIL else f'FAILURES: {FAIL}'}")


if __name__ == "__main__":
    main()

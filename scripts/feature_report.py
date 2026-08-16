"""Per-feature performance report -> docs/feature_report.md.

For every feature in the panel (81 cached + 8 shape), computed on bucket C
(never trained on): direction-corrected AUROC, TPR@FPR=1e-3 with achieved
FPR and Wilson CI, and finite-coverage %. Usage: .venv\\Scripts\\python scripts/feature_report.py
"""

import numpy as np
import pandas as pd

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr
from ai_text_detection.shape import SHAPE_FEATURE_NAMES, shape_features

OUT = "docs/feature_report.md"


def row_for(name: str, scores: np.ndarray, labels: np.ndarray, coverage: float) -> dict:
    finite = np.isfinite(scores)
    s, y = scores[finite], labels[finite]
    if len(s) < 50 or len(set(s)) < 3 or (y == 0).sum() == 0 or (y == 1).sum() == 0:
        return {"feature": name, "coverage": coverage, "auroc": np.nan, "sep": np.nan, "tpr": np.nan}
    roc = auroc(list(s[y == 1]), list(s[y == 0]))
    oriented = s if roc >= 0.5 else -s
    res = tpr_at_fpr(list(oriented[y == 1]), list(oriented[y == 0]))
    return {
        "feature": name,
        "coverage": coverage,
        "auroc": max(roc, 1 - roc),
        "direction": "ai>higher" if roc >= 0.5 else "ai>lower",
        "sep": abs(roc - 0.5) + 0.5,
        "tpr": res["tpr"],
        "ci": f"[{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]",
        "fpr_achieved": res["fpr_achieved"],
    }


def main() -> None:
    data = np.load("data/derived/full_features.npz")
    names = list(data["feature_names"])
    Xc, yc = data["X_C"], data["y_C"]

    rows = []
    for i, name in enumerate(names):
        col = Xc[:, i]
        rows.append(row_for(name, col, yc, float(np.isfinite(col).mean())))

    # shape features are not in the cache; compute on C
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    sub = split_buckets(df)["C"]
    shape_rows = [[shape_features(str(t))[k] for k in SHAPE_FEATURE_NAMES] for t in sub.generation]
    Xs = np.array(shape_rows, dtype=float)
    ys = (sub.model != "human").to_numpy(int)
    for i, name in enumerate(SHAPE_FEATURE_NAMES):
        col = Xs[:, i]
        rows.append(row_for(f"shape_{name}", col, ys, float(np.isfinite(col).mean())))

    table = pd.DataFrame(rows).sort_values("sep", ascending=False, na_position="last")
    cols = ("feature", "coverage", "auroc", "direction", "sep", "tpr", "ci", "fpr_achieved")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("# Per-feature performance on bucket C (source-disjoint, evaluated once)\n\n")
        fh.write("AUROC is direction-corrected (direction column gives the raw sign). ")
        fh.write("TPR at target FPR=1e-3 with achieved FPR + Wilson CI. coverage = fraction of docs with a finite value.\n\n")
        fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for _, r in table.iterrows():
            fh.write("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n")
        fh.write("\n")
    print(table.to_string(index=False, max_rows=30))
    print(f"\n{OUT} written ({len(table)} features)")


if __name__ == "__main__":
    main()

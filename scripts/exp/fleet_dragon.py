"""FLEET DRAGON — Kolmogorov proxies up the Navarro ladder (TODO #0).

Per-doc complexity statistics, cheapest-to-dearest:
  kol_lcp_*      self-similarity from the doc's own suffix array (Kasai LCP:
                 mean/p90/max) — intra-doc matching statistics
  kol_repair_*   Re-Pair grammar size: rules/char and (rules+final)/char —
                 grammar-based Kolmogorov proxy (greedy most-frequent-pair
                 substitution, deterministic)
  kol_sam_trans  SAM transition count / char (CDAWG-edge cousin)

DEV ONLY: solo AUROC/TPR@1e-2 (rank B, confirm C), then HGB increments.
Usage: .venv\\Scripts\\python scripts\\exp\\fleet_dragon.py  (slow: Re-Pair is Pythonic)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fleet_qgmid import eval_feat
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)
OUT = "docs/exp/fleet_dragon.md"
FEATS = ["kol_lcp_mean", "kol_lcp_p90", "kol_lcp_max",
         "kol_repair_rules_rate", "kol_repair_total_rate", "kol_sam_trans_rate"]


def lcp_feats(b: bytes) -> dict[str, float]:
    n = len(b)
    if n < 10:
        return {"kol_lcp_mean": np.nan, "kol_lcp_p90": np.nan, "kol_lcp_max": np.nan}
    sa = sorted(range(n), key=lambda i: b[i:])
    rank = [0] * n
    for i, s in enumerate(sa):
        rank[s] = i
    # Kasai
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
    a = np.array(lcp, dtype=float)
    return {"kol_lcp_mean": float(a.mean()), "kol_lcp_p90": float(np.percentile(a, 90)),
            "kol_lcp_max": float(a.max())}


def repair_feats(b: bytes) -> dict[str, float]:
    n0 = len(b)
    if n0 < 10:
        return {"kol_repair_rules_rate": np.nan, "kol_repair_total_rate": np.nan}
    seq = list(b)
    rules = 0
    top = n0  # fresh symbols start above byte range
    while True:
        counts: dict[tuple[int, int], int] = {}
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        (a, c), best = max(counts.items(), key=lambda kv: kv[1])
        if best < 2:
            break
        rules += 1
        top += 1
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
    return {"kol_repair_rules_rate": rules / n0,
            "kol_repair_total_rate": (rules + len(seq)) / n0}


def sam_trans_rate(b: bytes) -> float:
    if len(b) < 10:
        return np.nan
    link = [-1]
    length = [0]
    trans: list[dict[int, int]] = [{}]
    last = 0
    for c in b:
        cur = len(length)
        length.append(length[last] + 1)
        link.append(0)
        trans.append({})
        p = last
        while p != -1 and c not in trans[p]:
            trans[p][c] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = trans[p][c]
            if length[p] + 1 == length[q]:
                link[cur] = q
            else:
                clone = len(length)
                length.append(length[p] + 1)
                trans.append(trans[q].copy())
                link.append(link[q])
                while p != -1 and trans[p].get(c) == q:
                    trans[p][c] = clone
                    p = link[p]
                link[q] = link[cur] = clone
        last = cur
    return sum(len(t) for t in trans) / len(b)


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    vals: dict[str, np.ndarray] = {}
    for b in "ABC":
        rows = []
        for t in buckets[b].generation:
            text = str(t).encode("utf-8")
            row = lcp_feats(text)
            row.update(repair_feats(text))
            row["kol_sam_trans_rate"] = sam_trans_rate(text)
            rows.append([row[f] for f in FEATS])
        vals[b] = np.array(rows)
        print(f"bucket {b} done", flush=True)

    lines = ["# FLEET DRAGON — Kolmogorov proxies (Navarro ladder)\n\n",
             "| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |\n|---|---|---|---|---|\n"]
    for j, f in enumerate(FEATS):
        rb = eval_feat(vals["B"][:, j], labels["B"])
        rc = eval_feat(vals["C"][:, j], labels["C"])
        lines.append(f"| {f} | {rb[1]:.3f} | {rc[1]:.3f} | {rb[2]:.3f} | {rc[2]:.3f} |\n")

    panel = np.load("data/derived/full_features.npz")
    Xp = {b: panel[f"X_{b}"].astype(float) for b in "ABC"}
    means = np.nan_to_num(np.nanmean(Xp["A"], axis=0))

    def prep(X):
        X = X.copy()
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(means, bad[1])
        return X

    new_means = np.nan_to_num(np.nanmean(vals["A"], axis=0))

    def prep_new(b):
        X = vals[b].copy()
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(new_means, bad[1])
        return X

    lines.append("\n## HGB increment (train A, read C)\n\n")
    lines.append("| arm | n | AUROC C | TPR@1e-2 C |\n|---|---|---|---|\n")
    for arm, get in (("panel153", lambda b: prep(Xp[b])),
                     ("panel+dragon159", lambda b: np.column_stack([prep(Xp[b]), prep_new(b)])),
                     ("dragon6 alone", lambda b: prep_new(b))):
        m = HistGradientBoostingClassifier(**HGB_PARAMS).fit(get("A"), labels["A"])
        s = m.predict_proba(get("C"))[:, 1]
        roc = auroc(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
        r = tpr_at_fpr(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]), fpr=1e-2)
        lines.append(f"| {arm} | {get('A').shape[1]} | {roc:.3f} | "
                     f"{r['tpr']:.3f} [{r['tpr_lo']:.3f},{r['tpr_hi']:.3f}] |\n")
        print(lines[-1], flush=True)

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print(f"{OUT} written")


if __name__ == "__main__":
    main()

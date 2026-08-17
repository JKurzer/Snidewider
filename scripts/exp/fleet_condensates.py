"""FLEET K — the condensate packs (TODO 1+2+3): delta, ms_, structure-entropy.

  delta     repetitiveness: d_k = distinct char k-mers / k for k=1..8, max &
            argmax; word-level for k=1..3 (reversal-invariant per CPM 2026)
  ms        matching-statistics approximation vs A-built references: per
            position, longest k in K_POINTS whose k-mer hash sits in the
            reference (k-set approximation of the MS vector; exact-SAM path
            documented in docs/condensates.md). vs human ref, vs AI ref, plus
            contrasts. Hashes are SipHash ints (collision odds negligible at
            this scale; this is a statistical feature, not a proof).
  structure per-doc index sizes: SAM states, BWT runs, LZ77 phrases (rates).
            The principled successors of zlib_ratio (equi2026-qpm-review.md).

DEV ONLY. Solo AUROC/TPR@1e-2, rank on B; then HGB increment over the 153.
Usage: .venv\\Scripts\\python scripts\\exp\\fleet_condensates.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fleet_qgmid import eval_feat
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection import qgram
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr
from ai_text_detection.stats_features import WORD_RE

K_POINTS = (2, 3, 5, 8, 16, 32)
HGB_PARAMS = dict(max_iter=300, max_depth=4, learning_rate=0.08,
                  max_features=0.5, random_state=7)
OUT = "docs/exp/fleet_condensates.md"

DELTA_NAMES = ([f"delta_{k}" for k in range(1, 9)] + ["delta_max", "delta_argmax"]
               + [f"wdelta_{k}" for k in range(1, 4)])
MS_STATS = ("mean", "p90", "max", "frac0", "frac_ge8", "frac_ge16")
MS_NAMES = ([f"ms_hu_{s}" for s in MS_STATS] + [f"ms_ai_{s}" for s in MS_STATS]
            + ["ms_contrast_mean", "ms_contrast_frac16"])
STRUCT_NAMES = ["sam_states_rate", "bwt_runs_rate", "lz77_phrases_rate"]
FEATS = DELTA_NAMES + MS_NAMES + STRUCT_NAMES


def delta_feats(text: str) -> dict[str, float]:
    b = text.encode("utf-8")
    out = {}
    ds = []
    for k in range(1, 9):
        d = len(qgram.profile(b, k)) / k if len(b) >= k else np.nan
        out[f"delta_{k}"] = d
        ds.append(d)
    if any(np.isfinite(ds)):
        out["delta_max"] = float(np.nanmax(ds))
        out["delta_argmax"] = float(np.nanargmax(ds) + 1)
    else:
        out["delta_max"] = out["delta_argmax"] = np.nan
    toks = [w.lower() for w in WORD_RE.findall(text)]
    for k in (1, 2, 3):
        grams = {tuple(toks[i:i + k]) for i in range(len(toks) - k + 1)}
        out[f"wdelta_{k}"] = len(grams) / k if toks else np.nan
    return out


def build_ref(texts) -> dict[int, set[int]]:
    blob = b"\n\n".join(str(t).encode("utf-8") for t in texts)
    return {k: {hash(blob[i:i + k]) for i in range(len(blob) - k + 1)} for k in K_POINTS}


def ms_vector(b: bytes, ref: dict[int, set[int]]) -> np.ndarray:
    n = len(b)
    L = np.zeros(n, dtype=float)
    for i in range(n - 1):
        for k in K_POINTS:
            if i + k <= n and hash(b[i:i + k]) in ref[k]:
                L[i] = k
    return L


def ms_feats(text: str, ref_hu, ref_ai) -> dict[str, float]:
    b = text.encode("utf-8")
    if len(b) < 40:
        return {f"ms_{c}_{s}": np.nan for c in ("hu", "ai") for s in MS_STATS} | \
               {"ms_contrast_mean": np.nan, "ms_contrast_frac16": np.nan}
    out = {}
    means = {}
    for tag, ref in (("hu", ref_hu), ("ai", ref_ai)):
        L = ms_vector(b, ref)
        means[tag] = float(L.mean())
        out[f"ms_{tag}_mean"] = means[tag]
        out[f"ms_{tag}_p90"] = float(np.percentile(L, 90))
        out[f"ms_{tag}_max"] = float(L.max())
        out[f"ms_{tag}_frac0"] = float((L == 0).mean())
        out[f"ms_{tag}_frac_ge8"] = float((L >= 8).mean())
        out[f"ms_{tag}_frac_ge16"] = float((L >= 16).mean())
    out["ms_contrast_mean"] = means["ai"] - means["hu"]
    out["ms_contrast_frac16"] = out["ms_ai_frac_ge16"] - out["ms_hu_frac_ge16"]
    return out


def sam_states_rate(b: bytes) -> float:
    if len(b) < 10:
        return np.nan
    # incremental suffix automaton; states capped by 2n-1
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
    return len(length) / len(b)


def bwt_runs_rate(b: bytes) -> float:
    # canonical r: BWT over the SENTINEL-terminated string (fleet audit:
    # bare-suffix order \!= rotation order; only the sentinel version is the
    # literature's r). 0x00 never occurs in valid UTF-8 -> safe sentinel.
    n = len(b)
    if n < 10:
        return np.nan
    s = b + b"\x00"
    m = len(s)
    sa = sorted(range(m), key=lambda i: s[i:])
    bw = bytes(s[i - 1] if i else s[-1] for i in sa)
    runs = 1 + sum(1 for i in range(1, m) if bw[i] != bw[i - 1])
    return runs / m


def lz77_phrases_rate(b: bytes) -> float:
    n = len(b)
    if n < 10:
        return np.nan
    pos: dict[bytes, int] = {}
    i = phrases = 0
    while i < n:
        best_len = 0
        if i + 4 <= n:
            j = pos.get(b[i:i + 4], -1)
            if j >= 0:
                while i + best_len < n and b[j + best_len] == b[i + best_len]:
                    best_len += 1
        phrases += 1
        step = max(1, best_len)
        for t in range(i, min(i + step, n - 3)):
            pos[b[t:t + 4]] = t
        i += step
    return phrases / n


def struct_feats(text: str) -> dict[str, float]:
    b = text.encode("utf-8")
    return {"sam_states_rate": sam_states_rate(b),
            "bwt_runs_rate": bwt_runs_rate(b),
            "lz77_phrases_rate": lz77_phrases_rate(b)}


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {b: (sub.model != "human").to_numpy(int) for b, sub in buckets.items()}

    a = buckets["A"]
    # cross-bucket refs (fleet-F3/G lesson: scoring A rows vs A-built refs is
    # in-fold inflation). A rows vs B+C grams; B/C rows vs A grams.
    ref_a = (build_ref(a[a.model == "human"].generation),
             build_ref(a[a.model != "human"].generation))
    bc = pd.concat([buckets["B"], buckets["C"]])
    ref_bc = (build_ref(bc[bc.model == "human"].generation),
              build_ref(bc[bc.model != "human"].generation))
    print("ms refs built (cross-bucket)", flush=True)

    vals: dict[str, dict[str, np.ndarray]] = {}
    for b in "ABC":
        rhu, rai = (ref_bc if b == "A" else ref_a)
        rows = []
        for t in buckets[b].generation:
            text = str(t)
            row = delta_feats(text)
            row.update(ms_feats(text, rhu, rai))
            row.update(struct_feats(text))
            rows.append([row[f] for f in FEATS])
        vals[b] = np.array(rows)
        print(f"bucket {b} done", flush=True)

    lines = ["# FLEET K — condensate packs (delta / ms / structure-entropy)\n\n",
             "| feature | AUROC B | AUROC C | TPR@1e-2 B | TPR@1e-2 C |\n|---|---|---|---|---|\n"]
    rows_out = []
    for j, f in enumerate(FEATS):
        rb = eval_feat(vals["B"][:, j], labels["B"])
        rc = eval_feat(vals["C"][:, j], labels["C"])
        rows_out.append((f, rb, rc))
    rows_out.sort(key=lambda r: np.nan_to_num(r[1][1]), reverse=True)
    for f, rb, rc in rows_out:
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
                     ("panel+cond183", lambda b: np.column_stack([prep(Xp[b]), prep_new(b)])),
                     ("cond30 alone", lambda b: prep_new(b))):
        m = HistGradientBoostingClassifier(**HGB_PARAMS).fit(get("A"), labels["A"])
        s = m.predict_proba(get("C"))[:, 1]
        roc = auroc(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
        r = tpr_at_fpr(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]), fpr=1e-2)
        lines.append(f"| {arm} | {get('A').shape[1]} | {roc:.3f} | "
                     f"{r['tpr']:.3f} [{r['tpr_lo']:.3f},{r['tpr_hi']:.3f}] |\n")
        print(lines[-1], flush=True)

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print(f"{OUT} written | {len(FEATS)} features")


if __name__ == "__main__":
    main()

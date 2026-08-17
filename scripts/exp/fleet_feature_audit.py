"""FLEET I — per-feature bug audit over the 156-feat dev cache.

Flags:
  degenerate   zero variance or all-NaN on a bucket
  nan-heavy    NaN rate > 50% everywhere (coverage smell; known for long-doc gates)
  dup-pair     |spearman rho| > 0.9999 with another feature (copy/unit bug smell)
  orient-flip  AUROC direction disagrees between buckets (unreliable or buggy)
  length-ghost |spearman(feature, doc tokens)| > 0.95 (the F2 byte-length lesson)
  sentinel     edge texts (empty/1-word/same-char/punct-only) crash or misbehave

Usage: .venv\\Scripts\\python scripts\\exp\\fleet_feature_audit.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_text_detection.charstat import charstat_features
from ai_text_detection.collapse import collapse_features
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.feature_sets import qgram12_vector, relative_vector
from ai_text_detection.metrics import auroc
from ai_text_detection.shape import shape_features
from ai_text_detection.stats_features import stat_features

OUT = "docs/exp/fleet_feature_audit.md"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    cache = np.load("data/derived/full_features.npz")
    names = list(cache["feature_names"])
    X = {b: cache[f"X_{b}"].astype(float) for b in "ABC"}
    y = {b: cache[f"y_{b}"].astype(int) for b in "ABC"}
    n_tok = {b: np.array([len(str(t).split()) for t in buckets[b].generation], dtype=float)
             for b in "ABC"}

    lines = ["# FLEET I — per-feature bug audit (156 cols, dev cache)\n\n"]
    flags: list[str] = []

    for j, name in enumerate(names):
        notes = []
        signs = []
        for b in "ABC":
            col = X[b][:, j]
            finite = np.isfinite(col)
            nan_rate = 1 - finite.mean()
            if finite.sum() == 0:
                notes.append(f"ALL-NAN on {b}")
                continue
            if np.nanstd(col) == 0:
                notes.append(f"ZERO-VAR on {b}")
            if nan_rate > 0.5:
                notes.append(f"nan{b}={nan_rate:.2f}")
            ai, hu = col[(y[b] == 1) & finite], col[(y[b] == 0) & finite]
            if len(ai) >= 30 and len(hu) >= 30 and np.nanstd(col) > 0:
                signs.append(np.sign(auroc(list(ai), list(hu)) - 0.5))
            rho_len = spearman(col, n_tok[b])
            if np.isfinite(rho_len) and abs(rho_len) > 0.95:
                notes.append(f"LENGTH-GHOST {b} rho={rho_len:.3f}")
        if signs and len(set(signs)) > 1:
            notes.append(f"ORIENT-FLIP signs={signs}")
        if notes:
            flags.append(f"| {name} | {'; '.join(notes)} |\n")

    lines.append("## flagged features\n\n| feature | flags |\n|---|---|\n")
    lines.extend(flags or ["| — | none |\n"])

    # duplicate pairs (cheap: corr on bucket C finite rows)
    lines.append("\n## near-duplicate pairs (|rho| > 0.9999 on C)\n\n")
    Xc = X["C"]
    dups = []
    sd = np.nanstd(Xc, axis=0)
    live = np.where(sd > 0)[0]
    for ii in range(len(live)):
        for jj in range(ii + 1, len(live)):
            i, k = live[ii], live[jj]
            r = spearman(Xc[:, i], Xc[:, k])
            if np.isfinite(r) and abs(r) > 0.9999:
                dups.append(f"| {names[i]} | {names[k]} | {r:+.5f} |\n")
    lines.append("| feature a | feature b | rho |\n|---|---|---|\n")
    lines.extend(dups or ["| — | — | none |\n"])

    # sentinels on the pure per-doc packs
    lines.append("\n## sentinels (pure packs on edge texts)\n\n")
    edges = {"empty": "", "one-word": "hello", "same-char": "a" * 500,
             "punct-only": "!!!???,,;" * 20, "short-eng": "The cat sat. It was fine."}
    for tag, text in edges.items():
        try:
            vals = [stat_features(text), collapse_features(text),
                    charstat_features(text), shape_features(text),
                    dict(zip(range(12), qgram12_vector(text))), ]
            bad = None
            _ = relative_vector(text)
            lines.append(f"- {tag}: ok (no crash)\n")
        except Exception as exc:  # noqa: BLE001 - audit wants the type
            lines.append(f"- {tag}: **CRASH {type(exc).__name__}: {exc}**\n")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print("".join(lines))
    print(f"{OUT} written | {len(flags)} flagged | {len(dups)} dup pairs")


if __name__ == "__main__":
    main()

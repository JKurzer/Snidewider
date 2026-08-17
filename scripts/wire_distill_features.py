"""Wire the distillation packs (collapse + charstat) into both caches (120 -> 156).

Appends 13 col_* + 23 chr_* columns after the existing 120. Idempotent.
The chi2 reference is the bucket-A human char distribution, frozen into
charstat.ENGLISH_CHAR_REF on first wire (gate-spec pattern: derived once,
committed, reused forever). No rolling comparisons anywhere in these packs.

Usage: .venv\\Scripts\\python scripts\\wire_distill_features.py
"""

import string
from collections import Counter

import numpy as np
import pandas as pd

from ai_text_detection.charstat import CHARSTAT_FEATURE_NAMES, charstat_features
from ai_text_detection.collapse import COLLAPSE_FEATURE_NAMES, collapse_features
from ai_text_detection.evaldata import split_buckets

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"
NEW_NAMES = [f"col_{n}" for n in COLLAPSE_FEATURE_NAMES] + \
            [f"chr_{n}" for n in CHARSTAT_FEATURE_NAMES]


def distill_matrix(texts, ref) -> np.ndarray:
    rows = []
    for t in texts:
        text = str(t)
        col = collapse_features(text)
        chr_ = charstat_features(text, ref)
        rows.append([col[k] for k in COLLAPSE_FEATURE_NAMES]
                    + [chr_[k] for k in CHARSTAT_FEATURE_NAMES])
    return np.array(rows, dtype=float)


def rewire(path: str, keys: list[str], new_cols: dict[str, np.ndarray]) -> None:
    cache = np.load(path)
    names = [n for n in cache["feature_names"]
             if not (n.startswith("col_") or n.startswith("chr_"))]
    store = {k: cache[k] for k in cache if k != "feature_names"}
    for key in keys:
        store[key] = np.column_stack([store[key][:, : len(names)], new_cols[key]])
    store["feature_names"] = np.array(names + NEW_NAMES)
    np.savez(path, **store)
    print(f"{path}: -> {store[keys[0]].shape[1]} features")


def main() -> None:
    import ai_text_detection.charstat as charstat_mod

    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    ref = charstat_mod.ENGLISH_CHAR_REF
    if not ref:
        a_hu = buckets["A"][buckets["A"].model == "human"]
        counts: Counter = Counter()
        for t in a_hu.generation:
            counts.update(str(t).lower())
        total = sum(counts.values())
        ref = {c: counts.get(c, 0) / total for c in string.printable[:95]}
        print("ref derived from A humans (freeze it into charstat.py!)", flush=True)

    dev_new = {f"X_{b}": distill_matrix(buckets[b].generation, ref) for b in "ABC"}
    for b in "ABC":
        print(f"dev {b}: {dev_new[f'X_{b}'].shape}", flush=True)
    rewire(DEV_NPZ, ["X_A", "X_B", "X_C"], dev_new)

    hold = df[df.fold == "holdout"]
    hold_new = {
        "X_hu": distill_matrix(hold[hold.model == "human"].generation, ref),
        "X_ai": distill_matrix(hold[hold.model != "human"].sample(n=20_000, random_state=97)
                               .generation, ref),
    }
    for k, m in hold_new.items():
        print(f"holdout {k}: {m.shape}", flush=True)
    rewire(HOLD_NPZ, ["X_hu", "X_ai"], hold_new)

    print("\nENGLISH_CHAR_REF literal (paste into charstat.py):")
    print("{" + ", ".join(f"{c!r}: {p:.8f}" for c, p in sorted(ref.items()) if p > 0) + "}")


if __name__ == "__main__":
    main()

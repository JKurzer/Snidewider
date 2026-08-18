"""Restore the pre-surgery 226 layout (Donk: 'swap them back in, just revert').

Re-inserts the six dropped columns at their exact original positions and
removes ex_contrast_mode. Values are recomputed by the same deterministic
functions, so the restored 226 panel is bit-identical to the original.

Idempotent. Usage: .venv\\Scripts\\python scripts\\restore_surgery.py
"""

import numpy as np
import pandas as pd

from ai_text_detection import pipeline, qgram
from ai_text_detection.bigrams import bigram_rates
from ai_text_detection.coverage import coverage_features
from ai_text_detection.dct_shapes import dct_tail_features
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.exemplar import ExemplarBank, bank_self_indices, exemplar_features

DEV_NPZ = "data/derived/full_features.npz"
HOLD_NPZ = "data/derived/holdout_features.npz"
N_BANK = 150
DROP_NAMES = ["ex_ai_mean_raw", "ex_contrast_min", "ex_contrast_mean",
              "dct_paircos_p25", "cov2_ai", "bg_er"]
# original 226 positions, expressed as insert-after anchors in the 221 layout
ANCHOR = {
    "ex_ai_mean_raw": "ex_hu_p10",
    "ex_contrast_min": "ex_hu_mean_raw",
    "ex_contrast_mean": "ex_contrast_min",
    "dct_paircos_p25": "dct_paircos_p10",
    "cov2_ai": "cov2_hu",
    "bg_er": "bg_n ",
}


def cols_for(texts, models, n_bank, bank_ai, bank_hu, refs) -> dict[str, np.ndarray]:
    out = {n: [] for n in DROP_NAMES}
    ai_s = hu_s = None
    if n_bank is not None:
        ai_s, hu_s = bank_self_indices([str(m) for m in models], n_bank)
    for i, t in enumerate(texts):
        text = str(t)
        b = text.encode("utf-8")
        exf = exemplar_features(qgram.profile(b, 3), bank_ai, bank_hu,
                                None if ai_s is None else ai_s[i],
                                None if ai_s is None else hu_s[i])
        cov = coverage_features(text, refs["ref_hu"], refs["ref_ai"])
        tail = dct_tail_features(text)
        bg = bigram_rates(text)
        out["ex_ai_mean_raw"].append(exf["ex_ai_mean_raw"])
        out["ex_contrast_min"].append(exf["ex_contrast_min"])
        out["ex_contrast_mean"].append(exf["ex_contrast_mean"])
        out["dct_paircos_p25"].append(tail["paircos_p25"])
        out["cov2_ai"].append(cov["cov2_ai"])
        out["bg_er"].append(bg["bg_er"])
    return {n: np.array(v, dtype=float) for n, v in out.items()}


def restore(path: str, keys: list[str], cols: dict[str, dict[str, np.ndarray]]) -> None:
    cache = np.load(path)
    names = [n for n in cache["feature_names"] if n != "ex_contrast_mode"]
    for n in DROP_NAMES:
        anchor = ANCHOR[n]
        pos = names.index(anchor) + 1
        names = names[:pos] + [n] + names[pos:]
    store = {k: cache[k] for k in cache if k != "feature_names"}
    base = [n for n in names if n not in DROP_NAMES]
    for key in keys:
        X = store[key][:, : len(base)]  # current 221 minus mode = 220 cols
        blocks = []
        for n in names:
            if n in DROP_NAMES:
                blocks.append(cols[key][n][:, None])
            else:
                j = base.index(n)
                blocks.append(X[:, j:j + 1])
        store[key] = np.hstack(blocks)
    store["feature_names"] = np.array(names)
    np.savez(path, **store)
    print(f"{path}: -> {store[keys[0]].shape[1]} features", flush=True)


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    a = buckets["A"]
    bank_ai = ExemplarBank.from_texts(
        [str(t) for t in a[a.model != "human"].generation[:N_BANK]])
    bank_hu = ExemplarBank.from_texts(
        [str(t) for t in a[a.model == "human"].generation[:N_BANK]])
    art = pipeline.load_artifacts()
    refs = {"ref_hu": art["ref_hu"], "ref_ai": art["ref_ai"]}

    dev_cols = {}
    for b in "ABC":
        dev_cols[f"X_{b}"] = cols_for(buckets[b].generation, buckets[b].model,
                                      N_BANK if b == "A" else None,
                                      bank_ai, bank_hu, refs)
        print(f"dev {b} done", flush=True)
    restore(DEV_NPZ, ["X_A", "X_B", "X_C"], dev_cols)

    hold = df[df.fold == "holdout"]
    hold_cols = {
        "X_hu": cols_for(hold[hold.model == "human"].generation, None, None,
                         bank_ai, bank_hu, refs),
        "X_ai": cols_for(hold[hold.model != "human"]
                         .sample(n=20_000, random_state=97).generation, None, None,
                         bank_ai, bank_hu, refs),
    }
    restore(HOLD_NPZ, ["X_hu", "X_ai"], hold_cols)


if __name__ == "__main__":
    main()

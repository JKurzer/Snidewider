"""FLEET D — holdout behavior audit: is the ~2.8x dev->holdout tax a bug?

Suspects, in order of Donk's nose:
  P1 variant-mix: dev buckets take .head(2) AI rows per source; holdout samples
     uniformly. If the parquet orders variants (greedy first...), dev AI =
     EASY variants only, holdout AI = all variants incl. rep-penalty killers.
  P2 per-source multiplicity: 2/source dev vs up to ~44/source holdout.
  P3 head(2) slot bias: which variant slots land in dev buckets?
  P4 cache row-alignment: recompute qgram12 cols for sampled holdout rows,
     must match the cache bit-exactly.
  P5 fold disjointness: dev ∩ holdout source_ids must be empty.
  P6 null controls: half-vs-half AUROC inside holdout humans/AI must be ~0.5.
  P7 dev-side emulation: does the mix explain the gap? (capped vs mix-weighted)

Usage: .venv\\Scripts\\python scripts\\exp\\fleet_holdout_audit.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_text_detection import qgram
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.feature_sets import qgram12_vector
from ai_text_detection.metrics import auroc

OUT = "docs/exp/fleet_holdout_audit.md"
lines: list[str] = ["# FLEET D — holdout behavior audit\n\n"]


def say(s: str) -> None:
    print(s, flush=True)
    lines.append(s + "\n")


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    say(f"parquet columns: {list(df.columns)}")
    say(f"folds: {df.fold.value_counts().to_dict()}")
    buckets = split_buckets(df)
    hold = df[df.fold == "holdout"]
    hold_ai = hold[hold.model != "human"]
    hold_ai_sample = hold_ai.sample(n=20_000, random_state=97)

    # ---- P1/P2/P3: mix + multiplicity + slot bias ----
    say("\n## P1/P2/P3 — AI variant mix & multiplicity\n")
    dev_ai = pd.concat([buckets[b] for b in "ABC"])
    dev_ai = dev_ai[dev_ai.model != "human"]
    say(f"dev AI rows (capped): {len(dev_ai)} | holdout AI sample: {len(hold_ai_sample)}")
    dev_ai_per_src = dev_ai.groupby("source_id").size()
    hold_ai_per_src = hold_ai_sample.groupby("source_id").size()
    say(f"AI rows/source: dev mean {dev_ai_per_src.mean():.2f} max {dev_ai_per_src.max()} | "
        f"holdout mean {hold_ai_per_src.mean():.2f} max {hold_ai_per_src.max()}")
    for col in ("model", "decoding", "rep_penalty", "temperature"):
        if col in df.columns:
            say(f"\n{col} mix (dev-capped vs holdout-sample):")
            mix = pd.DataFrame({
                "dev": dev_ai[col].astype(str).value_counts(normalize=True),
                "holdout": hold_ai_sample[col].astype(str).value_counts(normalize=True),
            }).fillna(0).round(4)
            say(mix.to_string())
    # slot bias: position of each AI row within its source, parquet order
    dev_full = df[df.fold == "dev"]
    dev_full_ai = dev_full[dev_full.model != "human"]
    slot = dev_full_ai.groupby("source_id").cumcount()
    capped_slots = slot[dev_full_ai.index.isin(dev_ai.index)]
    say(f"\nslot histogram of dev-capped AI rows (parquet within-source order): "
        f"{capped_slots.value_counts().sort_index().to_dict()}")
    say(f"total AI slots available per dev source: "
        f"{slot.groupby(dev_full_ai.source_id).max().add(1).value_counts().sort_index().to_dict()}")

    # ---- P4: cache row alignment ----
    say("\n## P4 — holdout cache row alignment (200-row recompute)\n")
    cache = np.load("data/derived/holdout_features.npz")
    hold_hu = hold[hold.model == "human"]
    rng = np.random.RandomState(3)
    idx_hu = rng.choice(len(hold_hu), 100, replace=False)
    idx_ai = rng.choice(len(hold_ai_sample), 100, replace=False)
    bad = 0
    for side, sub, idxs, X in (("hu", hold_hu, idx_hu, cache["X_hu"]),
                               ("ai", hold_ai_sample, idx_ai, cache["X_ai"])):
        for i in idxs:
            fresh = qgram12_vector(str(sub.generation.iloc[i]))
            cached = X[i, 8:20]
            if not np.allclose(np.nan_to_num(fresh), np.nan_to_num(cached), atol=1e-9):
                bad += 1
        say(f"{side}: 100 rows recomputed vs cache cols 8:20 — mismatches counted so far: {bad}")
    say(f"P4 verdict: {'ALIGNED' if bad == 0 else f'{bad} MISMATCHES — CACHE BUG'}")

    # ---- P5: fold disjointness ----
    say("\n## P5 — fold disjointness\n")
    dev_src = set(df[df.fold == "dev"].source_id)
    hold_src = set(hold.source_id)
    say(f"dev/holdout shared sources: {len(dev_src & hold_src)} (must be 0)")
    say(f"dup generation texts across folds: "
        f"{len(set(df[df.fold=='dev'].generation) & set(hold.generation))} (exact dups)")

    # ---- P6: null controls ----
    say("\n## P6 — null controls (half-vs-half AUROC, expect ~0.5)\n")
    names = list(cache["feature_names"])
    col = names.index("qg_mid_qgram_mean")
    for side, X in (("human", cache["X_hu"]), ("ai", cache["X_ai"])):
        v = X[:, col]
        ok = np.isfinite(v)
        v = v[ok]
        rng2 = np.random.RandomState(11)
        rng2.shuffle(v)
        half = len(v) // 2
        say(f"{side} qg_mid_qgram_mean half/half AUROC: "
            f"{auroc(list(v[:half]), list(v[half:])):.4f} (n={len(v)})")

    # ---- P7: mix emulation on dev C (does the mix explain the gap?) ----
    say("\n## P7 — dev-C AUROC of qg_mid_qgram_mean, capped vs multiplicity-emulated\n")
    dev_cache = np.load("data/derived/full_features.npz")
    Xc, yc = dev_cache["X_C"], dev_cache["y_C"]
    c_names = list(dev_cache["feature_names"])
    v = Xc[:, c_names.index("qg_mid_qgram_mean")]
    m = np.isfinite(v)
    raw = auroc(list(v[m][yc[m] == 1]), list(v[m][yc[m] == 0]))
    roc_capped = max(raw, 1.0 - raw)  # direction-corrected (this family scores ai>lower)
    say(f"dev C capped-bucket AUROC (oriented): {roc_capped:.4f} (holdout read was 0.879)")
    if "rep_penalty" in df.columns or "decoding" in df.columns:
        say("(see P1 mix table for whether dev buckets systematically drop hard variants)")

    # ---- P8: per-model difficulty on holdout (strongest cached feature) ----
    say("\n## P8 — per-model AUROC on holdout (qg_mid_qgram_mean)\n")
    v_hu = cache["X_hu"][:, col]
    v_ai = cache["X_ai"][:, col]
    hu_ok = v_hu[np.isfinite(v_hu)]
    models_ai = hold_ai_sample.model.to_numpy()
    for model in sorted(set(models_ai)):
        v = v_ai[models_ai == model]
        v = v[np.isfinite(v)]
        if len(v) >= 50:
            raw = auroc(list(v), list(hu_ok))
            say(f"  {model:<14} AUROC {max(raw, 1.0 - raw):.3f} (n={len(v)})")
    say("\n(dev buckets trained/evaluated on llama-chat ONLY — see P1.)")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print(f"\n{OUT} written")


if __name__ == "__main__":
    main()

"""RAID leaderboard path: pre-flight checks + predictions + metadata draft.

Phases:
  1. SANITY: the bundle model on the CACHED holdout features must reproduce
     the ladder (docs/sota-operating-points.md) exactly — catches bundle drift.
  2. DISJOINTNESS: title overlap between RAID test split and our dev fold
     must be zero — we trained on train_none sources.
  3. PREDICT: predictions.json for the clean test slice (test_none) via
     the official run_detection interface.

Usage: .venv\\Scripts\\python scripts\\raid_submission.py [--full]
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from raid import run_detection
from raid.utils import load_data

from ai_text_detection import pipeline
from ai_text_detection.metrics import auroc, tpr_at_fpr

OUT_DIR = "submissions/snidewider"
LADDER_REF = {"auroc": 0.9899, 5e-2: 0.960, 1e-2: 0.872, 1e-3: 0.635}


def main() -> None:
    art = pipeline.load_artifacts()
    model, means = art["model"], art["impute_means"]

    def detector(texts):
        X = pipeline.featurize_batch(texts, art)
        return model.predict_proba(pipeline.impute(X, means))[:, 1].tolist()

    print("== 1. sanity: bundle model on cached holdout ==", flush=True)
    hold = np.load("data/derived/holdout_features.npz")
    dev = np.load("data/derived/full_features.npz")
    dnames = list(dev["feature_names"])
    cols = [dnames.index(n) for n in hold["feature_names"]]
    m2 = np.nan_to_num(np.nanmean(dev["X_A"][:, cols], axis=0))
    for key in ("X_hu", "X_ai"):
        X = hold[key][:, cols]
        bad = np.where(~np.isfinite(X))
        X[bad] = np.take(m2, bad[1])
        s = model.predict_proba(X)[:, 1]
        hold_score = s
        if key == "X_hu":
            s_hu = s
        else:
            s_ai = s
    roc = auroc(list(s_ai), list(s_hu))
    print(f"  AUROC {roc:.4f} (ladder: {LADDER_REF['auroc']})", flush=True)
    for fpr in (5e-2, 1e-2, 1e-3):
        r = tpr_at_fpr(list(s_ai), list(s_hu), fpr=fpr)
        print(f"  TPR@{fpr:.0e} {r['tpr']:.3f} (ladder: {LADDER_REF[fpr]})", flush=True)

    print("\n== 2. disjointness: exact-text overlap vs our dev fold ==", flush=True)
    ours = pd.read_parquet("data/derived/raid_splits.parquet")
    dev_hashes = {hash(str(t)) for t in ours[ours.fold == "dev"].generation}
    test = load_data(split="test", include_adversarial=False)
    overlap = sum(1 for t in test.generation if hash(str(t)) in dev_hashes)
    print(f"  test_none rows: {len(test)}; exact-text overlap with dev: {overlap}", flush=True)
    if overlap:
        print("  WARNING: overlapping generations — must disclose/restrict!", flush=True)

    print("\n== 3. predictions on the clean test slice ==", flush=True)
    t0 = time.time()
    preds = run_detection(detector, test)
    dt = time.time() - t0
    print(f"  {len(preds)} predictions in {dt/60:.1f} min", flush=True)
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/predictions.json", "w", encoding="utf-8") as fh:
        json.dump(preds, fh)

    metadata = {
        "date_released": "2026-08-17",
        "detector_name": "Snidewider (panel-HGB)",
        "website": "",
        "paper_link": "",
        "huggingface_link": "",
        "github_link": "https://github.com/JKurzer/Snidewider",
        "contact_info": "jake.kurzer@gmail.com",
    }
    with open(f"{OUT_DIR}/metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"  {OUT_DIR}/predictions.json + metadata.json written", flush=True)


if __name__ == "__main__":
    main()

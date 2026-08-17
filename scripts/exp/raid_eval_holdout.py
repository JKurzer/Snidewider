"""Final pre-flight: run OUR holdout through the OFFICIAL raid evaluation.

Same frozen bundle model, same cached holdout features, but scored by
raid.run_evaluation (their thresholds, per-domain, 5% FPR, their accuracy).
What the leaderboard bot would compute if our holdout were the test set.
"""

import numpy as np
import pandas as pd
from raid import run_evaluation

from ai_text_detection import pipeline


def main() -> None:
    art = pipeline.load_artifacts()
    model, means = art["model"], art["impute_means"]

    df = pd.read_parquet("data/derived/raid_splits.parquet")
    hold = df[df.fold == "holdout"]
    hu = hold[hold.model == "human"]
    ai = hold[hold.model != "human"].sample(n=20_000, random_state=97)
    sub = pd.concat([hu, ai]).copy()
    sub["attack"] = "none"
    sub["repetition_penalty"] = "none"

    cache = np.load("data/derived/holdout_features.npz")
    dev = np.load("data/derived/full_features.npz")
    dnames = list(dev["feature_names"])
    cols = [dnames.index(n) for n in cache["feature_names"]]
    m2 = np.nan_to_num(np.nanmean(dev["X_A"][:, cols], axis=0))
    X = {}
    for key in ("X_hu", "X_ai"):
        M = cache[key][:, cols].astype(float)
        bad = np.where(~np.isfinite(M))
        M[bad] = np.take(m2, bad[1])
        X[key] = M
    scores = np.concatenate([model.predict_proba(X["X_hu"])[:, 1],
                             model.predict_proba(X["X_ai"])[:, 1]])
    results = [{"id": i, "score": float(s)} for i, s in zip(sub.id, scores)]

    out = run_evaluation(results, sub, target_fpr=0.05, per_domain_tuning=True)
    allrow = [r for r in out["scores"] if r["domain"] == "all" and r["model"] == "all"
              and r["attack"] == "all"]
    for r in allrow:
        acc = r["accuracy"].get("0.05")
        print(f"ALL: AUROC {r['auroc']:.4f} | acc@5%FPR "
              f"{acc['accuracy'] if acc else 'n/a'} (tp={acc['tp'] if acc else '?'} "
              f"fn={acc['fn'] if acc else '?'})")
    print("\nper-domain (accuracy@5%FPR | AUROC):")
    for r in out["scores"]:
        if r["domain"] not in ("all",) and r["model"] == "all" and r["attack"] == "all":
            acc = r["accuracy"].get("0.05")
            a = f"{acc['accuracy']:.3f}" if acc else "n/a"
            print(f"  {r['domain']:<12} {a} | {r['auroc']:.4f}")


if __name__ == "__main__":
    main()

"""Per-attack preview on the RAID TRAIN adversarial slices (modest sample).

Scores ~400 rows per attack with the 253-panel bundle, reports per-attack
accuracy @5%FPR-style read (threshold from the clean-slice humans).
"""
import numpy as np
import pandas as pd
from raid.utils import load_data

from ai_text_detection import pipeline

SAMPLE = 400


def main() -> None:
    train = load_data(split="train", include_adversarial=True)
    print(f"train rows: {len(train)}")
    print("columns:", list(train.columns), flush=True)
    key = "attack" if "attack" in train.columns else None
    if key is None:
        for c in train.columns:
            if train[c].astype(str).str.contains("paraphrase|homoglyph").any():
                key = c
                break
    print("attack column:", key, flush=True)
    print(train[key].value_counts().to_string(), flush=True)

    art = pipeline.load_artifacts()
    model, means = art["model"], art["impute_means"]

    def score(df):
        X = pipeline.featurize_batch([str(t) for t in df.generation], art)
        return model.predict_proba(pipeline.impute(X, means))[:, 1]

    # clean-slice human threshold (5% FPR): from train_none humans
    clean = train[train[key] == "none"] if "none" in set(train[key]) else None
    if clean is not None:
        hu = clean[clean.model == "human"].sample(min(SAMPLE, 2000), random_state=1)
        thr = np.quantile(score(hu), 0.95)
        print(f"\nclean-slice 5%FPR threshold: {thr:.4f} (n_hu={len(hu)})", flush=True)

    print(f"\n{'attack':<22} {'n':>5} {'acc@thr':>8} {'ai_caught':>9}", flush=True)
    for atk, sub in train.groupby(key):
        if atk == "none":
            continue
        s = sub.sample(min(SAMPLE, len(sub)), random_state=1)
        scores = score(s)
        is_ai = (s.model != "human").to_numpy()
        caught = scores[is_ai] > thr
        acc = ((scores > thr) == is_ai).mean()
        print(f"{atk:<22} {len(s):>5} {acc:>8.3f} "
              f"{caught.mean() if is_ai.any() else float('nan'):>9.3f}", flush=True)


if __name__ == "__main__":
    main()

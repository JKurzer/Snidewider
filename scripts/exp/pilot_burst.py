"""Pilot: does windowed self-similarity separate human from AI on the dev fold?

All 2,000 dev humans + 2,000 sampled AI docs. Calibrate on half, evaluate on
the other half (by source_id, as always). Single-feature threshold at
FPR ~1e-3. This is a smell test, not the eval harness.
Usage: .venv\\Scripts\\python scripts/pilot_burst.py
"""

import statistics

import pandas as pd

from ai_text_detection import burst

FEATURES = ("mean", "stdev", "min", "frac_near_identical")


def featurize(text: str, metric: str) -> dict[str, float]:
    return burst.burst_features(text, metric=metric)


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    dev = df[df.fold == "dev"]
    humans = dev[dev.model == "human"]
    ai = dev[dev.model != "human"].sample(n=2000, random_state=7)

    for metric in ("ck2", "qgram"):
        rows = []
        for label, frame in (("human", humans), ("ai", ai)):
            for text in frame.generation:
                feats = featurize(str(text), metric)
                feats["label"] = label
                rows.append(feats)
        scores = pd.DataFrame(rows).dropna()

        print(f"\n== metric={metric} ({len(scores)} docs) ==")
        for feat in FEATURES:
            h = scores[scores.label == "human"][feat]
            a = scores[scores.label == "ai"][feat]
            print(f"  {feat:<22} human {h.mean():.3f}±{h.std():.3f}   ai {a.mean():.3f}±{a.std():.3f}")

        # calibrate on half, evaluate on the other half
        cal = scores.iloc[::2]
        ev = scores.iloc[1::2]
        cal_h = cal[cal.label == "human"]
        for feat in ("mean", "frac_near_identical"):
            direction_low = cal_h[feat].mean() > cal[cal.label == "ai"][feat].mean()
            # AI-lower: flag score <= t at human 0.1th pct; AI-higher: flag >= t at 99.9th pct
            t = cal_h[feat].quantile(0.001 if direction_low else 0.999)
            ev_h = ev[ev.label == "human"][feat]
            ev_a = ev[ev.label == "ai"][feat]
            if direction_low:
                fpr = (ev_h <= t).mean()
                tpr = (ev_a <= t).mean()
            else:
                fpr = (ev_h >= t).mean()
                tpr = (ev_a >= t).mean()
            print(f"  {feat:<22} threshold {t:+.4f} -> TPR {tpr:.3f} @ FPR {fpr:.4f}")


if __name__ == "__main__":
    main()

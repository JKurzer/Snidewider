"""Descriptive burst-feature stats across Beemo's bodies.

No direction assumed — we are LOOKING, not testing a hypothesis. Prints
per-body aggregates for each burst feature (ck2 metric, window=20, stride=10).
Usage: .venv\\Scripts\\python scripts/pilot_burst_beemo.py
"""

import pandas as pd

from ai_text_detection import burst

BODY_COLUMNS = ("human_output", "model_output", "human_edits", "llama-3.1-70b_edits", "gpt-4o_edits")
FEATURES = ("mean", "stdev", "min", "max", "frac_near_identical")


def explode_body(df: pd.DataFrame, col: str) -> pd.Series:
    values = df[col].dropna()
    if values.map(lambda v: isinstance(v, (list, tuple))).any():
        values = values.explode()
    return values


def main() -> None:
    df = pd.read_parquet("data/raw/beemo/train.parquet")
    print(f"beemo rows: {len(df)}  categories: {sorted(df.category.unique())}")

    for col in BODY_COLUMNS:
        texts = explode_body(df, col)
        rows = [burst.burst_features(str(t)) for t in texts]
        feats = pd.DataFrame(rows).dropna()
        print(f"\n== {col} ({len(feats)} texts) ==")
        for feat in FEATURES:
            s = feats[feat]
            print(f"  {feat:<22} mean {s.mean():.4f}  sd {s.std():.4f}  q10 {s.quantile(0.1):.4f}  q90 {s.quantile(0.9):.4f}")


if __name__ == "__main__":
    main()

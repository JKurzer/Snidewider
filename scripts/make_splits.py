"""Split raw RAID train_none.csv into dev/holdout folds by source_id.

Protocol (RULES #4): every variant of a source document shares one fold, so
no near-duplicate can leak across the tuning/evaluation boundary. Dev is for
feature work and threshold calibration; holdout is evaluated ONCE at the end.
Split is deterministic: source_ids are sha256-salted and sorted, first
--dev-sources go to dev, the rest to holdout (as large as we can keep it).

Usage: .venv\\Scripts\\python scripts/make_splits.py [--dev-sources 2000]
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/raid/train_none.csv")
OUT = Path("data/derived/raid_splits.parquet")
SALT = "ai-text-detection raid split v1"

KEEP = ["id", "source_id", "model", "decoding", "domain", "title", "generation"]


def fold_order(source_id: str) -> str:
    return hashlib.sha256(f"{SALT}:{source_id}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-sources", type=int, default=2000)
    args = parser.parse_args()

    df = pd.read_csv(RAW, usecols=KEEP)
    df["generation"] = df["generation"].astype("string")

    order = sorted(df["source_id"].unique(), key=fold_order)
    dev_ids = set(order[: args.dev_sources])
    df["fold"] = df["source_id"].isin(dev_ids).map({True: "dev", False: "holdout"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    summary = (
        df.groupby(["fold", "domain", df["model"].eq("human").map({True: "human", False: "ai"})])
        .size()
        .rename("rows")
    )
    print(summary.to_string())
    print(f"\n{OUT} written: {len(df)} rows, {df['source_id'].nunique()} sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

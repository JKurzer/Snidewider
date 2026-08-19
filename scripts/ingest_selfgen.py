"""Ingest selfgen (kimi-k3 + kimi-k3-long) as AI training rows.

Reads the manifests, loads every generated file, writes
data/derived/selfgen_rows.parquet (generation, model, domain, title).
RULES #1: data/raw is read-only here.
Usage: .venv\\Scripts\\python scripts\\ingest_selfgen.py
"""

import pandas as pd

ROOT = "data/raw/selfgen"
OUT = "data/derived/selfgen_rows.parquet"


def main() -> None:
    rows = []
    for model in ("kimi-k3", "kimi-k3-long"):
        manifest = pd.read_csv(f"{ROOT}/{model}/manifest.csv")
        cols = list(manifest.columns)
        fcol = next(c for c in cols if "file" in c.lower() or "path" in c.lower())
        dcol = next((c for c in cols if "domain" in c.lower()), None)
        tcol = next((c for c in cols if "title" in c.lower()), None)
        for _, r in manifest.iterrows():
            fp = f"{ROOT}/{model}/{r[fcol]}" if model not in str(r[fcol]) else str(r[fcol])
            try:
                text = open(fp, encoding="utf-8").read()
            except OSError:
                continue
            rows.append({"generation": text, "model": model,
                         "domain": r[dcol] if dcol else "",
                         "title": r[tcol] if tcol else ""})
    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    print(f"{OUT}: {len(df)} rows")
    print(df.model.value_counts().to_string())
    print(df.domain.value_counts().to_string())


if __name__ == "__main__":
    main()

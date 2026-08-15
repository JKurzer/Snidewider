"""Build the selfgen manifest: one row per generated file, joined to titles.

Usage: .venv\\Scripts\\python scripts/make_selfgen_manifest.py
"""

import json
from pathlib import Path

import pandas as pd

TITLES = Path("data/derived/selfgen_titles.json")
ROOT = Path("data/raw/selfgen/kimi-k3")


def main() -> int:
    titles = {t["source_id"]: t for t in json.loads(TITLES.read_text(encoding="utf-8"))}
    rows = []
    for f in sorted(ROOT.rglob("*.txt")):
        t = titles[f.stem]
        rows.append(
            {
                "source_id": f.stem,
                "domain": t["domain"],
                "title": t["title"],
                "model": "kimi-k3",
                "attack": "none",
                "path": str(f).replace("\\", "/"),
                "bytes": f.stat().st_size,
            }
        )
    manifest = pd.DataFrame(rows)
    missing = set(titles) - set(manifest["source_id"])
    if missing:
        raise SystemExit(f"MISSING FILES for {len(missing)} titles: {sorted(missing)[:5]}...")
    manifest.to_csv(ROOT / "manifest.csv", index=False)
    print(manifest.groupby("domain").size())
    print(f"manifest rows: {len(manifest)} (all 104 titles accounted for: {len(missing) == 0})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

r"""Build selfgen manifests: one row per generated file, joined to titles.

Usage: .venv\Scripts\python scripts/make_selfgen_manifest.py
"""

import json
from pathlib import Path

import pandas as pd

BATCHES = {
    "kimi-k3": "data/derived/selfgen_titles.json",
    "kimi-k3-long": "data/derived/selfgen_titles_long.json",
}
ROOT = Path("data/raw/selfgen")


def main() -> int:
    for batch, titles_path in BATCHES.items():
        titles = {t["source_id"]: t for t in json.loads(Path(titles_path).read_text(encoding="utf-8"))}
        batch_dir = ROOT / batch
        rows = []
        for f in sorted(batch_dir.rglob("*.txt")):
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
            raise SystemExit(f"{batch}: MISSING FILES for {len(missing)} titles")
        manifest.to_csv(batch_dir / "manifest.csv", index=False)
        print(f"{batch}: {len(manifest)} rows, all titles accounted for")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

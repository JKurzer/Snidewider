"""Build the frozen gate spec: select union components on A+B, verify once on C.

Replicates the G3 selection: grid over (run x levels) x (window x samples),
stats over each series, eligible = SHORT coverage >= 0.80 & auroc >= 0.52,
top-6 by zero-FPR TPR deduped by (map, stat). Thresholds frozen strictly
above the A+B SHORT human max of each oriented component score.

Output: data/derived/gate_spec.json + honest C-bucket fire rates.
Usage: .venv\\Scripts\\python scripts/build_gate.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_text_detection import gate
from ai_text_detection.metrics import auroc, zero_fpr_tpr

OUT = Path("data/derived/gate_spec.json")
MAP_GRID = [(run, lev) for run in (2, 3, 4) for lev in (2, 4)]
BURST_GRID = [(w, s) for w in (4, 8, 16) for s in (8, 16)]
STATS = ("mean", "stdev", "min", "max", "frac")


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    dev = df[df.fold == "dev"]
    sample = pd.concat(
        [dev[dev.model == "human"], dev[dev.model != "human"].sample(n=4000, random_state=52)]
    ).copy()
    sample["wc"] = sample.generation.astype(str).str.split().str.len()
    sample["label"] = (sample.model != "human").astype(int)

    sources = np.asarray(dev.source_id.unique())  # fleet partition, salt=41
    rng = np.random.RandomState(41)
    rng.shuffle(sources)
    n = len(sources)
    owner = {}
    for name, ids in (
        ("A", sources[: n // 2]),
        ("B", sources[n // 2 : 3 * n // 4]),
        ("C", sources[3 * n // 4 :]),
    ):
        owner.update({s: name for s in ids})
    sample["bucket"] = sample.source_id.map(owner)

    ab = sample[sample.bucket.isin(("A", "B"))]
    ab_short = ab[ab.wc <= 300]
    ab_short_hu = ab_short[ab_short.label == 0]
    ab_short_ai = ab_short[ab_short.label == 1]
    print(f"A+B SHORT: {len(ab_short_hu)} human, {len(ab_short_ai)} ai")

    # sweep: for each (map, burst, stat): coverage + AUROC + zero-FPR TPR on A+B SHORT
    rows = []
    for run, levels in MAP_GRID:
        streams_hu = [gate.dct_run_map_short(str(t), run, levels) for t in ab_short_hu.generation]
        streams_ai = [gate.dct_run_map_short(str(t), run, levels) for t in ab_short_ai.generation]
        for w, s in BURST_GRID:
            for stat in STATS:
                for mode in ("step", "rand"):
                    def score(stream, w=w, s=s, mode=mode, stat=stat):
                        series = (
                            gate.step_series(stream, w)
                            if mode == "step"
                            else gate.random_series(stream, w, s, w)
                        )
                        return gate.series_stats(series)[stat]

                    hu_s = [score(x) for x in streams_hu]
                    ai_s = [score(x) for x in streams_ai]
                    cov_hu = np.isfinite(hu_s).mean()
                    cov_ai = np.isfinite(ai_s).mean()
                    hu_f = [x for x in hu_s if np.isfinite(x)]
                    ai_f = [x for x in ai_s if np.isfinite(x)]
                    if len(hu_f) < 30 or len(ai_f) < 30:
                        continue
                    for direction, aa, hh in (("low", [-x for x in ai_f], [-x for x in hu_f]), ("high", ai_f, hu_f)):
                        roc = auroc(aa, hh)
                        z = zero_fpr_tpr(aa, hh)
                        rows.append(
                            {
                                "run": run, "levels": levels, "window": w, "samples": s,
                                "mode": mode, "stat": stat, "direction": direction,
                                "coverage": (cov_hu + cov_ai) / 2, "auroc": max(roc, 1 - roc),
                                "zfpr_tpr": z["tpr"], "hu_max": max(hh),
                            }
                        )
        print(f"  map run={run} levels={levels} swept")

    board = pd.DataFrame(rows)
    eligible = board[(board.coverage >= 0.80) & (board.auroc >= 0.52)]
    top = (
        eligible.sort_values("zfpr_tpr", ascending=False)
        .drop_duplicates(subset=["run", "levels", "stat"])
        .head(6)
    )
    print("\nunion components:")
    print(top.to_string(index=False))

    components = []
    for _, r in top.iterrows():
        components.append(
            {
                "name": f"r{r.run}_l{r.levels}_{r.mode}_w{r.window}_{r.stat}_{r.direction}",
                "run": int(r.run), "levels": int(r.levels), "window": int(r.window),
                "samples": int(r.samples), "mode": r["mode"], "stat": r["stat"],
                "direction": r["direction"],
                "threshold": float(np.nextafter(r.hu_max, np.inf)),
            }
        )
    spec = {
        "meta": {"built_from": "raid dev A+B SHORT cohort", "fleet": "g1-g3 consolidation"},
        "components": components,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    # verify ONCE on C
    g = gate.Gate(spec)
    c = sample[sample.bucket == "C"]
    c_short = c[c.wc <= 300]
    fires_hu = sum(g.flag(str(t)) for t in c_short[c_short.label == 0].generation)
    fires_ai = sum(g.flag(str(t)) for t in c_short[c_short.label == 1].generation)
    n_hu, n_ai = len(c_short[c_short.label == 0]), len(c_short[c_short.label == 1])
    print(f"\nC SHORT verify: gate fires {fires_ai}/{n_ai} AI (TPR {fires_ai/n_ai:.3f}) vs "
          f"{fires_hu}/{n_hu} human (FPR {fires_hu/n_hu:.4f})")
    print(f"{OUT} written")


if __name__ == "__main__":
    main()

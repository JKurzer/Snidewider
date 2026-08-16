"""THE FINAL EXAM — holdout fold, contacted once.

Frozen protocol (RULES #3/#4, zero leakage):
  A trains the 4 family detectors (HGB each)
  B trains the HGB meta on detector scores
  C sets the threshold (just above the C-human max => target FPR <= 1e-3)
  HOLDOUT (11,371 human + 20K AI sample) is measured once: achieved FPR,
  TPR w/ Wilson CI, AUROC, per-domain TPR. Plus the frozen gate's fire rates.

Output: docs/final_exam.md + console. Usage: .venv\\Scripts\\python scripts/final_exam.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection import qgram
from ai_text_detection.dct_shapes import dct_tail_vector
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.exemplar import ExemplarBank, exemplar_vector
from ai_text_detection.feature_sets import qgram12_vector, relative_vector
from ai_text_detection.gate import Gate
from ai_text_detection.metrics import auroc, wilson_ci

N_BANK = 150
HOLDOUT_AI_SAMPLE = 20_000
OUT = "docs/final_exam.md"


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    labels = {name: (sub.model != "human").to_numpy(int) for name, sub in buckets.items()}

    a = buckets["A"]
    bank_ai = ExemplarBank.from_texts([str(t) for t in a[a.model != "human"].generation[:N_BANK]])
    bank_hu = ExemplarBank.from_texts([str(t) for t in a[a.model == "human"].generation[:N_BANK]])

    def featurize(text: str, ai_self=None, hu_self=None) -> list[float]:
        return (
            relative_vector(text)
            + qgram12_vector(text)
            + exemplar_vector(qgram.profile(text.encode("utf-8"), 3), bank_ai, bank_hu,
                              ai_self, hu_self)
            + dct_tail_vector(text)
        )

    print("featurizing A/B/C (fresh, for model training)...", flush=True)
    fams = {"relative-burst": (0, 8), "qgram12": (8, 20), "exemplar": (20, 31), "dct-nobase": (31, 81)}
    X = {}
    for name, sub in buckets.items():
        ai_self = hu_self = [None] * len(sub)
        if name == "A":  # leave-one-out for rows inside the exemplar banks
            from ai_text_detection.exemplar import bank_self_indices
            ai_self, hu_self = bank_self_indices([str(m) for m in sub.model], N_BANK)
        rows = np.array([featurize(str(t), ai_self[i], hu_self[i])
                         for i, t in enumerate(sub.generation)], dtype=float)
        col_means = np.nanmean(rows, axis=0)
        bad = np.where(~np.isfinite(rows))
        rows[bad] = np.take(col_means, bad[1])
        X[name] = rows
        print(f"  {name} done", flush=True)

    detectors = {}
    for fam, (lo, hi) in fams.items():
        model = HistGradientBoostingClassifier(random_state=7).fit(X["A"][:, lo:hi], labels["A"])
        detectors[fam] = model
        print(f"  detector {fam} trained", flush=True)

    def panel_scores(bucket: str) -> np.ndarray:
        return np.column_stack(
            [detectors[fam].predict_proba(X[bucket][:, lo:hi])[:, 1] for fam, (lo, hi) in fams.items()]
        )

    Za = panel_scores("A")
    Zb = panel_scores("B")
    meta = HistGradientBoostingClassifier(random_state=7).fit(Zb, labels["B"])
    Zc = panel_scores("C")
    # threshold from C-humans' META scores, just above their max (target FPR <= 1e-3)
    c_human_meta = meta.predict_proba(Zc)[:, 1][labels["C"] == 0]
    threshold = np.nextafter(c_human_meta.max(), np.inf)
    print(f"threshold (C-human max + eps): {threshold:.6f}", flush=True)

    # ---------------- holdout, contacted once ----------------
    hold = df[df.fold == "holdout"]
    hold_hu = hold[hold.model == "human"]
    hold_ai = hold[hold.model != "human"].sample(n=HOLDOUT_AI_SAMPLE, random_state=97)
    print(f"holdout: {len(hold_hu)} human, {len(hold_ai)} ai — featurizing...", flush=True)

    gate = Gate.load()
    results = {}
    for name, sub in (("human", hold_hu), ("ai", hold_ai)):
        scores, gate_fires, domains = [], [], []
        for text, domain in zip(sub.generation, sub.domain):
            feats = featurize(str(text))
            row = np.array(feats, dtype=float).reshape(1, -1)
            col_means = np.nanmean(X["A"], axis=0)
            bad = np.where(~np.isfinite(row))
            row[bad] = np.take(col_means, bad[1])
            det_scores = [detectors[fam].predict_proba(row[:, lo:hi])[:, 1] for fam, (lo, hi) in fams.items()]
            scores.append(meta.predict_proba(np.array(det_scores).reshape(1, -1))[:, 1][0])
            gate_fires.append(gate.flag(str(text)))
            domains.append(domain)
        results[name] = (np.array(scores), np.array(gate_fires), np.array(domains))
        print(f"  holdout {name} scored", flush=True)

    s_hu, fire_hu, _ = results["human"]
    s_ai, fire_ai, dom_ai = results["ai"]
    fp = int((s_hu >= threshold).sum())
    tp = int((s_ai >= threshold).sum())
    fpr = fp / len(s_hu)
    tpr = tp / len(s_ai)
    lo, hi = wilson_ci(tp, len(s_ai))
    roc = auroc(list(s_ai), list(s_hu))

    lines = []
    lines.append("# FINAL EXAM — holdout fold, contacted once\n")
    lines.append("Protocol: A trains detectors, B trains meta, C sets threshold, holdout answers once.\n")
    lines.append(f"Threshold (C-human max + eps): {threshold:.6f}\n")
    lines.append(f"Holdout: {len(s_hu)} human, {len(s_ai)} AI (sampled from {386614})\n")
    lines.append(f"\n## Headline\n")
    lines.append(f"- AUROC: **{roc:.4f}**\n")
    lines.append(f"- achieved FPR: **{fpr:.5f}** ({fp}/{len(s_hu)}; target <= 0.001)\n")
    lines.append(f"- TPR at that threshold: **{tpr:.4f}** [{lo:.4f}, {hi:.4f}]\n")
    lines.append(f"- gate: AI fire rate {fire_ai.mean():.4f}, human fire rate {fire_hu.mean():.5f}\n")
    lines.append("\n## Per-domain TPR\n")
    for dom in sorted(set(dom_ai)):
        mask = dom_ai == dom
        n = int(mask.sum())
        lines.append(f"- {dom}: {int((s_ai[mask] >= threshold).sum())}/{n} = {(s_ai[mask] >= threshold).mean():.3f}\n")

    text = "".join(lines)
    print("\n" + text)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"{OUT} written")


if __name__ == "__main__":
    main()

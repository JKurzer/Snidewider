"""Wire Hanada-search hit features into the panel as a fifth detector.

Banks (150 AI + 150 human chunks) from bucket A; bank-source docs excluded
from A's detector training (self-hits are trivially 1). Stack on B, read once
on C. Usage: .venv\\Scripts\\python scripts/exp_hits.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.hits import HIT_FEATURE_NAMES, ChunkBank, hit_features
from ai_text_detection.metrics import auroc, tpr_at_fpr

DETS = ("relative-burst", "qgram12", "exemplar", "dct-nobase")
N_BANK = 150


def main() -> None:
    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    cached = np.load("data/derived/base_scores.npz")

    a = buckets["A"]
    ai_texts = [str(t) for t in a[a.model != "human"].generation]
    hu_texts = [str(t) for t in a[a.model == "human"].generation]
    # pre-filter to bank-eligible texts so bank rows align exactly
    ai_elig = [i for i, t in enumerate(ai_texts) if len(t.encode("utf-8")) >= 150]
    hu_elig = [i for i, t in enumerate(hu_texts) if len(t.encode("utf-8")) >= 150]
    ai_bank = ChunkBank.from_texts([ai_texts[i] for i in ai_elig[:N_BANK]])
    hu_bank = ChunkBank.from_texts([hu_texts[i] for i in hu_elig[:N_BANK]])

    feats = {}
    labels = {}
    for name, sub in buckets.items():
        labels[name] = (sub.model != "human").to_numpy(int)
        rows = []
        for text in sub.generation:
            rows.append([hit_features(str(text).encode("utf-8"), ai_bank, hu_bank)[k] for k in HIT_FEATURE_NAMES])
        feats[name] = np.array(rows)
        print(f"  {name} featurized")

    # bank-source docs self-hit by construction: exclude them from A training
    a_ai_rows = np.flatnonzero(labels["A"] == 1)
    a_hu_rows = np.flatnonzero(labels["A"] == 0)
    bank_rows = {int(a_ai_rows[i]) for i in ai_elig[:N_BANK]} | {
        int(a_hu_rows[i]) for i in hu_elig[:N_BANK]
    }
    keep = np.array([i not in bank_rows for i in range(len(labels["A"]))])

    hits_model = HistGradientBoostingClassifier(random_state=7).fit(
        feats["A"][keep], labels["A"][keep]
    )
    hits_c = hits_model.predict_proba(feats["C"])[:, 1]
    roc = auroc(list(hits_c[labels["C"] == 1]), list(hits_c[labels["C"] == 0]))
    res = tpr_at_fpr(list(hits_c[labels["C"] == 1]), list(hits_c[labels["C"] == 0]))
    print(f"\nhits solo on C: AUROC {roc:.3f} | TPR@1e-3 {res['tpr']:.3f} [{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]")

    hits_b = hits_model.predict_proba(feats["B"])[:, 1]
    Zb = np.column_stack([cached[f"{d}_B"] for d in DETS] + [hits_b])
    Zc = np.column_stack([cached[f"{d}_C"] for d in DETS] + [hits_c])
    for name in ("hgb",):
        meta = HistGradientBoostingClassifier(random_state=7).fit(Zb, labels["B"])
        s = meta.predict_proba(Zc)[:, 1]
        roc = auroc(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
        res = tpr_at_fpr(list(s[labels["C"] == 1]), list(s[labels["C"] == 0]))
        print(f"5-detector stack ({name}): AUROC {roc:.3f} | TPR@1e-3 {res['tpr']:.3f} [{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]")
    print("\n(4-detector reference: hgb AUROC 0.939 | TPR@1e-3 0.241)")


if __name__ == "__main__":
    main()

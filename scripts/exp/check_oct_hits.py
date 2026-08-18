"""Hand-verify oct_hits_features, then solo-bench it on B/C."""
import numpy as np
import pandas as pd
from fleet_qgmid import eval_feat

from ai_text_detection.evaldata import split_buckets
from ai_text_detection.token_bigrams import oct_hits_features

# hand-checks
r = oct_hits_features("a b c d e f g h i j k l m n o p q")
print(f"no-repeat:        hits {r['oct_hits']:.0f} (expect 0)")

# 'the the': token0 sees 'the' in succeeding oct (1), token1 sees it in
# preceding oct (1) -> cumulative 2
r = oct_hits_features("the the quick brown fox jumps over the lazy dog today "
                      "somewhere nearby right now exactly yes indeed ok fine")
print(f"one repeat pair:  hits {r['oct_hits']:.0f} (expect >=2)")

# 'x x x': t0 sees 2 ahead, t1 sees 1 back + 1 ahead, t2 sees 2 back -> 6
r = oct_hits_features("x x x " + " ".join(f"w{i}" for i in range(20)))
print(f"triple run:       hits {r['oct_hits']:.0f} (expect 6)")

# repeat beyond the octgram span -> not counted
r = oct_hits_features("dog " + " ".join(f"w{i}" for i in range(9)) + " dog "
                      + " ".join(f"v{i}" for i in range(10)))
print(f"9-apart repeat:   hits {r['oct_hits']:.0f} (expect 0)")

# solo bench
df = pd.read_parquet("data/derived/raid_splits.parquet")
buckets = split_buckets(df)
for b in "BC":
    sub = buckets[b]
    y = (sub.model != "human").to_numpy(int)
    vals = np.array([oct_hits_features(str(t))["oct_hits_rate"] for t in sub.generation])
    name, roc, tail = eval_feat(vals, y)
    print(f"oct_hits_rate {b}: AUROC {roc:.3f} TPR@1e-2 {tail:.3f}")
    vals = np.array([oct_hits_features(str(t))["oct_hits"] for t in sub.generation])
    name, roc, tail = eval_feat(vals, y)
    print(f"oct_hits      {b}: AUROC {roc:.3f} TPR@1e-2 {tail:.3f}")

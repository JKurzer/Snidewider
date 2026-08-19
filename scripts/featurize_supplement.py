"""Featurize the training supplements (selfgen 208 + adversarial 231K) with
the production pipeline (261 panel). 6-worker pool; writes
data/derived/supp_features.npz (X, y, feature_names, source).
Usage: .venv\\Scripts\\python scripts\\featurize_supplement.py
"""

from __future__ import annotations

import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

N_WORKERS = 6
_CHUNK = 400
_ART = None


def _init_worker():
    global _ART
    from ai_text_detection import pipeline
    _ART = pipeline.load_artifacts()


def _score_chunk(texts):
    global _ART
    from ai_text_detection import pipeline
    return pipeline.featurize_batch(texts, _ART)


def main() -> None:
    sg = pd.read_parquet("data/derived/selfgen_rows.parquet")
    sg["source"] = "selfgen"
    adv = pd.read_parquet("data/derived/adv_supplement.parquet")
    adv["source"] = "adv_" + adv.attack.astype(str)
    df = pd.concat([sg[["generation", "model", "source"]],
                    adv[["generation", "model", "source"]]], ignore_index=True)
    texts = [str(t) for t in df.generation]
    print(f"featurizing {len(texts)} supplement rows", flush=True)

    chunks = [texts[i:i + _CHUNK] for i in range(0, len(texts), _CHUNK)]
    t0 = time.time()
    with Pool(N_WORKERS, initializer=_init_worker) as pool:
        parts = pool.map(_score_chunk, chunks)
    X = np.vstack(parts)
    dt = time.time() - t0
    print(f"featurized in {dt/60:.1f} min: {X.shape}", flush=True)

    names = list(np.load("data/derived/full_features.npz")["feature_names"])
    y = (df.model != "human").to_numpy(int)
    np.savez("data/derived/supp_features.npz",
             X=X, y=y, feature_names=np.array(names),
             source=df.source.to_numpy())
    print("data/derived/supp_features.npz written", flush=True)


if __name__ == "__main__":
    main()

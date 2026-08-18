"""RAID submission, FULL test set (all domains x generators x decoding x
repetition x adversarial). The bot's warning was explicit: partial coverage
means no aggregate and no main board. Parallel featurize+predict over a
worker pool (bundle loaded once per worker), official run_detection shape.

Usage: .venv\\Scripts\\python scripts\\raid_submission_full.py
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

from raid import run_detection
from raid.utils import load_data

OUT_DIR = "submissions/snidewider"
N_WORKERS = 6
_CHUNK = 800

_ART = None  # per-worker bundle


def _init_worker():
    global _ART
    from ai_text_detection import pipeline
    _ART = pipeline.load_artifacts()


def _score_chunk(texts):
    global _ART
    from ai_text_detection import pipeline
    X = pipeline.featurize_batch(texts, _ART)
    return _ART["model"].predict_proba(pipeline.impute(X, _ART["impute_means"]))[:, 1].tolist()


def parallel_detector(texts):
    chunks = [texts[i:i + _CHUNK] for i in range(0, len(texts), _CHUNK)]
    with Pool(N_WORKERS, initializer=_init_worker) as pool:
        parts = pool.map(_score_chunk, chunks)
    return [s for part in parts for s in part]


def main() -> None:
    test = load_data(split="test", include_adversarial=True)
    print(f"full test set: {len(test)} rows", flush=True)
    print("attacks:", test.attack.value_counts().to_dict() if "attack" in test else "n/a",
          flush=True)

    t0 = time.time()
    preds = run_detection(parallel_detector, test)
    dt = time.time() - t0
    print(f"{len(preds)} predictions in {dt/60:.1f} min", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/predictions.json", "w", encoding="utf-8") as fh:
        json.dump(preds, fh)
    print(f"{OUT_DIR}/predictions.json written ({len(preds)} rows)", flush=True)


if __name__ == "__main__":
    main()

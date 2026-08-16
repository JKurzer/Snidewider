"""D2 sampling sweep: how should the DCT encoder draw segments from a doc?

Variants:
  sent-k2    — sentence split (status quo, reference row)
  sub16-k*   — 16 sentences sampled without replacement, seeded by the doc's
               sha256 (same trick as burst.random_change_series), original
               document order preserved, then the usual feature math
  win{W}-k*  — non-overlapping whitespace-token windows of W in {8..64}

Protocol (RULES #4): DCT detector = HGB(random_state=7) trained on bucket A;
stack = HGB(random_state=7) on bucket B's 4 scores (cached relative-burst /
qgram12 / exemplar + the DCT variant); numbers read ONCE on bucket C.
Cached A scores are ignored (in-sample).

Run (from the main checkout so data/ resolves):
  set PYTHONPATH=..\\ai-text-detection-d2-sampling\\src && .venv\\Scripts\\python ..\\ai-text-detection-d2-sampling\\scripts\\exp_d2.py
"""

import hashlib
import random
import re
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import ai_text_detection
from ai_text_detection import dct
from ai_text_detection.evaldata import split_buckets
from ai_text_detection.metrics import auroc, tpr_at_fpr

WINDOWS = (8, 16, 24, 32, 48, 64)
KS = (2, 4)
SUB_SAMPLES = 16
BASE_DETECTORS = ("relative-burst", "qgram12", "exemplar")

CONFIGS = (
    [("sent", None, 2)]
    + [("sub16", None, k) for k in KS]
    + [("win", w, k) for w in WINDOWS for k in KS]
)


def config_name(mode: str, window: int | None, k: int) -> str:
    if mode == "win":
        return f"win{window}-k{k}"
    return f"{mode}-k{k}"


def subsample_indices(text: str, n_sentences: int, samples: int = SUB_SAMPLES) -> list[int]:
    """Deterministic per-doc sentence pick (content-hash seed, order preserved)."""
    if n_sentences <= samples:
        return list(range(n_sentences))
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
    return sorted(random.Random(seed).sample(range(n_sentences), samples))


def doc_embeddings(text: str):
    """Embed a doc once: sentence matrices + flat token matrix with offsets.

    Token-flat trick: regex-matching each whitespace token separately then
    concatenating equals embed_sentence over the joined window (space is not
    in [A-Za-z0-9'], so matches can't merge across the join).
    """
    sentences = [s for s in re.split(r"[.!?]+", text) if len(s.strip()) > 2]
    sent_mats = [dct.embed_sentence(s) for s in sentences]
    vocab, matrix = dct._load()
    tokens = text.split()
    offsets = np.zeros(len(tokens) + 1, dtype=np.int64)
    indices: list[int] = []
    for i, tok in enumerate(tokens):
        idx = [vocab[t] for t in dct._TOKEN_RE.findall(tok.lower()) if t in vocab]
        indices.extend(idx)
        offsets[i + 1] = offsets[i] + len(idx)
    flat = matrix[indices] if indices else np.zeros((0, matrix.shape[1]), np.float32)
    return sent_mats, tokens, offsets, flat


def window_mats(tokens, offsets, flat, window: int):
    return [
        flat[offsets[i] : offsets[i + window]]
        for i in range(0, len(tokens) - window + 1, window)
    ]


def featurize(buckets: dict[str, pd.DataFrame]) -> dict[str, dict[str, list[list[float]]]]:
    """One pass over docs; every config's 4 features per visit (O(1) memory)."""
    rows = {name: {config_name(*c): [] for c in CONFIGS} for name in buckets}
    for name, sub in buckets.items():
        t0 = time.perf_counter()
        for text in sub.generation:
            text = str(text)
            sent_mats, tokens, offsets, flat = doc_embeddings(text)
            sub_idx = subsample_indices(text, len(sent_mats))
            for mode, window, k in CONFIGS:
                if mode == "sent":
                    mats = sent_mats
                elif mode == "sub16":
                    mats = [sent_mats[i] for i in sub_idx]
                else:
                    mats = window_mats(tokens, offsets, flat, window)
                feats = dct.features_from_embeddings(mats, k)
                rows[name][config_name(mode, window, k)].append(
                    [feats[f] for f in dct.DCT_FEATURE_NAMES]
                )
        print(f"  bucket {name}: {len(sub)} docs in {time.perf_counter() - t0:.1f}s")
    return rows


def impute_like_a(Xa: np.ndarray, X: np.ndarray) -> np.ndarray:
    col_means = np.nanmean(Xa, axis=0)
    bad = np.where(~np.isfinite(X))
    X = X.copy()
    X[bad] = np.take(col_means, bad[1])
    return X


def main() -> None:
    print("module:", ai_text_detection.__file__)
    assert "d2-sampling" in ai_text_detection.__file__, "WRONG WORKTREE"

    df = pd.read_parquet("data/derived/raid_splits.parquet")
    buckets = split_buckets(df)
    store = np.load("data/derived/base_scores.npz")
    labels = {}
    for name, sub in buckets.items():
        labels[name] = (sub.model != "human").to_numpy(int)
        assert np.array_equal(labels[name], store[f"labels_{name}"])
        print(f"bucket {name}: {len(sub)} docs ({(sub.model == 'human').sum()} human)")

    print("featurizing (one pass, all configs)...")
    rows = featurize(buckets)

    def report(tag: str, scores: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
        ai, hu = list(scores[y == 1]), list(scores[y == 0])
        roc = auroc(ai, hu)
        res = tpr_at_fpr(ai, hu)
        print(
            f"  {tag:<28} AUROC {roc:.3f} | TPR@1e-3 {res['tpr']:.3f} "
            f"[{res['tpr_lo']:.3f}, {res['tpr_hi']:.3f}]"
        )
        return roc, res

    # Reference: the 3-detector stack without any DCT (the bar to beat).
    print("\n== reference: 3-detector stack (cached scores only) ==")
    Zb = np.column_stack([store[f"{d}_B"] for d in BASE_DETECTORS])
    Zc = np.column_stack([store[f"{d}_C"] for d in BASE_DETECTORS])
    meta = HistGradientBoostingClassifier(random_state=7).fit(Zb, labels["B"])
    report("3-det stack", meta.predict_proba(Zc)[:, 1], labels["C"])

    print("\n== sweep: DCT variant standalone on C | 4-det stack on C ==")
    results = []
    for cfg in CONFIGS:
        name = config_name(*cfg)
        Xa = np.array(rows["A"][name], dtype=float)
        Xa_imp = impute_like_a(Xa, Xa)
        model = HistGradientBoostingClassifier(random_state=7).fit(Xa_imp, labels["A"])
        scores = {}
        for bucket in ("B", "C"):
            X = np.array(rows[bucket][name], dtype=float)
            scores[bucket] = model.predict_proba(impute_like_a(Xa, X))[:, 1]
        print(f"-- {name}")
        dct_roc, dct_res = report("dct standalone", scores["C"], labels["C"])
        Zb4 = np.column_stack([Zb, scores["B"]])
        Zc4 = np.column_stack([Zc, scores["C"]])
        meta4 = HistGradientBoostingClassifier(random_state=7).fit(Zb4, labels["B"])
        stk_roc, stk_res = report("4-det stack", meta4.predict_proba(Zc4)[:, 1], labels["C"])
        results.append((name, dct_roc, dct_res["tpr"], stk_roc, stk_res["tpr"]))

    print("\n== summary (C bucket; target: stack TPR@1e-3 > 0.170, AUROC ~0.896) ==")
    print("| config | dct AUROC | dct TPR | stack AUROC | stack TPR |")
    print("|---|---|---|---|---|")
    for name, dr, dt, sr, st in results:
        print(f"| {name} | {dr:.3f} | {dt:.3f} | {sr:.3f} | {st:.3f} |")


if __name__ == "__main__":
    main()

"""Smoke test for _l0learn_native: recover a known 2-sparse logistic signal."""
import sys

sys.path.insert(0, "src")

import numpy as np

from ai_text_detection import _l0learn_native as L

print("import OK")

rng = np.random.default_rng(0)
n, p = 500, 30
X = rng.normal(size=(n, p))
beta = np.zeros(p)
beta[3] = 2.0
beta[17] = -1.5
y = (rng.random(n) < 1 / (1 + np.exp(-(X @ beta)))).astype(float)

penalty = sys.argv[1] if len(sys.argv) > 1 else "L0"
n_lambda = int(sys.argv[2]) if len(sys.argv) > 2 else 20
max_nnz = int(sys.argv[3]) if len(sys.argv) > 3 else 10
print(f"fitting penalty={penalty} n_lambda={n_lambda} max_nnz={max_nnz}", flush=True)
out = L.fit(np.asfortranarray(X), y, penalty=penalty, n_lambda=n_lambda, max_nnz=max_nnz)
print("fit returned", flush=True)
B = np.asarray(out["betas"][0])
nnz_row0 = list(out["nnz"])[: B.shape[1]]  # row 0 of the gamma grid
print("nnz path (row 0):", nnz_row0[:12], "| NaN:", np.isnan(B).any())
k = min(range(len(nnz_row0)), key=lambda i: abs(nnz_row0[i] - 2))
top = np.argsort(-np.abs(B[:, k]))[:4]
print("top feats @~2nnz:", top.tolist(), "(truth: 3, 17)")

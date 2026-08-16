"""Diagnostic: dump lambda path + NaN check for the pure-L0 fit."""
import sys

sys.path.insert(0, "src")

import numpy as np

from ai_text_detection import _l0learn_native as L

rng = np.random.default_rng(0)
n, p = 500, 30
X = rng.normal(size=(n, p))
beta = np.zeros(p)
beta[3] = 2.0
beta[17] = -1.5
y = (rng.random(n) < 1 / (1 + np.exp(-(X @ beta)))).astype(float)

out = L.fit(np.asfortranarray(X), y, penalty="L0", n_lambda=20, max_nnz=10)
lam = np.asarray(out["lambdas"])
nnz = np.asarray(out["nnz"])
B = np.asarray(out["betas"][0])
conv = np.asarray(out["converged"])
print("lambdas:", lam)
print("nnz:    ", nnz)
print("converged:", conv)
print("NaN in B:", np.isnan(B).any(), "| inf in B:", np.isinf(B).any())
print("col0 (lambda=max) nnz:", np.count_nonzero(B[:, 0]), "max|B|:", np.abs(B[:, 0]).max())

# reference: what SHOULD lambdamax be? y in +-1, gradient of logistic loss at 0:
yv = np.where(y > 0, 1.0, -1.0)
ytXmax = 0.5 * np.abs(yv @ X).max()
lipconst = 0.25
print("expected lambdamax ~", (ytXmax**2) / (2 * lipconst))

"""Ridge gating network shared by the classification and regression inner poolers.

The gate maps preprocessed features to per-sample head weights: one Ridge regressor per
head predicts that head's share of an oracle target (softmax of per-sample
log-likelihoods), and a softmax across heads normalises the projection at inference.
Only the oracle differs by task -- Bernoulli likelihood for classification, Gaussian for
regression -- so everything else lives here.

The weight computation is numpy-only; fitting needs scikit-learn.
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.linear_model import Ridge

    _FIT_DEPS_AVAILABLE = True
except ImportError:
    Ridge = None  # type: ignore[assignment,misc]
    _FIT_DEPS_AVAILABLE = False


def _heuristic_alpha(X: np.ndarray) -> float:
    """alpha = mean_feature_variance * p / n, floored at 0.01."""
    return max(0.01, float(np.mean(np.var(X, axis=0)) * X.shape[1] / X.shape[0]))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fit_gate(X_prep: np.ndarray, target_w: np.ndarray):
    """Fit one Ridge per head on the oracle weights.

    Parameters
    ----------
    X_prep : ndarray, shape (n, p)
    target_w : ndarray, shape (n, n_heads)
        Oracle per-sample head weights, rows summing to 1.

    Returns
    -------
    coef : ndarray, shape (n_heads, p)
    intercept : ndarray, shape (n_heads,)
    """
    n_heads = target_w.shape[1]
    alpha = _heuristic_alpha(X_prep)
    coef = np.zeros((n_heads, X_prep.shape[1]), dtype=float)
    intercept = np.zeros(n_heads, dtype=float)
    for j in range(n_heads):
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X_prep, target_w[:, j])
        coef[j] = ridge.coef_
        intercept[j] = float(ridge.intercept_)
    return coef, intercept


def gate_weights(X_prep: np.ndarray, coef, intercept, n_heads: int) -> np.ndarray:
    """Per-sample head weights ``(n_samples, n_heads)``; uniform when no gate was fitted."""
    if coef is None:
        return np.full((len(X_prep), n_heads), 1.0 / n_heads)
    return _softmax(X_prep @ coef.T + intercept)

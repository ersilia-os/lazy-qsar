"""Dataset statistics shared by the head and preprocessor profilers."""

from __future__ import annotations

import numpy as np


def estimate_feature_signal(
    X, y: np.ndarray, n_sample: int = 5000, p_sample: int = 500
):
    """Mean and 90th percentile of |Pearson r| between sampled features and y.

    Up to 5,000 rows and 500 columns, drawn with a fixed seed. Constant columns, and
    a constant y, contribute nothing.
    """
    n, p = X.shape
    n_s = min(n_sample, n)
    rng = np.random.RandomState(42)
    row_idx = rng.choice(n, n_s, replace=False) if n > n_s else np.arange(n_s)
    # Same draws in the same order as always; the columns are selected before the cast
    # to float64 rather than after, which copied the whole matrix to keep 500 columns.
    col_idx = rng.choice(p, p_sample, replace=False) if p > p_sample else None
    X_s = X[row_idx] if hasattr(X, "toarray") else np.asarray(X)[row_idx]
    if col_idx is not None:
        X_s = X_s[:, col_idx]
    X_s = X_s.toarray() if hasattr(X_s, "toarray") else X_s
    X_s = np.asarray(X_s).astype(float)
    y_s = y[row_idx].astype(float)
    x_std = X_s.std(axis=0)
    X_s = X_s[:, x_std > 0]
    y_std = float(y_s.std())
    if X_s.shape[1] == 0 or y_std == 0.0:
        return 0.0, 0.0
    X_c = X_s - X_s.mean(axis=0)
    y_c = y_s - y_s.mean()
    cov = (X_c * y_c[:, None]).mean(axis=0)
    x_stds = X_c.std(axis=0)
    mask = x_stds > 0
    corrs = np.clip(np.abs(cov[mask] / (x_stds[mask] * y_std)), 0.0, 1.0)
    if corrs.size == 0:
        return 0.0, 0.0
    return float(corrs.mean()), float(np.percentile(corrs, 90))

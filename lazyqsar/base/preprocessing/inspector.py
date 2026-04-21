"""
Dataset profiling for base preprocessing pipeline selection.

Computes all statistics about X and y needed to choose a scaler and
dimensionality reducer without any search or cross-validation.
"""

from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class PreprocessingProfile:
    # Shape
    n_samples: int
    n_features: int
    n_p_ratio: float            # n_samples / n_features

    # Feature type composition
    sparsity: float                    # fraction of zeros in X (0.0–1.0)
    is_sparse_counts: bool             # fingerprint-like data (integer, sparse, small values)
    binary_feature_fraction: float     # fraction of features that only take {0, 1} values

    # Distribution characteristics (estimated from a subsample of features)
    median_feature_skewness: float     # median |skewness| across features
    outlier_fraction: float            # fraction of features with >5% IQR outliers
    near_zero_variance_fraction: float # fraction of features with variance < 1e-6

    # Redundancy
    median_abs_correlation: float      # median |Pearson r| across sampled feature pairs

    # Signal
    feature_signal_p90: float          # 90th-percentile |Pearson r| with target

    # Task
    task: str                          # "classification" or "regression"

    # Target characteristics (regression)
    y_skewness: float = 0.0
    y_all_positive: bool = False

    # Target characteristics (classification)
    n_minority_class: int = 0   # min(n_positives, n_negatives); 0 for regression

    def __repr__(self):
        lines = [
            "PreprocessingProfile(",
            f"  n_samples={self.n_samples}, n_features={self.n_features}, n_p_ratio={self.n_p_ratio:.2f}",
            f"  sparsity={self.sparsity:.3f}, is_sparse_counts={self.is_sparse_counts}",
            f"  binary_feature_fraction={self.binary_feature_fraction:.3f}",
            f"  median_feature_skewness={self.median_feature_skewness:.3f}, "
            f"outlier_fraction={self.outlier_fraction:.3f}",
            f"  near_zero_variance_fraction={self.near_zero_variance_fraction:.3f}, "
            f"median_abs_correlation={self.median_abs_correlation:.3f}",
            f"  feature_signal_p90={self.feature_signal_p90:.3f}",
            f"  task={self.task!r}",
        ]
        if self.task == "regression":
            lines.append(f"  y_skewness={self.y_skewness:.3f}, y_all_positive={self.y_all_positive}")
        if self.task == "classification":
            lines.append(f"  n_minority_class={self.n_minority_class}")
        lines.append(")")
        return "\n".join(lines)


def _compute_sparsity(X) -> float:
    if hasattr(X, "nnz"):
        n_total = X.shape[0] * X.shape[1]
        return 1.0 - X.nnz / n_total
    arr = np.asarray(X)
    return float((arr == 0).mean())


def _detect_sparse_counts(X, sparsity: float) -> bool:
    if sparsity < 0.5:
        return False
    n_sample = min(5000, X.shape[0])
    if hasattr(X, "toarray"):
        sample = X[:n_sample].toarray()
    else:
        sample = np.asarray(X[:n_sample])
    nonzero_vals = sample[sample != 0]
    if nonzero_vals.size == 0:
        return False
    is_integer_like = float((nonzero_vals == np.floor(nonzero_vals)).mean()) > 0.95
    if not is_integer_like:
        return False
    median_val = float(np.median(nonzero_vals))
    return sparsity >= 0.85 or median_val <= 5


def _compute_binary_feature_fraction(X, n_sample: int = 5000) -> float:
    n_s = min(n_sample, X.shape[0])
    if hasattr(X, "toarray"):
        sample = X[:n_s].toarray()
    else:
        sample = np.asarray(X[:n_s])
    is_binary = ((sample == 0) | (sample == 1)).all(axis=0)
    return float(is_binary.mean())


def _estimate_feature_signal(X, y: np.ndarray, n_sample: int = 5000,
                              p_sample: int = 500):
    n, p = X.shape
    n_s = min(n_sample, n)
    rng = np.random.RandomState(42)
    row_idx = rng.choice(n, n_s, replace=False) if n > n_s else np.arange(n_s)
    if hasattr(X, "toarray"):
        X_s = X[row_idx].toarray().astype(float)
    else:
        X_s = np.asarray(X)[row_idx].astype(float)
    y_s = y[row_idx].astype(float)
    if p > p_sample:
        col_idx = rng.choice(p, p_sample, replace=False)
        X_s = X_s[:, col_idx]
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


def _compute_median_feature_skewness(X, n_sample: int = 5000,
                                     p_sample: int = 200) -> float:
    rng = np.random.RandomState(42)
    n_s = min(n_sample, X.shape[0])
    p_s = min(p_sample, X.shape[1])
    row_idx = rng.choice(X.shape[0], n_s, replace=False) if X.shape[0] > n_s else np.arange(n_s)
    col_idx = rng.choice(X.shape[1], p_s, replace=False) if X.shape[1] > p_s else np.arange(p_s)
    if hasattr(X, "toarray"):
        Xs = X[row_idx][:, col_idx].toarray().astype(float)
    else:
        Xs = np.asarray(X)[np.ix_(row_idx, col_idx)].astype(float)
    Xs = Xs[:, Xs.std(axis=0) > 0]
    if Xs.shape[1] == 0:
        return 0.0
    skews = np.abs(stats.skew(Xs, axis=0))
    skews = skews[np.isfinite(skews)]
    return float(np.median(skews)) if skews.size > 0 else 0.0


def _compute_outlier_fraction(X, n_sample: int = 5000, p_sample: int = 500) -> float:
    rng = np.random.RandomState(42)
    n_s = min(n_sample, X.shape[0])
    p_s = min(p_sample, X.shape[1])
    row_idx = rng.choice(X.shape[0], n_s, replace=False) if X.shape[0] > n_s else np.arange(n_s)
    col_idx = rng.choice(X.shape[1], p_s, replace=False) if X.shape[1] > p_s else np.arange(p_s)
    if hasattr(X, "toarray"):
        Xs = X[row_idx][:, col_idx].toarray().astype(float)
    else:
        Xs = np.asarray(X)[np.ix_(row_idx, col_idx)].astype(float)
    q1 = np.percentile(Xs, 25, axis=0)
    q3 = np.percentile(Xs, 75, axis=0)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    outlier_mask = (Xs < lo[None, :]) | (Xs > hi[None, :])
    frac_per_feature = outlier_mask.mean(axis=0)
    frac_per_feature[iqr == 0] = 0.0
    return float((frac_per_feature > 0.05).mean())


def _compute_near_zero_variance_fraction(X, n_sample: int = 5000,
                                         threshold: float = 1e-6) -> float:
    n_s = min(n_sample, X.shape[0])
    if hasattr(X, "toarray"):
        sample = X[:n_s].toarray().astype(float)
    else:
        sample = np.asarray(X[:n_s]).astype(float)
    var = sample.var(axis=0)
    return float((var < threshold).mean())


def _compute_median_abs_correlation(X, n_sample: int = 5000,
                                    n_pairs: int = 50) -> float:
    rng = np.random.RandomState(42)
    n_s = min(n_sample, X.shape[0])
    row_idx = rng.choice(X.shape[0], n_s, replace=False) if X.shape[0] > n_s else np.arange(n_s)
    if hasattr(X, "toarray"):
        Xs = X[row_idx].toarray().astype(float)
    else:
        Xs = np.asarray(X)[row_idx].astype(float)
    p = Xs.shape[1]
    if p < 2:
        return 0.0
    n_actual = min(n_pairs, p * (p - 1) // 2)
    i_idx = rng.randint(0, p, size=n_actual)
    j_idx = rng.randint(0, p, size=n_actual)
    same = i_idx == j_idx
    j_idx[same] = (j_idx[same] + 1) % p
    corrs = []
    for i, j in zip(i_idx, j_idx):
        xi, xj = Xs[:, i], Xs[:, j]
        if xi.std() > 0 and xj.std() > 0:
            c = np.corrcoef(xi, xj)[0, 1]
            if np.isfinite(c):
                corrs.append(abs(c))
    return float(np.median(corrs)) if corrs else 0.0


def inspect(X, y, task: str) -> PreprocessingProfile:
    """
    Profile the dataset and return a PreprocessingProfile.

    Parameters
    ----------
    X : array-like or scipy sparse matrix, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    task : str
        "classification" or "regression".

    Returns
    -------
    PreprocessingProfile
    """
    y = np.asarray(y).ravel()
    n_samples, n_features = X.shape

    if task not in ("classification", "regression"):
        raise ValueError(
            f"task must be 'classification' or 'regression', got {task!r}"
        )

    sparsity = _compute_sparsity(X)
    is_sparse_counts = _detect_sparse_counts(X, sparsity)
    binary_feature_fraction = _compute_binary_feature_fraction(X)
    _, feature_signal_p90 = _estimate_feature_signal(X, y)
    median_feature_skewness = _compute_median_feature_skewness(X)
    outlier_fraction = _compute_outlier_fraction(X)
    near_zero_variance_fraction = _compute_near_zero_variance_fraction(X)
    median_abs_correlation = _compute_median_abs_correlation(X)
    n_p_ratio = float(n_samples) / n_features

    y_skewness = 0.0
    y_all_positive = False
    if task == "regression":
        y_skewness = float(stats.skew(y))
        y_all_positive = bool((y > 0).all())

    n_minority_class = 0
    if task == "classification":
        counts = np.bincount(y.astype(int))
        n_minority_class = int(counts.min())

    return PreprocessingProfile(
        n_samples=n_samples,
        n_features=n_features,
        n_p_ratio=n_p_ratio,
        sparsity=sparsity,
        is_sparse_counts=is_sparse_counts,
        binary_feature_fraction=binary_feature_fraction,
        median_feature_skewness=median_feature_skewness,
        outlier_fraction=outlier_fraction,
        near_zero_variance_fraction=near_zero_variance_fraction,
        median_abs_correlation=median_abs_correlation,
        feature_signal_p90=feature_signal_p90,
        task=task,
        y_skewness=y_skewness,
        y_all_positive=y_all_positive,
        n_minority_class=n_minority_class,
    )

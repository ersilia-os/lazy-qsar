"""
Dataset profiling for base XGBoost parameter selection.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class DatasetProfile:
    # Shape
    n_samples: int
    n_features: int
    n_p_ratio: float
    sparsity: float
    is_sparse_counts: bool
    binary_feature_fraction: float
    feature_signal_strength: float
    feature_signal_p90: float
    task: str
    imbalance_ratio: float = 1.0
    y_skewness: float = 0.0
    y_all_positive: bool = False

    def __repr__(self):
        lines = [
            f"DatasetProfile(",
            f"  n_samples={self.n_samples}, n_features={self.n_features}, n_p_ratio={self.n_p_ratio:.2f}",
            f"  sparsity={self.sparsity:.3f}, is_sparse_counts={self.is_sparse_counts}",
            f"  binary_feature_fraction={self.binary_feature_fraction:.3f}, "
            f"feature_signal_strength={self.feature_signal_strength:.3f}",
            f"  task={self.task!r}",
        ]
        if self.task == "classification":
            lines.append(f"  imbalance_ratio={self.imbalance_ratio:.2f}")
        else:
            lines.append(f"  y_skewness={self.y_skewness:.3f}, y_all_positive={self.y_all_positive}")
        lines.append(f"  feature_signal_p90={self.feature_signal_p90:.3f}")
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
    max_val = float(nonzero_vals.max())
    return sparsity >= 0.85 or max_val <= 10


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


def _detect_task(y: np.ndarray) -> str:
    unique = np.unique(y)
    if len(unique) == 2 and set(unique).issubset({0, 1}):
        return "classification"
    return "regression"


def inspect(X, y, task: Optional[str] = None) -> DatasetProfile:
    y = np.asarray(y).ravel()
    n_samples, n_features = X.shape
    if task is None:
        task = _detect_task(y)
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
    sparsity = _compute_sparsity(X)
    is_sparse_counts = _detect_sparse_counts(X, sparsity)
    binary_feature_fraction = _compute_binary_feature_fraction(X)
    feature_signal_strength, feature_signal_p90 = _estimate_feature_signal(X, y)
    n_p_ratio = float(n_samples) / n_features
    if task == "classification":
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) != 2:
            raise ValueError(f"classification requires exactly 2 classes, found {len(unique)}")
        label_counts = dict(zip(unique, counts))
        pos_count = label_counts.get(1, counts.min())
        neg_count = label_counts.get(0, counts.max())
        imbalance_ratio = float(neg_count / pos_count) if pos_count > 0 else 1.0
        return DatasetProfile(
            n_samples=n_samples,
            n_features=n_features,
            n_p_ratio=n_p_ratio,
            sparsity=sparsity,
            is_sparse_counts=is_sparse_counts,
            binary_feature_fraction=binary_feature_fraction,
            feature_signal_strength=feature_signal_strength,
            feature_signal_p90=feature_signal_p90,
            task=task,
            imbalance_ratio=imbalance_ratio,
        )
    else:
        y_skewness = float(stats.skew(y))
        y_all_positive = bool((y > 0).all())
        return DatasetProfile(
            n_samples=n_samples,
            n_features=n_features,
            n_p_ratio=n_p_ratio,
            sparsity=sparsity,
            is_sparse_counts=is_sparse_counts,
            binary_feature_fraction=binary_feature_fraction,
            feature_signal_strength=feature_signal_strength,
            feature_signal_p90=feature_signal_p90,
            task=task,
            y_skewness=y_skewness,
            y_all_positive=y_all_positive,
        )

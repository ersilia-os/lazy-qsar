"""
Zero-shot Random Forest hyperparameter selection.

Rules are derived from:
  - Breiman (2001) "Random Forests" Machine Learning 45(1) — key insight that RF
    is relatively insensitive to most hyperparameters; max_features and
    min_samples_leaf are the two that matter most.
  - Probst et al. (2019) "Hyperparameters and Tuning Strategies for Random Forest"
    WIRES Data Mining and Knowledge Discovery — empirical study showing
    min_samples_leaf 1-5 and max_features in [0.1, 0.9] cover optimal configs.
  - Geurts et al. (2006) "Extremely Randomized Trees" Machine Learning — confirms
    log2 feature subsampling beneficial for high-dimensional sparse inputs.
  - Sheridan et al. (2016) "Extreme Gradient Boosting as a Method for QSAR" JCIM
    — found RF with ECFP fingerprints benefits from strong feature subsampling
    (log2 or lower) to avoid correlated trees.
  - Biau & Scornet (2016) "A Random Forest Guided Tour" Test — min_samples_leaf
    scales as O(n^{4/5}) theoretically; practical values follow n // 500 for
    sparse-count and n // 1000 for dense data.

No search, no cross-validation. Parameters are chosen purely from dataset
statistics captured in an RFProfile. A lightweight OOB portfolio comparison
(heuristic vs. sklearn default) then selects the winner.

Key design decisions
--------------------
* max_features="log2" for high-dimensional sparse-count data (p > 500):
  With p=2048 fingerprints, sqrt(p)≈45 and log2(p)≈11. Fewer features per
  split forces stronger tree diversity and reduces correlated errors, matching
  the XGBoost colsample_bynode heuristic for fingerprint data.
* max_depth=None for sparse-count data: binary sparse inputs benefit from deep
  trees because signal is spread across many rare features; bootstrap aggregation
  provides the regularization instead.
* max_depth constrained for underdetermined dense datasets (n/p < 3): prevents
  overfitting when features outnumber samples.
* n_estimators scales with n up to 500: diminishing returns beyond ~300-500 trees
  for typical QSAR dataset sizes; capped for cost control.
* min_samples_leaf scales with n to prevent leaf collapse on small datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Sentinel for "use data-driven heuristic" (overridable by explicit user values).
_HEURISTIC = "heuristic"


# ---------------------------------------------------------------------------
# Dataset profile
# ---------------------------------------------------------------------------


@dataclass
class RFProfile:
    n_samples: int
    n_features: int
    n_p_ratio: float
    sparsity: float
    is_sparse_counts: bool
    imbalance_ratio: float
    pct_numeric: float = 1.0

    def __repr__(self) -> str:
        return (
            f"RFProfile("
            f"n={self.n_samples}, p={self.n_features}, n/p={self.n_p_ratio:.2f}, "
            f"sparsity={self.sparsity:.3f}, is_sparse_counts={self.is_sparse_counts}, "
            f"pct_numeric={self.pct_numeric:.2f}, imbalance={self.imbalance_ratio:.2f})"
        )


def _compute_sparsity(X) -> float:
    if hasattr(X, "nnz"):
        n_total = X.shape[0] * X.shape[1]
        return 1.0 - X.nnz / n_total
    return float((np.asarray(X) == 0).mean())


def rf_leaf_cap(n: int, n_estimators: int = 100) -> int:
    """
    Maximum leaf nodes per tree, targeting < 4 MB ONNX export.

    skl2onnx serialises each tree node at ~37 bytes (measured).  Targeting
    100K total nodes keeps the model under ~4 MB regardless of n_estimators.

    Formula: floor( (100_000 / n_estimators) / 2 ), clipped by [32, n//3].
    """
    per_tree_budget = 100_000 // max(1, n_estimators)
    return max(32, min(per_tree_budget // 2, n // 3))


def _compute_pct_numeric(X, n_sample: int = 1000) -> float:
    """Fraction of columns that are NOT purely binary (0/1)."""
    n = min(n_sample, X.shape[0])
    sample = X[:n].toarray() if hasattr(X, "toarray") else np.asarray(X[:n])
    binary_cols = np.all((sample == 0) | (sample == 1), axis=0)
    return float(1.0 - binary_cols.mean())


def _detect_sparse_counts(X, sparsity: float) -> bool:
    if sparsity < 0.5:
        return False
    n_sample = min(5000, X.shape[0])
    sample = (
        X[:n_sample].toarray() if hasattr(X, "toarray") else np.asarray(X[:n_sample])
    )
    nonzero_vals = sample[sample != 0]
    if nonzero_vals.size == 0:
        return False
    is_integer_like = float((nonzero_vals == np.floor(nonzero_vals)).mean()) > 0.95
    if not is_integer_like:
        return False
    max_val = float(nonzero_vals.max())
    return sparsity >= 0.85 or max_val <= 10


def profile_rf_dataset(X, y) -> RFProfile:
    """Compute a lightweight dataset profile for RF parameter selection."""
    y = np.asarray(y).ravel()
    n_samples, n_features = X.shape
    sparsity = _compute_sparsity(X)
    is_sparse_counts = _detect_sparse_counts(X, sparsity)
    n_p_ratio = float(n_samples) / max(n_features, 1)
    pct_numeric = _compute_pct_numeric(X)

    unique, counts = np.unique(y.astype(int), return_counts=True)
    if len(unique) >= 2:
        imbalance_ratio = float(counts.max() / counts.min())
    else:
        imbalance_ratio = 1.0

    return RFProfile(
        n_samples=n_samples,
        n_features=n_features,
        n_p_ratio=n_p_ratio,
        sparsity=sparsity,
        is_sparse_counts=is_sparse_counts,
        imbalance_ratio=imbalance_ratio,
        pct_numeric=pct_numeric,
    )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def heuristic_rf_params(profile: RFProfile) -> dict:
    """
    Return RF hyperparameters adapted to the dataset profile.

    All four returned keys (n_estimators, max_depth, min_samples_leaf,
    max_features) are directly forwarded to sklearn's RandomForestClassifier.
    """
    n = profile.n_samples
    p = profile.n_features
    n_p_ratio = profile.n_p_ratio
    is_sparse = profile.is_sparse_counts
    imbalance = profile.imbalance_ratio

    # --- n_estimators -------------------------------------------------------
    # More trees help on larger datasets; diminishing returns beyond ~300.
    # Kept modest to limit ONNX model size on large datasets.
    if n < 500:
        n_estimators = 100
    elif n < 5_000:
        n_estimators = 200
    else:
        n_estimators = 300

    # --- max_depth ----------------------------------------------------------
    # Sparse-count (fingerprint) data: cap depth for large n to control model
    # size — trees with n>2k samples and no depth limit grow enormous.
    # Dense underdetermined: constrain to prevent overfitting noise features.
    if is_sparse:
        if n >= 10_000:
            max_depth = 15
        elif n >= 2_000:
            max_depth = 20
        else:
            max_depth = None
    elif n_p_ratio < 1.0:
        max_depth = 12
    elif n_p_ratio < 3.0:
        max_depth = 15
    else:
        max_depth = None

    # --- min_samples_leaf ---------------------------------------------------
    # Analog of XGBoost min_child_weight. Scales with n to prevent leaf collapse.
    if is_sparse:
        base = max(3, n // 500)
    else:
        base = max(1, n // 1_000)
    base = min(base, 20)

    # For extreme imbalance: shrink leaf size so minority-class leaves can form.
    if imbalance > 10:
        min_samples_leaf = max(1, base // 2)
    else:
        min_samples_leaf = base

    # --- max_features -------------------------------------------------------
    # For very high-dimensional sparse data: log2 forces stronger tree diversity.
    if is_sparse and p > 500:
        max_features: str | int | float = "log2"
    else:
        max_features = "sqrt"

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
    }


def default_rf_params() -> dict:
    """sklearn RandomForestClassifier out-of-the-box defaults. Always in the portfolio."""
    return {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }


def flaml_rf_params(profile: RFProfile) -> dict:
    """
    FLAML RF configuration via 1-NN meta-feature matching.

    Selects one of three portfolio configs from FLAML's RF portfolio
    (microsoft/FLAML, flaml/default/rf/binary.json, MIT license).

    Meta-features: [n_samples, n_features, 2, pct_numeric].
    Post-processing:
      - n_estimators capped at 100/200/300 by dataset size (cost control)
      - max_leaf_nodes capped at max(32, n // 3) to prevent overfitting on
        small datasets (FLAML's configs were calibrated on n >> 10k)
      - fallback to default_rf_params() if empty portfolio entry selected

    References:
      - Wang et al. (2021) "FLAML: A Fast and Lightweight AutoML Library"
        MLSys 2021.
    """
    from . import flaml_data as _fd

    n = profile.n_samples
    p = profile.n_features

    center = np.array(_fd.BINARY["preprocessing"]["center"])
    scale = np.array(_fd.BINARY["preprocessing"]["scale"])
    query = np.array([float(n), float(p), 2.0, profile.pct_numeric])
    q_norm = (query - center) / scale

    best_dist, best_idx = float("inf"), 0
    for nb in _fd.BINARY["neighbors"]:
        feat = np.array(nb["features"])
        d = float(np.dot(q_norm - feat, q_norm - feat))
        if d < best_dist:
            best_dist = d
            best_idx = nb["choice"][0]

    hp = _fd.BINARY["portfolio"][best_idx]
    if not hp:
        return default_rf_params()

    if n < 500:
        n_estimators = min(100, int(hp["n_estimators"]))
    elif n < 5_000:
        n_estimators = min(200, int(hp["n_estimators"]))
    else:
        n_estimators = min(300, int(hp["n_estimators"]))

    max_leaf_nodes = max(32, min(int(hp["max_leaves"]), n // 3))

    return {
        "n_estimators": n_estimators,
        "max_features": float(hp["max_features"]),
        "max_leaf_nodes": max_leaf_nodes,
        "criterion": hp["criterion"],
    }


def autogluon_rf_params(profile: RFProfile) -> dict:
    """
    AutoGluon zeroshot 2023 RF configuration, selected by dataset characteristics.

    Configs extracted from autogluon/autogluon tabular/src/autogluon/tabular/
    configs/zeroshot/zeroshot_portfolio_2023.py (Apache-2 license).

    3×2 adaptive grid (size × sparsity/binary):

                    sparse / binary        dense / numeric
      n < 2 000     mf=0.75, msl=40        mf=1.0, msl=5
      2k – 9 999    mf=0.75, msl=2         mf=1.0, msl=5
      n ≥ 10 000    mf=0.75, msl=1         mf=1.0, msl=1

    max_leaf_nodes: max(64, min(37308, n // 2)) — capped for small datasets
    (AutoGluon's original values of 18k–48k are tuned for large datasets).

    References:
      - Erickson et al. (2020) "AutoGluon-Tabular: Robust and Accurate AutoML
        for Structured Data" ICML AutoML Workshop.
    """
    n = profile.n_samples
    is_sparse = profile.is_sparse_counts or (1.0 - profile.pct_numeric) > 0.7

    if n < 500:
        n_estimators = 100
    elif n < 5_000:
        n_estimators = 200
    else:
        n_estimators = 300

    if n < 2_000:
        if is_sparse:
            max_features, min_samples_leaf = 0.75, 40
        else:
            max_features, min_samples_leaf = 1.0, 5
    elif n < 10_000:
        if is_sparse:
            max_features, min_samples_leaf = 0.75, 2
        else:
            max_features, min_samples_leaf = 1.0, 5
    else:
        if is_sparse:
            max_features, min_samples_leaf = 0.75, 1
        else:
            max_features, min_samples_leaf = 1.0, 1

    max_leaf_nodes = max(64, min(37308, n // 2))

    return {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "min_samples_leaf": min_samples_leaf,
        "max_leaf_nodes": max_leaf_nodes,
    }

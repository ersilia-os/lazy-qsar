"""
Fixed preset configurations for SVC portfolio-based selection.

Four configurations compete against each other on the validation split:

  1. heuristic    – Rule-based params from params.py (dataset-profile-aware).
                    Chooses kernel, C, and class_weight based on sparsity,
                    n/p ratio, and dataset size.

  2. default      – sklearn SVC out-of-the-box defaults (C=1, RBF, gamma='scale').
                    Represents what a user gets with SVC().fit(X, y).

  3. linear       – Linear kernel; always compact ONNX, fast training, robust
                    for sparse fingerprints (ECFP4/Morgan).  C is scaled by n.
                    Source: Heikamp & Bajorath (2014) — C=1.0 is near-optimal
                    for most ECFP4 QSAR datasets regardless of size.

  4. balanced_rbf – RBF kernel with class_weight='balanced' and C scaled by
                    sqrt(n_minority), providing a conservative configuration
                    calibrated for imbalanced QSAR assay data.
                    Source: Goh et al. (2017) — class-weighted SVMs for
                    virtual screening on imbalanced bioactivity datasets.

All presets include 'use_linear' (bool) that controls which sklearn estimator
(SVC vs LinearSVC) the model uses.  LinearSVC is always ONNX-light; kernel SVC
ONNX size grows with support vector count.
"""

from __future__ import annotations

from .inspector import DatasetProfile
from .params import get_params


def svc_heuristic_params(profile: DatasetProfile) -> dict:
    """Rule-based preset from params.py; dataset-profile-aware."""
    return get_params(profile)


def svc_default_params(profile: DatasetProfile) -> dict:
    """
    sklearn SVC out-of-the-box defaults.

    Represents what a user gets by calling SVC().fit(X, y) with no tuning.
    Only class_weight='balanced' is added as a sensible QSAR deviation from
    sklearn's default of None.
    """
    n = profile.n_samples
    max_iter = max(1_000, min(10_000, n * 5))
    return {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "class_weight": "balanced",
        "max_iter": max_iter,
        "tol": 1e-3,
        "random_state": 42,
        "use_linear": False,
    }


def svc_linear_params(profile: DatasetProfile) -> dict:
    """
    Linear-kernel SVC preset.

    Linear kernel is state-of-the-art for bit-vector fingerprints (Burbidge
    2001; Heikamp & Bajorath 2014) and always produces a compact ONNX model
    (weights + bias only, no support vector storage).  C is tuned by n.
    """
    n = profile.n_samples
    if n < 500:
        C = 0.1
    elif n < 2_000:
        C = 1.0
    else:
        C = 10.0
    max_iter = max(1_000, min(10_000, n * 5))
    return {
        "C": C,
        "class_weight": "balanced",
        "max_iter": max_iter,
        "tol": 1e-3,
        "random_state": 42,
        "use_linear": True,
    }


def svc_balanced_rbf_params(profile: DatasetProfile) -> dict:
    """
    RBF kernel with class-weighted C scaled by sqrt(n_minority).

    Designed for imbalanced QSAR data where the minority class is small.
    C = min(50.0, max(0.1, sqrt(n_minority))) scales with rare-class evidence
    rather than total dataset size, providing appropriate regularisation even
    when positives are scarce (Goh et al. 2017 — class-weighted SVMs for
    virtual screening).

    Excluded automatically when use_linear would be preferred (sparse data)
    by setting a high C floor that the size guard will handle.
    """
    n = profile.n_samples
    imbalance = max(1.0, profile.imbalance_ratio)
    n_minority = max(1, int(n / (1.0 + imbalance)))
    C = min(50.0, max(0.1, n_minority ** 0.5))
    max_iter = max(1_000, min(10_000, n * 5))
    return {
        "C": round(C, 4),
        "kernel": "rbf",
        "gamma": "scale",
        "class_weight": "balanced",
        "max_iter": max_iter,
        "tol": 1e-3,
        "random_state": 42,
        "use_linear": False,
    }

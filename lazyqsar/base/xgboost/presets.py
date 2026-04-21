"""
Fixed preset configurations for portfolio-based selection.

Four externally-derived XGBoost configurations that compete against the
rule-based heuristic preset on the validation split:

  1. xgb_default  – XGBoost built-in defaults (lr=0.3, max_depth=6).
                    Baseline: what you get with XGBClassifier().fit(X, y).

  2. flaml        – FLAML: 1-NN meta-feature matching on the
                    portfolio from microsoft/FLAML (MIT license).
                    Lossguide (max_leaves) variant with colsample_bylevel —
                    FLAML's distinctive ingredient.  Meta-features:
                    [n_samples, n_features, n_classes, pct_numeric_features].

  3. autogluon    – AutoGluon tabular XGBoost default configuration.
                    From autogluon/autogluon tabular/.../xgboost/ (Apache-2):
                    lr=0.1, max_depth=6, colsample_bytree=1.0, mcw=1.
                    subsample/reg_alpha/reg_lambda are NOT in AutoGluon's HPO
                    space; XGBoost defaults (1.0, 0.0, 1.0) apply.

  4. rf           – XGBoost configured as a boosted Random Forest.
                    num_parallel_tree=3 + colsample_bynode=sqrt(p)/p mirrors
                    sklearn RF's per-split diversity rule.

All functions accept (profile, device) and return a full params dict
that includes n_estimators, early_stopping_rounds, objective, and
eval_metric — ready to be consumed by _train_phase1() in model.py.

Task-specific parameters (objective, eval_metric, scale_pos_weight) are
injected by _add_task_params() so all presets share the same evaluation
metric for a fair comparison.
"""

from typing import Dict, Any

import numpy as np

from .inspector import DatasetProfile
from . import flaml_data as _fd


# Metrics where higher is better (all others are minimised by XGBoost)
MAXIMIZE_METRICS = frozenset({"auc", "aucpr", "map", "ndcg"})


def _add_task_params(params: Dict[str, Any], profile: DatasetProfile) -> None:
    """Inject objective, eval_metric, and imbalance correction into params."""
    if profile.task == "classification":
        params["objective"] = "binary:logistic"
        ratio = profile.imbalance_ratio
        params["scale_pos_weight"] = round(ratio, 4)
        if ratio > 10:
            params["max_delta_step"] = 1
        params["eval_metric"] = "aucpr" if ratio > 10 else "auc"
    else:
        # Simple squarederror for all external presets; the heuristic preset
        # has a more refined skewness-based objective selection.
        params["objective"] = "reg:squarederror"
        params["eval_metric"] = "rmse"


def xgb_default_params(
    profile: DatasetProfile, device: str, nthread: int = -1
) -> Dict[str, Any]:
    """
    XGBoost out-of-the-box defaults.

    Represents what a user gets by calling XGBClassifier().fit(X, y) with
    no tuning.  Only the task objective, eval metric, and imbalance correction
    are added to enable a fair comparison.
    """
    n = profile.n_samples
    max_bin = 128 if n > 100_000 else 256
    params: Dict[str, Any] = {
        "tree_method": "hist",
        "device": "cuda" if device == "gpu" else "cpu",
        "learning_rate": 0.3,  # XGBoost default
        "max_depth": 6,  # XGBoost default
        "min_child_weight": 1,  # XGBoost default
        "subsample": 1.0,  # XGBoost default
        "colsample_bytree": 1.0,  # XGBoost default
        "reg_alpha": 0.0,  # XGBoost default
        "reg_lambda": 1.0,  # XGBoost default
        "max_bin": max_bin,
        "n_estimators": 2000,
        "early_stopping_rounds": 50,
    }
    if device == "cpu":
        import os

        params["nthread"] = (os.cpu_count() or 1) if nthread == -1 else nthread
    _add_task_params(params, profile)
    return params


def flaml_params(
    profile: DatasetProfile, device: str, nthread: int = -1
) -> Dict[str, Any]:
    """
    FLAML configuration via 1-NN meta-feature matching.

    Selects one of four portfolio configs from FLAML's lossguide portfolio
    (microsoft/FLAML, flaml/default/xgboost/{binary,regression}.json, MIT).

    Post-processing applied for small-dataset robustness:
      - n_estimators capped at 2000 (early stopping is the real control)
      - max_leaves capped at max(64, n // 10) to prevent overfitting on
        small datasets (FLAML's configs were calibrated on n >> 10k)
      - min_child_weight floored at 1 (FLAML's near-zero values assume
        large n; flooring prevents single-sample leaves on small datasets)
      - early_stopping_rounds = min(200, max(50, 50 × 0.1 / lr)) so that
        very slow FLAML learners (lr ≈ 0.001) don't require 5000+ patience
        rounds within the portfolio comparison budget
    """
    n = profile.n_samples
    p = profile.n_features
    pct_numeric = 1.0 - profile.binary_feature_fraction

    data = _fd.BINARY if profile.task == "classification" else _fd.REGRESSION
    n_classes = 2 if profile.task == "classification" else 0

    center = np.array(data["preprocessing"]["center"])
    scale = np.array(data["preprocessing"]["scale"])
    query = np.array([n, p, n_classes, pct_numeric])
    q_norm = (query - center) / scale

    best_dist, best_idx = float("inf"), 0
    for nb in data["neighbors"]:
        feat = np.array(nb["features"])
        d = float(np.dot(q_norm - feat, q_norm - feat))  # squared L2
        if d < best_dist:
            best_dist = d
            best_idx = nb["choice"][0]

    hp = data["portfolio"][best_idx]
    lr = float(hp["learning_rate"])
    max_bin = 128 if n > 100_000 else 256

    params: Dict[str, Any] = {
        "tree_method": "hist",
        "device": "cuda" if device == "gpu" else "cpu",
        "grow_policy": "lossguide",
        "max_depth": 0,  # unlimited; max_leaves controls capacity
        "max_leaves": max(64, min(n // 10, int(hp["max_leaves"]))),
        "learning_rate": lr,
        "min_child_weight": max(1.0, float(hp["min_child_weight"])),
        "subsample": float(hp["subsample"]),
        "colsample_bylevel": float(hp["colsample_bylevel"]),
        "colsample_bytree": float(hp["colsample_bytree"]),
        "reg_alpha": float(hp["reg_alpha"]),
        "reg_lambda": float(hp["reg_lambda"]),
        "max_bin": max_bin,
        "n_estimators": 2000,
        "early_stopping_rounds": min(200, max(50, int(round(50 * 0.1 / lr)))),
    }
    if device == "cpu":
        import os

        params["nthread"] = (os.cpu_count() or 1) if nthread == -1 else nthread
    _add_task_params(params, profile)
    return params


def autogluon_params(
    profile: DatasetProfile, device: str, nthread: int = -1
) -> Dict[str, Any]:
    """
    AutoGluon zeroshot 2023 XGBoost configuration, selected by dataset characteristics.

    AutoGluon's zeroshot portfolio (autogluon/autogluon tabular/src/autogluon/tabular/
    configs/zeroshot/zeroshot_portfolio_2023.py, Apache-2 license) contains 9 XGBoost
    configurations discovered via ensemble simulation on 200 datasets.  Rather than
    running all 9, we select one based on two dataset axes:

      Size axis  (n_samples):
        small  < 2 000   → shallow trees, higher min_child_weight
        medium  2 000 – 9 999 → moderate depth and learning rate
        large  ≥ 10 000  → deep trees with low learning rate

      Feature axis (is_sparse_counts or binary_feature_fraction > 0.7):
        sparse/binary  → stronger colsample subsampling for feature diversity
        dense/numeric  → milder colsample, higher min_child_weight

    Resulting 3×2 grid (AutoGluon suffix / priority in parentheses):

                    sparse / binary          dense / numeric
      small         _r89  lr=0.088 d=5       _r95  lr=0.066 d=5  mcw=1.41
      medium        _r49  lr=0.038 d=7       _r194 lr=0.093 d=7
      large         _r33  lr=0.018 d=6       _r34  lr=0.029 d=6

    Note: the original AutoGluon _r33/_r34 configs use max_depth=10, which is
    designed for ensemble search (where many shallow models are combined).
    Standalone use requires depth ≤ 6 to stay within the portfolio cost budget
    (max_depth=10 → 1024 leaves → cost ratio >300× default, always filtered).

    early_stopping_rounds is set proportionally to the learning rate
    (same formula as flaml_params): min(200, max(50, round(50 × 0.1 / lr))).
    """
    n = profile.n_samples
    is_sparse = profile.is_sparse_counts or profile.binary_feature_fraction > 0.7

    if n < 2_000:
        if is_sparse:
            # _r89: fast shallow learner with column subsampling
            lr, max_depth, min_child_weight, colsample_bytree = 0.088, 5, 0.63, 0.66
        else:
            # _r95: regularised shallow learner (high min_child_weight)
            lr, max_depth, min_child_weight, colsample_bytree = 0.066, 5, 1.41, 0.98
    elif n < 10_000:
        if is_sparse:
            # _r49: moderate depth with column subsampling
            lr, max_depth, min_child_weight, colsample_bytree = 0.038, 7, 0.56, 0.75
        else:
            # _r194: moderate depth, mild column subsampling
            lr, max_depth, min_child_weight, colsample_bytree = 0.093, 7, 0.80, 0.91
    else:
        if is_sparse:
            # _r33: slow learning rate, depth capped at 6 for cost viability
            # (original d=10 → cost ratio >300× default; d=6 → ~19× at any n)
            lr, max_depth, min_child_weight, colsample_bytree = 0.018, 6, 0.60, 0.69
        else:
            # _r34: strong column subsampling, depth capped at 6 for cost viability
            lr, max_depth, min_child_weight, colsample_bytree = 0.029, 6, 1.15, 0.55

    max_bin = 128 if n > 100_000 else 256
    early_stopping_rounds = min(200, max(50, int(round(50 * 0.1 / lr))))

    params: Dict[str, Any] = {
        "tree_method": "hist",
        "device": "cuda" if device == "gpu" else "cpu",
        "learning_rate": lr,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "colsample_bytree": colsample_bytree,
        "subsample": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "max_bin": max_bin,
        "n_estimators": 2000,
        "early_stopping_rounds": early_stopping_rounds,
    }
    if device == "cpu":
        import os

        params["nthread"] = (os.cpu_count() or 1) if nthread == -1 else nthread
    _add_task_params(params, profile)
    return params


def rf_params(
    profile: DatasetProfile, device: str, nthread: int = -1
) -> Dict[str, Any]:
    """
    XGBoost configured as a boosted Random Forest.

    Mirrors sklearn's RandomForestClassifier() defaults as closely as the
    XGBoost boosted-RF architecture allows:

      colsample_bynode = sqrt(p)/p  — exactly sklearn's max_features='sqrt':
        at each split, sample sqrt(p) candidate features.  No floor is
        applied; very high-dimensional data (e.g. ECFP4 fingerprints) gets
        the same aggressive subsampling as sklearn RF.

      subsample = 0.632  — approximates sklearn's bootstrap=True: sampling
        with replacement leaves ~63.2% unique rows per tree, so subsample
        without replacement at 0.632 is the standard approximation.

      max_depth — sklearn RF grows trees to purity (max_depth=None).
        XGBoost hist is more expensive per level than sklearn RF, but for
        small datasets the cost is negligible and deeper trees close the gap:
          n <  1 000  → depth 12  (≈ purity on typical drug datasets)
          n <  5 000  → depth 10
          n < 20 000  → depth  8
          n ≥ 20 000  → depth  6  (cost-efficient at large scale)

    Note on num_parallel_tree: setting num_parallel_tree=3 (XGBoost RF mode)
    was tested and found to be both slower (3.7×) and worse in AUC than
    num_parallel_tree=1 on fingerprint data.  The parallel trees add variance
    rather than reducing it here — RF diversity is already provided by
    colsample_bynode and subsample.  It is therefore not set in this preset.
    """
    n = profile.n_samples
    p = profile.n_features
    # Exact sklearn sqrt(p) rule — no floor so large-p datasets (e.g. ECFP4)
    # get proper feature diversity rather than the previous 0.05 clamp.
    csn = round(min(0.5, 1.0 / (p**0.5)), 4)
    max_bin = 128 if n > 100_000 else 256

    # Adaptive depth: sklearn RF grows to purity; approximate this for small n
    # where deep trees are cheap and significantly close the RF performance gap.
    if n < 1_000:
        max_depth = 12
    elif n < 5_000:
        max_depth = 10
    elif n < 20_000:
        max_depth = 8
    else:
        max_depth = 6

    params: Dict[str, Any] = {
        "tree_method": "hist",
        "device": "cuda" if device == "gpu" else "cpu",
        # num_parallel_tree intentionally absent: tested and found to hurt AUC
        # while tripling cost; colsample_bynode + subsample provide RF diversity.
        "subsample": 0.632,  # bootstrap approximation (was 0.8)
        "colsample_bynode": csn,  # exact sqrt(p)/p, no floor (was clamped at 0.05)
        "colsample_bytree": 1.0,
        "learning_rate": 0.1,
        "max_depth": max_depth,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "max_bin": max_bin,
        "n_estimators": 2000,
        "early_stopping_rounds": 50,
    }
    if device == "cpu":
        import os

        params["nthread"] = (os.cpu_count() or 1) if nthread == -1 else nthread
    _add_task_params(params, profile)
    return params

# base/xgboost

Automatic XGBoost hyperparameter selection based on dataset profiling and portfolio comparison — no grid search, no manual tuning.

## Classes

| Class | Description |
|-------|-------------|
| `BaseXGBClassifier` | Binary classifier with auto-selected XGBoost parameters |
| `BaseXGBRegressor` | Regressor with auto-selected XGBoost parameters |

## Training procedure

### Phase 0 — dataset profiling

The following statistics are computed from X and y:

| Statistic | Role |
|-----------|------|
| `n_samples`, `n_features`, `n_p_ratio` | Drive depth, regularization, subsampling |
| `sparsity`, `is_sparse_counts` | Switch between depthwise and lossguide tree growth |
| `binary_feature_fraction` | Adjust max_depth for one-hot style inputs |
| `feature_signal_strength`, `feature_signal_p90` | Modulate regularization strength (±15%) |
| `imbalance_ratio` | Set `scale_pos_weight`, `max_delta_step`, eval metric |
| `y_skewness`, `y_all_positive` | Select regression objective (squarederror / tweedie / pseudohubererror) |

### Phase 1 — heuristic parameter selection

Parameters are derived by rule, not search. Key rules:

**Learning rate** — scales with n and data type:

| Condition | Learning rate |
|-----------|--------------|
| n < 1K, sparse counts | 0.10 |
| 1K ≤ n < 10K, sparse counts | 0.05 |
| n < 10K, dense | 0.10 |
| 10K ≤ n < 100K, sparse | 0.10 |
| 10K ≤ n < 100K, dense | 0.05 |
| 100K ≤ n < 1M, dense | 0.02 |
| n ≥ 1M, dense | 0.05 |

**Tree growth strategy:**
- `is_sparse_counts` and p > 200 → `grow_policy="lossguide"`, `max_leaves = max(64, min(256, n//10))`, `max_depth=0`
- otherwise → depthwise; `max_depth` 3–6 depending on n, capped further by n/p ratio and binary_feature_fraction

**Regularization** — direct table keyed on (is_sparse_counts, n/p ratio); no multiplicative stacking. Signal strength modulates the result ±15%. Hard caps: `reg_lambda ≤ 4.0`, `reg_alpha ≤ 1.5`.

**min_child_weight** — `max(3, n//500)` for sparse counts (Probst et al. 2023 recommendation); `max(1, n//1000)` for dense. Halved when imbalance_ratio > 10.

**num_parallel_tree** — set to 3 for 200 ≤ n < 2000 and n/p ≥ 0.8 (RF-style bagging to reduce variance at small n).

**early_stopping_rounds** — `max(20, round(50 × 0.1 / lr))` so slower learners get proportional patience.

### Phase 2 — portfolio comparison (optional, default on)

When n ≥ 200, four preset configurations compete on a 90/10 validation split (Stage 1, fast budget), then the winner is calibrated on repeated splits (Stage 2):

| Preset | Description |
|--------|-------------|
| `heuristic` | Phase 1 rule-based parameters |
| `default` | XGBoost out-of-the-box defaults (lr=0.3, max_depth=6) |
| `flaml` | FLAML 1-NN portfolio: nearest neighbour in (n, p, n_classes, %numeric) space (microsoft/FLAML, MIT) |
| `autogluon` | AutoGluon tabular XGBoost configs selected by (n, is_sparse) grid (apache-2) |

A non-default preset wins only if its Stage-1 score exceeds the default by a noise-aware threshold (`max(0.005, coef/sqrt(n_minority))`).

### Phase 3 — full retraining

The winning preset is retrained on 100% of the data for `max(best_iteration, early_stopping_rounds, 100)` rounds (no early stopping).

## Outputs

`save(directory)` writes two files:
- `xgboost.onnx` — the booster serialised via `onnxmltools`
- `xgboost.json` — metadata: task, preset_name, best_iteration, params, dataset profile, portfolio scores

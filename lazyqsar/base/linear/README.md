# base/linear

Automatic linear model selection based on dataset shape — solver, regularization strength, and feature selection are derived from data characteristics without manual tuning.

## Classes

| Class | Description |
|-------|-------------|
| `BaseLinearClassifier` | Binary logistic regression with embedded L1/ElasticNet feature selection |
| `BaseLinearRegressor` | Linear regression with embedded feature selection and sample weighting |
| `BaseLinearArtifact` | Inference-only loader (ONNX or joblib) |

## Regime detection

The classifier and regressor both expose three regimes, but the classifier now
switches to the SGD-based `large` path based on estimated CV cost as well as raw
dataset shape so that medium-sized dense problems do not pay for an expensive
`LogisticRegressionCV` search.

| Regime | Condition |
|--------|-----------|
| `standard` | classifier: small `n·p` cost and `p ≤ n`; regressor: `n ≤ 50K` and `p ≤ n` |
| `high_dim` | classifier: small `n·p` cost and `p > n`; regressor: `n ≤ 50K` and `p > n` |
| `large` | classifier: large `n·p` cost or `n > 50K`; regressor: `n > 50K` |

Feature scaling is not applied internally — it is the caller's responsibility.

## Classifier logic

### Preprocessing

`VarianceThreshold(0.0)` is always applied first (removes only constant features).

### Regime-specific strategy

| Regime | Solver | Feature selection |
|--------|--------|------------------|
| `standard` | `LogisticRegressionCV` (saga, L1) | None beyond VarianceThreshold |
| `high_dim` | `LogisticRegressionCV` (saga, ElasticNet) | `SelectFromModel(LogisticRegression)` pre-filter, max_features = min(p, 2n) |
| `large` | `SGDClassifier` (ElasticNet, alpha tuned on subsample) | None beyond VarianceThreshold |

For the classifier, `LogisticRegressionCV` is intentionally reserved for
datasets with modest dense feature work. A shape like `5000 × 2000` is routed
to `large` even though `n < 50K`, because the CV search is too expensive for
that matrix size.

### Hyperparameter heuristics

| Parameter | Rule |
|-----------|------|
| C grid | Centered on `sqrt(n)/p` (lasso-theory optimum), ±2 log-decades, adaptive grid: 20 / 12 / 8 points by `n·p` cost |
| alpha grid (SGD) | Derived from C grid: `1/(C·n)` |
| l1_ratio | p/n > 10 → 0.5; p/n > 2 → 0.7; else 0.9 (more grouping at high p/n) |
| CV folds | n < 200 → min(min_class, 10); n < 1K → 5; else 3 |
| CV scoring | `roc_auc` |
| class_weight | `"balanced"` always |

## Regressor logic

### Regime-specific strategy

| Regime | Solver | Feature selection |
|--------|--------|------------------|
| `standard` | `RidgeCV` (L2) | None beyond VarianceThreshold |
| `high_dim` | `ElasticNetCV` | `SelectFromModel(Lasso)` pre-filter, max_features = min(p, 2n) |
| `large` | `SGDRegressor` (ElasticNet, alpha tuned on subsample) | None beyond VarianceThreshold |

### Hyperparameter heuristics

| Parameter | Rule |
|-----------|------|
| CV scoring | `neg_mean_absolute_error` when \|skewness(y)\| > 1; else `r2` |
| Sample weights | Inverse-frequency over quantile bins of y (analogous to `class_weight="balanced"`) |
| l1_ratio | Same p/n heuristic as classifier |
| CV folds | n < 200 → min(n//10, 10); n < 1K → 5; else 3 |

## Outputs

`save(directory)` writes two files:
- `linear.onnx` — the full pipeline (VarianceThreshold + optional SelectFromModel + estimator) serialised via `skl2onnx`
- `linear.json` — metadata: task, format, regime, n_features_in, feature_mask, classes (classifier only)

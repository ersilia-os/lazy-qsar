# base/preprocessing

Automatic preprocessing pipeline selection based on dataset profiling. Picks a scaler and feature reducer from a single pass over the data — no cross-validation, no grid search.

## Classes

| Class | Description |
|-------|-------------|
| `BasePreprocessor` | Core transformer; requires `task="classification"` or `"regression"` |
| `BaseClassifierPreprocessor` | `BasePreprocessor` fixed to classification |
| `BaseRegressorPreprocessor` | `BasePreprocessor` fixed to regression |
| `BasePreprocessorArtifact` | Inference-only loader (ONNX or joblib) |

## Pipeline structure

The pipeline is always fixed to four sequential steps:

1. `SimpleImputer(strategy="median", keep_empty_features=True)` — imputes missing values; no-op if data is complete
2. `VarianceThreshold(1e-6)` — removes constant and near-constant features before scaling
3. **Scaler** — one of four options, selected by rule (see below)
4. **Reducer** — one of two options, selected by rule (see below)

## Scaler selection rules

Rules are applied in priority order (first match wins):

| Priority | Condition | Scaler | Rationale |
|----------|-----------|--------|-----------|
| 1 | `is_sparse_counts=True` | `MaxAbsScaler` | Preserves sparsity; bits are already in {0,1} or small counts |
| 2 | `binary_feature_fraction ≥ 0.8` | `MaxAbsScaler` | Effectively a no-op on binary features |
| 3 | `sparsity > 0.5` | `MaxAbsScaler` | Avoids densification from RobustScaler/PowerTransformer |
| 4 | `outlier_fraction > 0.3` | `RobustScaler` | Median/IQR-based; immune to extreme values |
| 5 | `median_feature_skewness > 1.5` | `PowerTransformer` (Yeo-Johnson) | Handles skewed continuous features; falls back to `RobustScaler` on fit failure |
| 6 | default | `StandardScaler` | Zero mean, unit variance |

## Reducer selection rules

| Condition | Reducer | Behaviour |
|-----------|---------|-----------|
| p ≤ 50 or n/p ≥ 20 | `VarianceThreshold(1e-6)` | No reduction beyond constant-feature removal |
| otherwise | `CorrelationFilter` (preceded by `VarianceThreshold`) | Removes the lower-variance feature from each pair with \|Pearson r\| > 0.90 |

## Dataset profiling

The following statistics drive scaler and reducer selection:

| Statistic | How computed |
|-----------|-------------|
| `sparsity` | Fraction of zero entries in X |
| `is_sparse_counts` | sparsity > 0.5, non-zero values integer-like, max ≤ 10 or sparsity ≥ 0.85 |
| `binary_feature_fraction` | Fraction of features with only {0,1} values |
| `outlier_fraction` | Fraction of features where >5% of values fall outside [Q1−1.5·IQR, Q3+1.5·IQR] |
| `median_feature_skewness` | Median \|skewness\| across a random subsample of features |
| `median_abs_correlation` | Median \|Pearson r\| across randomly sampled feature pairs |
| `feature_signal_p90` | 90th-percentile \|Pearson r\| between features and target |
| `n_p_ratio` | n_samples / n_features |

## Outputs

`save(directory)` writes two files:
- `preprocessor.onnx` — the fitted pipeline serialised via `skl2onnx`
- `preprocessor.json` — metadata: task, scaler, reducer, n_features_in, n_features_out, kept_feature_indices

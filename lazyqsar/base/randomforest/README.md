# base/randomforest

Automatic Random Forest hyperparameter selection based on dataset profiling and OOB portfolio comparison — no grid search, no manual tuning.

## Classes

| Class | Description |
|-------|-------------|
| `BaseRFClassifier` | Binary classifier with auto-selected Random Forest parameters |
| `BaseRFRegressor` | Regressor with the same portfolio, scored by out-of-bag Spearman ρ |
| `BaseRFArtifact` | Inference-only loader (ONNX), for either task |

## Training procedure

### Phase 0 — dataset profiling

Key statistics computed from X and y:

| Statistic | Role |
|-----------|------|
| `n_samples`, `n_features`, `n_p_ratio` | Drive depth and leaf size selection |
| `imbalance_ratio` | Switch `class_weight` between `balanced` and `balanced_subsample` |
| `binary_feature_fraction`, `sparsity` | Adjust `max_features` strategy |

### Phase 1 — portfolio comparison (OOB-based)

When n ≥ 200, up to four preset configurations compete using out-of-bag (OOB) AUC scores. FLAML and AutoGluon presets are skipped for p > 200 (they were calibrated on low-dimensional tabular data):

| Preset | Description |
|--------|-------------|
| `heuristic` | Rule-based parameters derived from dataset profile |
| `default` | sklearn defaults (`n_estimators=100`, `max_features="sqrt"`) |
| `flaml` | FLAML 1-NN portfolio config (skipped if p > 200) |
| `autogluon` | AutoGluon tabular RF config (skipped if p > 200) |

A non-default preset wins only if its OOB AUC exceeds the default by at least `max(0.005, coef/√n_minority)`.

When n < 200, OOB estimates are unreliable so the default preset is used directly.

### Phase 2 — calibration

The full RF is fitted on all data, then k-fold OOF probabilities are collected by training separate fold-level estimators. The OOF scores are used to:
- Fit an isotonic or Platt calibrator (isotonic if minority count ≥ 500)
- Learn a decision cutoff that maximises balanced accuracy on OOF data
- Build a rank ECDF for `predict_rank()`

### Heuristic parameter rules

| Parameter | Rule |
|-----------|------|
| `n_estimators` | 100–500 depending on n and p |
| `max_depth` | None (unlimited) for large n; 10–20 for small n |
| `min_samples_leaf` | `max(1, n // 500)` |
| `max_features` | `"sqrt"` for sparse/binary features; `"log2"` or float for dense |
| `class_weight` | `"balanced"` by default; `"balanced_subsample"` when imbalance ≥ 3:1 |

## Regressor

`BaseRFRegressor` follows the same phases with these differences:

- **Portfolio score**: out-of-bag Spearman ρ; the threshold is `max(0.005, coef/√n)` (every
  sample counts, there is no minority class).
- **Presets**: the heuristic uses Breiman's `max_features=0.33` for dense data (sqrt/log2
  rules for sparse fingerprints are unchanged); `default` is sklearn's regressor default
  (`max_features=1.0`) with a leaf cap so the ONNX export stays bounded; `flaml` reads
  FLAML's RF *regression* portfolio (`flaml/default/rf/regression.json`, verbatim) and
  never sets a classification `criterion`.
- **Phase 2**: quantile-stratified k-fold out-of-fold predictions instead of calibration,
  giving `oof_preds_`, `oof_rho_`, `train_rho_` and the ECDF behind `predict_rank`.

## Outputs

`save(directory)` writes two files:
- `randomforest.onnx` — the estimator serialised via `skl2onnx`
- `randomforest.json` — metadata: task, selected_preset, n_features_in, decision_cutoff, calibrator, ranker knots

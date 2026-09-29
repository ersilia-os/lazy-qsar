# base/svc

Support Vector Classifier with automatic kernel and `C` selection from dataset profiling,
then a four-way portfolio comparison — no grid search, no manual tuning.

## Classes

| Class | Description |
|-------|-------------|
| `BaseSVCClassifier` | Binary classifier with auto-selected SVC parameters |
| `BaseSVCArtifact` | Inference-only loader (ONNX) |

## Training procedure

### Phase 0 — dataset profiling

`inspector.DatasetProfile` measures the same statistics the other heads profile on. Three
of them decide the configuration:

| Statistic | Role |
|-----------|------|
| `is_sparse_counts` | Sparse count features (Morgan/ECFP) force the linear kernel |
| `n_samples` | Above 5,000 forces the linear kernel; also scales `C` |
| `n_p_ratio` | On the RBF branch, sets the base `C` |

### Phase 1 — heuristic parameters (`params.get_params`)

The kernel is chosen before `C`, because it decides which `C` scale applies:

```
use_linear = is_sparse_counts or n > 5000
```

`LinearSVC` is chosen rather than `SVC(kernel="linear")` when that holds. The reason is
the export, not the fit: a kernel SVC's ONNX graph grows with its support-vector count,
while `LinearSVC` is a fixed-size coefficient vector whatever the dataset.

`C` then follows the branch:

| Branch | Rule |
|--------|------|
| Linear | 0.1 below n=500, 1.0 below n=2,000, else 10.0 |
| RBF | base 1.0 / 10.0 / 100.0 by `n_p_ratio` < 2 / < 5 / else, scaled by `min(1, sqrt(n/1000))` and capped at 100 |

The RBF scaling exists to stop `C=100` meeting a small `n`, which overfits hard. Both
branches set `class_weight="balanced"`, `gamma="scale"`, `tol=1e-3` and
`max_iter = clip(n * 10, 5_000, 50_000)`.

### Phase 2 — portfolio comparison

Four configurations compete on a validation split (`presets.py`):

| Preset | What it is |
|--------|------------|
| `heuristic` | Phase 1's rule-based parameters |
| `default` | sklearn's own defaults — what `SVC().fit(X, y)` gives, as the baseline to beat |
| `linear` | Linear kernel with `C` scaled by n. Heikamp & Bajorath (2014): `C=1.0` is near-optimal for most ECFP4 QSAR datasets regardless of size |
| `balanced_rbf` | RBF with `class_weight="balanced"` and `C` scaled by `sqrt(n_minority)`. Goh et al. (2017): class-weighted SVMs for virtual screening on imbalanced bioactivity data |

Including sklearn's defaults as a competitor is deliberate: it means the heuristics have to
earn their place on every dataset rather than being assumed better.

### Phase 3 — calibration

`SVC` has no calibrated probability of its own here — `probability=True` is deliberately
**not** passed, because it runs its own internal cross-validation and is deprecated in
scikit-learn 1.9. Instead `calibrate()` fits a calibrator on `sigmoid(decision_function)`
over stratified out-of-fold splits.

Portfolio selection runs **once**, inside the first `_fit_raw`; the k-fold loop reuses the
winning parameters rather than re-racing the portfolio per fold. Re-selecting per fold
would let different folds pick different kernels, and the out-of-fold scores would then
come from a model that does not exist.

`fit()` falls back to `_fit_raw` (uncalibrated) when either class has fewer than two
members, since there is nothing to hold out.

## Outputs

`predict_proba`, `predict_score` (pre-calibration), `predict_logit`, `predict_rank`
(percentile against this head's own out-of-fold scores) and `predict`. `predict` takes an
optional `cutoff`; without one it uses the head's learned decision cutoff.

## Export

`to_onnx` writes a float32 graph via `skl2onnx`. One caveat worth knowing: the SVC
converter reads `probA_`/`probB_` off the fitted estimator, and those attributes are
deprecated in scikit-learn 1.9 and scheduled for removal in 1.11. The pinned
`scikit-learn==1.9.1` keeps this working; it is a known upgrade blocker rather than a
current fault.

# LazyQSAR: Complete Pipeline Reference

---

## What is LazyQSAR?

LazyQSAR is an automated QSAR (Quantitative Structure-Activity Relationship) model-building library. Given a set of molecules and their measured biological activities (active/inactive labels), it automatically constructs, calibrates, and deploys a predictive classifier — with no hyperparameter tuning required from the user.

The core design principle is: **every decision that a practitioner would normally make manually is instead made by inspecting the data and applying theory-grounded rules.** There is no black-box AutoML search, no random hyperparameter sampling, and no manual feature engineering.

---

## Part 1: The Top-Level Interface

### 1.1 Two Modes of Use

**Mode A — SMILES-aware** (`LazyClassifierQSAR`)

The user provides a list of SMILES strings and binary labels. The library computes molecular descriptors internally and trains a model. The `mode` parameter controls which descriptor types are used:

| Mode | Descriptors computed |
|---|---|
| `"fast"` | Morgan fingerprints (2048 bits) |
| `"slow"` | CDDD + Chemeleon + CLAMP + Morgan + RDKit physicochemical |

**Critically: the descriptors are never concatenated.** Each descriptor type produces its own independent `LazyClassifier` trained on its own feature matrix. At prediction time, each model produces its own `P(y=1)` estimate and the final output is their **AUC-weighted average**. Weights combine a global skill score (from OOF and proxy AUCs) with a per-sample reliability score derived from rank-based prediction confidence. The ensemble operates at the prediction level, not the feature level.

```
SMILES
  ├── Morgan features →  [full LazyClassifier]  →  P(y=1)_morgan   ─┐
  ├── RDKit features  →  [full LazyClassifier]  →  P(y=1)_rdkit    ─┤
  ├── Chemeleon       →  [full LazyClassifier]  →  P(y=1)_chemeleon─┤
  ├── CLAMP           →  [full LazyClassifier]  →  P(y=1)_clamp    ─┤  W(i,j) weights
  └── CDDD            →  [full LazyClassifier]  →  P(y=1)_cddd     ─┘
                                                          ↓
                                              AUC+reliability-weighted average
                                                          ↓
                                                   final P(y=1)
```

Descriptor computation results are cached by MD5 hash of the SMILES list, so repeated calls with the same molecules skip recomputation.

**Mode B — Descriptor-agnostic** (`LazyClassifier`)

The user provides a pre-computed numeric feature matrix (numpy array or HDF5 file). This mode skips descriptor computation entirely. Everything below this point describes what happens inside a single `LazyClassifier`, regardless of which mode triggered it.

---

## Part 2: Inside a Single `LazyClassifier`

A single `LazyClassifier.fit(X, y)` runs four sequential stages:

1. Portfolio selection
2. Batch planning
3. Per-batch model fitting (the main pipeline)
4. Aggregation

---

## Part 3: Stage 1 — Portfolio Selection

Before any model is trained, the library decides **which of four base model types** to include in the ensemble: logistic regression (`lr`), XGBoost (`xgb`), random forest (`rf`), and support vector classifier (`svc`). This is handled by the `Portfolio` class.

### 3.1 The Three Profilers

Three independent data profilers exist in the codebase. They are used at different stages:

**`DatasetProfile`** (XGBoost inspector) — used by the Portfolio, XGBoost head, and SVC head:
- Sample size (n), feature count (p), n/p ratio
- Sparsity fraction (fraction of zeros in the matrix)
- `is_sparse_counts`: True when data looks like integer fingerprints (sparse, integer-valued, e.g. Morgan counts)
- Binary feature fraction (fraction of features that only take values 0 or 1)
- Mean feature–target signal strength (`feature_signal_strength`): mean absolute Pearson correlation between features and labels, estimated on a random subsample of up to 5000 rows and 500 features
- 90th-percentile signal (`feature_signal_p90`): the same, but at the 90th percentile
- Class imbalance ratio (n_majority / n_minority)

**`RFProfile`** (Random Forest inspector) — used only by the RF head. A lighter profile specific to RF parameter selection:
- n, p, n/p ratio, sparsity, `is_sparse_counts`, `imbalance_ratio`
- `pct_numeric`: fraction of columns that are NOT purely binary (complementary to `binary_feature_fraction` in `DatasetProfile`)

**`PreprocessingProfile`** (preprocessing inspector) — used only to select the scaler and reducer:
- Same n, p, n/p, sparsity, `is_sparse_counts`, binary fraction, `feature_signal_p90`
- Additionally: median feature skewness, outlier fraction (IQR-based), near-zero variance fraction, median pairwise correlation

All statistics are estimated stochastically (sampling rows and columns) to keep profiling fast.

### 3.2 Portfolio Decision Rules

RF is always included. LR, XGB, and SVC are subject to the following logic.

**Hard guards** — evaluated first, override all scoring:

| Condition | Portfolio |
|---|---|
| n > 5,000 | `[xgb, rf]` — LR and SVC excluded (kernel SVC cost is O(n²); LinearSVC duplicates LR) |
| n < 300 **or** minority class < 25 samples | `[lr, xgb, rf, svc]` — everything included (small data benefits from diversity) |
| n/p < 1.0 (more features than samples) | `[lr, xgb, rf, svc]` — underdetermined regime favors regularized models |
| p ≥ 2,000 **and** n/p < 5.0 | `[lr, xgb, rf, svc]` — wide feature space with limited data |

**Scoring** — applied when no hard guard fires (300 ≤ n ≤ 5,000):

LR gains points for: n/p < 1.5 (+3), p ≥ 2,000 (+2), binary or sparse-count features (+2), n < 1,000 (+1), imbalance ≥ 20:1 (+1). LR loses points for large dense well-determined data (n ≥ 20,000, n/p ≥ 10, non-sparse, non-binary: −2).

XGB gains points for: n ≥ 5,000 (+3), signal_p90 ≥ 0.15 (+2), mixed feature types (+2), n/p ≥ 5 (+1), mean signal ≥ 0.05 (+1). XGB loses points for very sparse data (sparsity ≥ 0.85 and is_sparse_counts: −2).

SVC gains points for: n < 500 (+3), 500 ≤ n < 2,000 (+2), 2,000 ≤ n ≤ 5,000 (+1), dense continuous features (+2), signal_p90 ≥ 0.1 (+1), n/p < 3 (+1). SVC loses points for sparse fingerprints (−2) or extreme imbalance > 50:1 (−1).

**Final decision**: LR included when XGB does not lead by ≥ 2 points. SVC included when its score ≥ 2. The resulting portfolio is any subset of `[lr, xgb, rf, svc]` with RF always mandatory.

---

## Part 4: Stage 2 — Batch Planning

Before fitting, the training data may be split into batches. Two strategies:

**Balanced strategy** (imbalance ratio ≤ 100):
- If n ≤ 100,000: single batch (all data)
- If n > 100,000: sequential slices of up to 100,000 rows

**Imbalanced strategy** (imbalance ratio > 100):
- All positive samples appear in every batch
- Negative samples are randomly shuffled and partitioned into equal-sized slices
- Each batch = all positives + one negative slice

This guarantees no positive sample is ever excluded from training, while keeping each batch's class ratio manageable. A separate, fully independent `_BatchLazyClassifier` is fitted on each batch's data.

---

## Part 5: Stage 3 — Per-Batch Fitting

The pipeline inside each batch runs: **Preprocessor → Heads → Pooler**.

---

### 5.1 Preprocessing

The preprocessing pipeline is automatically configured from the `PreprocessingProfile`. It runs in a fixed order:

```
MedianImputer → VarianceThreshold(threshold=1e-6) → Scaler → [optional CorrelationFilter]
```

**Step 1 — Median Imputation**: replaces NaN and missing values with the per-feature median. Ensures downstream steps receive no missing values.

**Step 2 — VarianceThreshold**: removes features with near-zero variance (< 1e-6). Constant or near-constant features carry no signal.

**Step 3 — Scaler selection** (rule-based, checked in this order):

| Condition | Chosen scaler | Why |
|---|---|---|
| `is_sparse_counts` (e.g. Morgan fingerprints) | `MaxAbsScaler` | Preserves zero-entry sparsity; scales to [−1, 1] without centering |
| Binary fraction ≥ 0.8 | `MaxAbsScaler` | Centering would destroy the 0/1 semantics |
| Sparsity > 0.5 | `MaxAbsScaler` | Same reason: centering destroys sparsity |
| Outlier fraction > 0.3 | `RobustScaler` | IQR-based scaling is insensitive to extreme values |
| Median skewness > 1.5 | `PowerTransformer` (Yeo-Johnson) | Reduces skewness to make distributions more Gaussian |
| Default | `StandardScaler` | Zero mean, unit variance |

If `PowerTransformer` raises an exception during fitting, the pipeline automatically falls back to `RobustScaler`.

**Step 4 — Reducer selection** (rule-based):

| Condition | Chosen reducer |
|---|---|
| p ≤ 50 **or** n/p ≥ 20 | `VarianceThreshold` only (data is already low-dimensional or well-determined) |
| Otherwise | `VarianceThreshold` + `CorrelationFilter` |

The `CorrelationFilter` is a custom sklearn-compatible transformer that removes one feature from any pair whose absolute Pearson r exceeds 0.90, keeping the higher-variance one. It is designed to be ONNX-serializable, implemented as a `Gather` op over the kept column indices.

The entire preprocessing pipeline is saved to ONNX (opset 15) by default, enabling inference with only `onnxruntime`, with no scikit-learn dependency.

---

### 5.2 Head 1 — Linear Model

The linear head automatically detects one of three **regimes** based on dataset shape and a computational cost proxy (n × p):

| Regime | Triggering condition | Full-data estimator |
|---|---|---|
| `standard` | p ≤ n **and** n×p ≤ 2,000,000 | `LogisticRegressionCV` (saga solver, L1 penalty) |
| `high_dim` | p > n **and** n ≤ 50,000 **and** n×p ≤ 1,000,000 | `SelectKBest(f_classif)` → `LogisticRegressionCV` (elasticnet) |
| `large` | n > 50,000 **or** cost exceeds thresholds | `SGDClassifier` (log loss, elasticnet) |

All hyperparameters are chosen by rules, not search:

- **C grid**: centered on the lasso-theory optimal value √n/p, log-spaced ±2 decades. Grid size adapts to cost: work ≤ 200k → 20 values; work ≤ 2M → 12 values; work > 2M → 8 values.
- **CV folds**: n < 200 → `min(minority_count, 10)`; n < 1,000 → `min(minority_count, 5)`; n ≥ 1,000 → `min(minority_count, 3)`.
- **L1 ratio** (for elasticnet): p/n > 10 → 0.5; p/n > 2 → 0.7; otherwise → 0.9. More features relative to samples → more L2 grouping effect, less L1 sparsity.

**Calibration workflow** (the default, `calibrated=True`):

This is a two-step process:

1. **Full `_fit_raw` on all data** — runs the regime-appropriate estimator with its full CV search, producing the best hyperparameter (best_C or best_alpha).
2. **k-fold OOF pass** — the data is split into k stratified folds. On each fold, a plain `LogisticRegression` (no inner CV) is fitted with the pre-selected best_C. This collects out-of-fold predicted probabilities across all samples.
3. **Calibrator fitted** on OOF scores: isotonic regression if minority count ≥ 500, else Platt scaling (logistic regression on the scalar OOF scores, 2 parameters).
4. **Decision cutoff** learned by exhaustive search over OOF probabilities to maximize balanced accuracy.
5. **ECDF ranker** built from sorted OOF scores (up to 10,000 knots) for `predict_rank()`.

The rationale for step 2 using a plain `LogisticRegression` (not CV) is efficiency: the best hyperparameter was already found in step 1 on all the data, so repeating the search on each fold would be redundant and expensive.

---

### 5.3 Head 2 — XGBoost

The XGBoost head runs a two-phase process. **Fallback**: if n < 200, portfolio comparison is skipped entirely and the heuristic preset is used directly.

#### Phase 1 — Portfolio Comparison (preset selection)

Four preset configurations compete on a single 90/10 stratified split with a capped budget (300 rounds, early stopping patience 30):

1. **`heuristic`**: rule-based parameters derived from the `DatasetProfile` (see below)
2. **`default`**: XGBoost out-of-the-box defaults (lr=0.3, max_depth=6)
3. **`flaml`**: FLAML 1-nearest-neighbour meta-feature portfolio config
4. **`autogluon`**: AutoGluon tabular XGBoost config (lr=0.1, max_depth=6)

When n < 2,000, the ranking is averaged over 3 random splits to reduce noise from small validation sets.

**Noise-aware win threshold**: a non-default preset must beat `default` by at least `max(0.005, 0.3 / √n_minority)` to be declared winner. This prevents preset selection from overfitting on tiny validation folds.

**Best-iteration calibration (Stage 2)**: the winning preset is re-evaluated on 1–3 repeated 90/10 splits with full parameters (no budget cap) to calibrate `best_iteration`. If cost ratio > 15, this stage is skipped and `best_iter ≈ patience × (0.1/lr)` is used as a heuristic.

#### Phase 2 — Full Training

The winning preset is trained on **100% of the training data** for exactly `best_iteration` rounds, with no early stopping. The number of rounds is fixed from Phase 1; the final model sees every training sample.

#### Heuristic Parameter Rules

The `heuristic` preset is the most complex part of the library. Parameters are derived from the `DatasetProfile` using rules from published QSAR and gradient boosting literature. Key decisions:

**Learning rate** — slows down as dataset grows, to exploit better gradient estimates:
- Small sparse fingerprints (n < 1,000): 0.1
- Medium sparse (1,000 ≤ n < 10,000): 0.05
- Large dense: 0.02–0.05 depending on n

**Tree growth strategy** — the most consequential architectural decision:
- For `is_sparse_counts` data (e.g. Morgan fingerprints, p > 200): `grow_policy="lossguide"` (leaf-wise, LightGBM style) with `max_depth=0` (unlimited) and `max_leaves = max(64, min(256, n//10))`. Leaf-wise growth builds asymmetric trees that follow discriminative bit paths, while standard depth-wise trees waste capacity on the ~93% of zero-valued bits.
- For all other data: standard depthwise growth with depth adapted to n/p and feature type.

**Other key parameters**:

| Parameter | Rule |
|---|---|
| `n_estimators` | Fixed at 2,000 (high ceiling); actual rounds come from Phase 1 calibration |
| `early_stopping_rounds` | `max(20, 50 × (0.1/lr))` — scales with 1/lr so slower learners get proportionally more patience |
| `min_child_weight` | `max(3, n//500)` for sparse; halved for imbalance > 10:1. Prevents overfitting on rare structural fragments |
| `subsample` | 1.0 (n < 1k), 0.8 (1k–1M), 0.6 (≥ 1M) |
| `colsample_bynode` | `clamp(1/√p, 0.05, 0.3)` for p > 200 — targets √p features per split (RF-equivalent diversity) |
| `num_parallel_tree` | 3 for 200 ≤ n < 2,000 — mini random forest component for variance reduction |
| `reg_lambda/alpha` | Table keyed on (is_sparse_counts, n/p), hard caps at 4.0/1.5 |
| `max_bin` | 64 for sparse counts (integer values fit exactly); 128 for large/high-dim; 256 default |

**Calibration workflow**: identical to the linear head — full fit → k-fold OOF → calibrator (isotonic or Platt) → decision cutoff → ECDF ranker.

---

### 5.4 Head 3 — Random Forest

The RF head runs a portfolio comparison across four preset configurations using out-of-bag (OOB) scores. **Fallback**: if n < 200, portfolio comparison is skipped and the `default` preset is used directly (OOB estimates are unreliable on tiny datasets).

#### Presets

1. **`heuristic`**: rule-based parameters derived from an `RFProfile` (see literature: Breiman 2001, Probst et al. 2019, Sheridan 2016)
2. **`default`**: sklearn `RandomForestClassifier` out-of-the-box (100 trees, `max_features="sqrt"`, no depth limit)
3. **`flaml`**: 1-nearest-neighbour meta-feature matching against FLAML's RF portfolio (Wang et al. 2021). Skipped when p > 200 (FLAML was calibrated on low-dimensional datasets)
4. **`autogluon`**: AutoGluon zeroshot 2023 config, selected from a 3×2 grid of (size × sparsity/binary). Skipped when p > 200 for the same reason

All four presets are trained simultaneously on the full dataset with `oob_score=True`. The winner is chosen by OOB AUC with the same noise-aware margin formula as XGBoost: `max(0.005, coef / √n_minority)` where `coef = 0.3` for n < 2,000 else `0.1`. The default preset's model is always kept in memory so the winner can be switched without refitting.

#### Heuristic Parameter Rules

Key decisions from `heuristic_rf_params(profile)`:

| Parameter | Rule |
|---|---|
| `n_estimators` | 100 (n < 500), 200 (n < 5,000), 300 (n ≥ 5,000) |
| `max_features` | `"log2"` for sparse-count data with p > 500 (forces stronger tree diversity on fingerprints); `"sqrt"` otherwise |
| `max_depth` | `None` for dense well-determined data; 12–20 for sparse or underdetermined (n/p < 3) to control ONNX size |
| `min_samples_leaf` | `max(3, n//500)` for sparse-count, `max(1, n//1000)` for dense; halved when imbalance > 10:1 |

#### ONNX size control

The `rf_leaf_cap` function targets < 4 MB per model: `floor(100,000 / n_estimators / 2)` total leaf nodes per tree, clipped to `[32, n//3]`. Applied only to `flaml` and `autogluon` presets (which use `max_leaf_nodes`); the `heuristic` and `default` presets use `max_depth` for size control instead.

#### Calibration workflow

Identical to the linear and XGBoost heads: full fit (winner preset on all data) → k-fold OOF with separate fold-level forest instances → isotonic/Platt calibration → decision cutoff → ECDF ranker. `class_weight` automatically switches from `"balanced"` to `"balanced_subsample"` when imbalance ratio ≥ 3.0.

---

### 5.5 Head 4 — Support Vector Classifier

The SVC head runs a portfolio comparison across four preset configurations on a 90/10 validation split. **Fallback**: if n < 200, the `heuristic` preset is used directly.

#### Presets

1. **`heuristic`**: rule-based parameters from `get_params(profile)` (see literature: Burbidge et al. 2001, Heikamp & Bajorath 2014, Sheridan 2016)
2. **`default`**: sklearn SVC out-of-the-box (C=1.0, RBF, `gamma='scale'`)
3. **`linear`**: `LinearSVC` always (compact ONNX, state-of-the-art for bit fingerprints)
4. **`balanced_rbf`**: RBF + `class_weight='balanced'` + C scaled by √n_minority — suited to imbalanced data

Presets are evaluated by AUC on the 90/10 split using `decision_function()` scores (not probabilities). A preset is skipped if its estimated training cost exceeds 20× the default's cost (kernel SVC cost ∝ n²×p; LinearSVC cost ∝ n×p). The winner must beat default by at least `max(0.005, 0.3/√n_minority)`.

#### Heuristic Parameter Rules

**Kernel selection**:
- `is_sparse_counts` or n > 5,000 → `LinearSVC` (linear kernel; Tanimoto dot product is linear in binary fingerprint space)
- Dense data with n ≤ 5,000 → `SVC(kernel='rbf')` (captures non-linear structure in continuous descriptor spaces)

**C for linear kernel**: 0.1 (n < 500), 1.0 (500 ≤ n < 2,000), 10.0 (n ≥ 2,000)

**C for RBF kernel**: base from n/p ratio (1.0 / 10.0 / 100.0), scaled by `min(1.0, √(n/1000))`, cap 100.0

All presets use `class_weight='balanced'` and `gamma='scale'`.

#### ONNX size guard

After phase-2 (full-data) refit, kernel SVC models are checked: if `n_sv × n_features × 4 > 5 MB`, C is halved and the model is refit, up to 3 retries. If still over budget, the model falls back to joblib. `LinearSVC` is always compact (weights + bias only; O(n_features)).

#### ONNX inference

The ONNX model outputs raw decision function scores (not probabilities). `BaseSVCArtifact.run()` applies `sigmoid(score)` → calibrator → final calibrated probabilities. Calibration workflow is identical to other heads: OOF decision scores → isotonic (minority ≥ 500) or Platt → decision cutoff → ECDF ranker.

---

### 5.6 Pooling (Combining the Heads)

After all heads are fitted, the `InnerClassifierPooler` learns how to combine their predictions. It operates on the OOF predicted probabilities collected during each head's calibration step.

**The pooler has three modes**:

**`passthrough`** — triggered when only one head is in the portfolio. The single head's output is returned directly with no transformation.

**`equal`** — triggered when multiple heads are present but OOF data is unavailable (e.g. calibration was disabled, or a head had too few samples). All head outputs are averaged with uniform weights.

**`gating`** — triggered when multiple heads are present and OOF data is available. A Ridge regression is fitted per head on the preprocessed feature matrix X_prep, learning **per-sample, feature-conditional weights**. The regularization alpha is adaptive: `mean_feature_variance × p / n`, floored at 0.01.

The gating target for head j on sample i is:

```
target_w[i,j] = softmax( log P(y_i | head_j) )  across j
```

This assigns high weight to the head that was most confident and correct on that sample. At inference, per-sample weights are computed as a linear projection of the features through the Ridge coefficients, then softmax-normalized across heads. The pooler therefore learns which compounds each head is best suited for, rather than applying a single global weight.

The composite scoring metric used to evaluate OOF quality is a weighted sum of AUROC, AUPR, and BEDROC — the last of which up-weights correct predictions at the top of the ranked list, making it sensitive to virtual screening performance.

---

## Part 6: Stage 4 — Aggregation across Batches

At inference time, each batch model produces its own `predict_proba()`. Before averaging, a **prior correction** is applied per batch:

```
corrected_odds = (population_prior / train_prior) / ((1 - population_prior) / (1 - train_prior)) × odds
```

This is a Bayes odds-ratio adjustment. If a batch's training positive fraction (`train_prior`) differs from the overall dataset's positive fraction (`population_prior`), the model's calibrated probabilities are shifted to reflect the true population base rate. The final output is the mean of all prior-corrected batch probabilities.

---

## Part 7: Save and Load Format

The on-disk structure depends on which entry point was used to train the model.

### Descriptor-agnostic mode (`LazyClassifier`)

```
model_dir/
  metadata.json
  batch_0/
    preprocessor.onnx
    preprocessor.json
    xgboost.onnx          <- only if xgb in portfolio
    xgboost.json          <- only if xgb in portfolio
    linear.onnx           <- only if lr in portfolio
    linear.json           <- only if lr in portfolio
    randomforest.onnx     <- always present (rf always in portfolio)
    randomforest.json     <- always present
    svc.onnx              <- only if svc in portfolio
    svc.json              <- only if svc in portfolio
    pooler.json
  batch_1/
    ...
```

**`metadata.json`** — top-level manifest written by `LazyClassifier.save()`:
- `portfolio`: list of head names included, e.g. `["xgb", "rf"]`
- `num_batches`: number of batch subdirectories
- `max_batch_size`: the batch size cap used during training
- `max_imbalance_ratio`: the imbalance ratio threshold used for batch planning
- `population_prior`: overall fraction of positives in the training set
- `batch_priors`: per-batch fraction of positives (used for prior correction at inference)
- `decision_cutoff`: OOF-learned decision threshold averaged across batches

**`preprocessor.json`** — written by `BasePreprocessor.save()`:
- `task`, `scaler`, `reducer`
- `n_features_in`, `n_features_out`
- `kept_feature_indices`: column indices surviving the preprocessing pipeline

**`xgboost.json` / `linear.json` / `randomforest.json` / `svc.json`** — written by each head's `save()`:
- `task`, `n_features_in`, `decision_cutoff`, `decision_cutoff_source`
- `calibrator`: isotonic or Platt calibrator parameters (if calibration was run)
- `ranker`: sorted OOF score knots for `predict_rank()` (if calibration was run)

**`pooler.json`** — written by `InnerClassifierPooler.save()`:
- `portfolio`: head name list
- `gating_coef`, `gating_intercept`: Ridge weights per head (present only when gating mode is active)
- `score`: composite OOF score of the gated ensemble (present only when gating mode is active)

Note: `mode` is **not serialised**. It is inferred at load time: if `gating_coef` is present → gating; otherwise → equal. Passthrough is determined at runtime by portfolio size.

---

### SMILES-aware mode (`LazyClassifierQSAR`)

Each descriptor type gets its own subdirectory, named after the descriptor. Inside each subdirectory sits a complete `LazyClassifier` structure (identical to the agnostic layout above) plus the descriptor's own saved files.

```
model_dir/
  rdkit/                        <- one subdirectory per descriptor type
    featurizer.json             <- featurizer name + RDKit version (for compatibility check)
    metadata.json               <- LazyClassifier manifest for this descriptor
    batch_0/
      preprocessor.onnx + .json
      xgboost.onnx + .json
      randomforest.onnx + .json
      pooler.json
    batch_1/
      ...
  morgan/
    featurizer.json
    metadata.json
    batch_0/
      ...
  chemeleon/
    featurizer.json             <- plus additional DL checkpoint files
    metadata.json
    batch_0/
      ...
```

The subdirectory names match the descriptor keys in `DESCRIPTOR_TYPES` (`"rdkit"`, `"morgan"`, `"chemeleon"`, `"cddd"`). At load time, `LazyClassifierQSAR.load()` scans the top-level directory for any subdirectory whose name matches a known descriptor key, reconstructs each descriptor and its `LazyClassifier`, and infers the original `mode` from the set of descriptor names found.

**`featurizer.json`** is the descriptor's own save file. For RDKit-based descriptors (`rdkit`, `morgan`) it contains only the featurizer name and the RDKit version used during training. No weights are saved because these descriptors are purely algorithmic — given the same SMILES and RDKit version, output is deterministic. The stored version is checked against the current environment at load time; a mismatch raises an error because fingerprint bit assignments can shift across RDKit releases. For deep-learning descriptors (`chemeleon`, `cddd`), additional checkpoint files are also present in the same subdirectory.

---

### ONNX vs Joblib

All models default to **ONNX format**. ONNX artifacts require only `onnxruntime` at inference time — scikit-learn and XGBoost are not needed. Joblib is available as a fallback when ONNX conversion fails. The `CorrelationFilter` in the preprocessing pipeline is exported via a custom `Gather` ONNX op; if the skl2onnx version changes incompatibly, this component is the most likely source of conversion failure.

ONNX opset versions vary by component: preprocessing and SVC use opset 15; XGBoost, RF, and linear use library-default opsets (via `onnxmltools`/`skl2onnx`); applicability domain uses opset 13 / IR version 7.

---

## Part 8: Prediction Methods

| Method | Output shape | What it represents |
|---|---|---|
| `predict_proba()` | (n, 2) | Calibrated P(y=0) and P(y=1), prior-corrected and batch-averaged |
| `predict()` | (n,) | Binary label using the OOF-learned decision cutoff (not 0.5) |
| `predict_score()` | (n, 2) | Raw pre-calibration scores; no prior correction applied |
| `predict_rank()` | (n, 2) | Percentile rank within the training OOF score distribution, in [0, 1] |
| `predict_logit()` | (n, 2) | Log-odds derived from `predict_proba()` |
| `predict_lift()` | (n, 2) | `predict_proba()[:,1] / population_prior` — enrichment over random baseline |

---

## Part 9: Applicability Domain

The `ApplicabilityDomain` class estimates whether a query compound falls within the distribution of the training set. It is designed to work on any feature matrix without requiring re-training alongside the classifier.

### How it works

1. **StandardScaler** normalises the input (applied independently of the classifier's preprocessor)
2. **PCA** projects to at most `min(max(1, int(√p)), 100, p, n-1)` components, removing redundant and zero-variance directions. This is essential for fingerprints (p=2048) where the raw covariance matrix is huge and singular
3. **Tikhonov-regularised Mahalanobis distance** is computed in the PCA subspace for each training sample and each query
4. **Empirical survival function**: the score is `P(d_train > d(query))` — the fraction of training samples further from the centroid than the query

**Score interpretation**:
- `1.0` = query is closer to the training centroid than every training sample (fully in-domain)
- `0.0` = query is further than every training sample (completely out-of-distribution)

The survival function is stored as 200 quantile knots and evaluated at inference via a vectorised ONNX graph built from primitive ops (`Sub`, `MatMul`, `Mul`, `Relu`, `Sqrt`, `Reshape`, `Less`, `Cast`, `Div`) — no custom extensions, no sklearn dependency at inference time.

---

## Part 10: Descriptor Portfolio Selection

When using SMILES mode (`LazyClassifierQSAR`), the `DescriptorPortfolio` class gates which descriptor types enter the ensemble before any `LazyClassifier` is trained. At most `_MAX_DESCRIPTORS = 3` descriptors are kept.

### Selection procedure

**Step 1 — Unsupervised applicability gate**: each descriptor's `is_applicable(smiles_list)` method is called. Descriptors that fail (too many out-of-domain SMILES) are dropped immediately without computing features.

**Step 2 — Supervised greedy forward selection** (only when `y` is provided and > 1 descriptor passes Step 1): A lightweight 3-fold Random Forest CV (proxy) is run per descriptor to estimate solo OOF AUC. Descriptors are then selected greedily starting from a reference anchor (`morgan` in fast mode, `chemeleon` in slow mode), adding the descriptor that improves ensemble AUC most at each step, up to `_MAX_DESCRIPTORS`.

### Ensemble weighting

The final prediction in `LazyClassifierQSAR` is not a simple average. Each selected descriptor's `LazyClassifier` contributes a weight combining:
- **Global skill score**: derived from OOF AUC and proxy AUC
- **Per-sample reliability score**: derived from rank-based prediction confidence (rank→error curves)

The two scores are multiplied and softmax-normalised across descriptors to produce per-sample, per-descriptor weights.

---

## Part 11: Key Points for Interpreting Results

### For users

**The decision threshold in `predict()` is not 0.5.** It is the value that maximises balanced accuracy on OOF predictions from the training set. On imbalanced datasets this threshold can be substantially below 0.5. For any ranking task or threshold sweep, use `predict_proba()[:,1]` directly and define your own cutoff externally.

**In SMILES mode, each descriptor type is a completely separate model.** The final prediction is an AUC-weighted average of all descriptor-type predictions. Using `mode="slow"` (5 descriptors) means 5 full training runs, 5 full ensembles, and 5 separate prediction outputs combined via weighted averaging. It does not mean a 5× larger feature matrix.

**`predict_rank()` is relative to the training set, not absolute.** A rank of 0.9 means the compound scored higher than 90% of the training OOF samples. It is not a probability and is not comparable across models trained on different datasets.

**`predict_lift()` is relative to the training prior.** A lift of 5 on a dataset with 10% positives means the model assigns that compound a 50% chance of being active. It is only meaningful in the context of the training set's base rate.

**Prior correction is automatic.** When the batch-level training positive fraction differs from the overall population prior, `predict_proba()` already adjusts for this. Applying additional prior correction externally will double-count the adjustment.

### For developers

**n/p is the single most decisive quantity in the entire pipeline.** It simultaneously drives: scaler selection (MaxAbs vs Standard vs Power), reducer selection (VarianceThreshold only vs CorrelationFilter), portfolio selection (whether LR and SVC are included), linear regime detection (standard vs high_dim vs large), XGBoost regularization strength (lambda/alpha table), XGBoost tree depth, SVC kernel choice (RBF vs linear), and SVC C value. A dataset with 200 compounds and 2048 Morgan bits (n/p ≈ 0.10) follows an entirely different code path than one with 5,000 compounds and 200 RDKit descriptors (n/p = 25).

**`is_sparse_counts` triggers different logic throughout the pipeline.** It affects scaler choice (MaxAbsScaler), XGBoost grow policy (lossguide vs depthwise), `max_bin` (64 vs 256), `min_child_weight` formula, L1/L2 regularization table, `colsample_bytree` floor, and gamma value. The detection rule differs slightly by profiler: `DatasetProfile` (used by portfolio/XGBoost/SVC) uses `sparsity ≥ 0.5`, `95%+ integer-like values`, and `sparsity ≥ 0.85 or max_val ≤ 10`; `PreprocessingProfile` (used by scaler selection) uses the same sparsity and integer-like checks but applies `median_val ≤ 5` instead of `max_val ≤ 10`.

**The three profilers are independent and compute overlapping but different statistics.** `DatasetProfile` (XGBoost/SVC/Portfolio inspector) includes mean feature–target signal (`feature_signal_strength`) but lacks skewness, outlier fraction, and correlation statistics. `PreprocessingProfile` includes those but lacks `feature_signal_strength`. `RFProfile` is the lightest — only n, p, sparsity, `is_sparse_counts`, imbalance, and `pct_numeric` (no signal statistics). All three are run independently at different stages of the pipeline.

**In linear calibration, the hyperparameter search runs exactly once** on the full data in `_fit_raw`. The subsequent OOF folds each fit a plain `LogisticRegression` (no inner CV) using the best_C found in that full-data search. This is intentional — repeating the C-search on each fold would be correct in theory but prohibitively expensive in practice, and the bias introduced by fixing C from full-data is small.

**The XGBoost gating pooler only activates when all heads have `oof_probas_`**, which requires `calibrated=True` and sufficient data for OOF folds in every head. If any head was fitted without calibration or with too few samples to run OOF, the pooler falls back to equal weighting.

**Phase 2 of XGBoost uses 100% of the data with a fixed round count.** There is no validation set in Phase 2 and no early stopping. The round count `best_iteration` is fully determined by Phase 1 on the 90/10 split. Overfitting in Phase 2 is controlled by the regularization parameters, not by validation-based stopping.

**The calibration threshold (isotonic vs Platt) is 500 minority samples**, applied uniformly across all four heads (linear, XGB, RF, SVC). Below 500, Platt scaling (logistic regression on the scalar OOF scores, 2 free parameters) is used. Above 500, isotonic regression (non-parametric, potentially many parameters) is used. A dataset sitting just at this boundary will behave differently depending on which side it falls on.

**The `lossguide` growth policy** (leaf-wise, LightGBM-style) is activated only for `is_sparse_counts=True` and p > 200. It builds asymmetric trees that descend deeper along high-signal bit paths without wasting node capacity on near-constant zero-valued bits. Datasets with continuous descriptors or embeddings — even if high-dimensional — use standard depthwise growth.

---

## Part 12: Dependency Structure

The library is split into three installation profiles:

| Profile | Dependencies | Use case |
|---|---|---|
| Base (default) | numpy, onnxruntime, pandas, h5py, rich, loguru | Inference only from ONNX artifacts |
| `[fit]` | + scikit-learn, xgboost, scipy, skl2onnx | Model training |
| `[descriptors]` | + rdkit, FPSim2 | SMILES featurization |

A model saved as ONNX can be loaded and used for prediction in an environment with only the base dependencies installed — no scikit-learn, no XGBoost, no training dependencies required.

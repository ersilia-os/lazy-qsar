# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Collaboration style

Always use the `AskUserQuestion` tool when facing design decisions or ambiguous requirements — do not assume. The user prefers to be consulted before implementation choices are made.

## Commands

```bash
# Install for development (training + fitting)
pip install -e .[fit]

# Install everything (includes molecular descriptors: rdkit, FPSim2)
pip install -e .[all]

# Run full test suite
pytest

# Run a single test file
pytest dev/tests/test_classifier_unit.py -v

# Quick smoke tests (validates pipeline end-to-end with synthetic data)
python dev/smoke/smoke_classifier.py           # 500 samples, 100 features
python dev/smoke/smoke_classifier.py 1000 200  # custom size
```

## Architecture

The library has two entry points:

- **`LazyClassifierQSAR`** (`lazyqsar/qsar.py`) — takes raw SMILES strings, computes molecular descriptors internally, trains an ensemble, exports to ONNX.
- **`LazyClassifier`** (`lazyqsar/agnostic.py`) — takes pre-computed feature arrays or HDF5 files; descriptor-agnostic.

`LazyClassifierQSAR` wraps multiple `LazyClassifier` instances (one per descriptor type) and combines their predictions via an AUC-weighted ensemble. Descriptor mode is selected at init: `fast` (Morgan fingerprints only), `slow` (CDDD, Chemeleon, CLAMP, Morgan, RDKit).

### Training pipeline (inside `LazyClassifier`)

```
LazyClassifier.fit()
  └─ _BatchLazyClassifier.fit()   [assemblers/classifier.py]
       ├─ Preprocessor.fit()      [preprocessors/classification/prep.py]
       │    └─ scaler + optional feature reducer (auto-selected from dataset profile)
       ├─ Portfolio selection     [portfolios/classification/portfolio.py]
       │    └─ decides which heads (lr / xgb / rf) to train based on n, p, sparsity
       ├─ Head training (parallel)
       │    ├─ BaseLinearClassifier   [base/linear/model.py]
       │    ├─ BaseXGBClassifier      [base/xgboost/model.py]
       │    └─ BaseRFClassifier       [base/randomforest/model.py]
       └─ InnerClassifierPooler.fit() [poolers/classification/inner_pooler.py]
            └─ RidgeCV gating network: learns per-sample head weights from OOF predictions
```

For severely imbalanced data (ratio > 100), `LazyClassifier` splits into balanced batches and trains a separate `_BatchLazyClassifier` per batch, then averages predictions at inference.

### Inference (ONNX artifact path)

All models export to ONNX at save time. At inference, only `numpy` + `onnxruntime` are needed — no sklearn, xgboost, or scipy. Artifact classes live in `lazyqsar/artifacts/`.

### Base models

The four base classifiers (`base/linear/`, `base/xgboost/`, `base/randomforest/`, `base/preprocessing/`) are self-contained and can be used independently. Each:
- Auto-selects hyperparameters from dataset statistics (zero-shot, no grid search)
- Runs an OOB/validation portfolio comparison between a data-driven heuristic preset and sklearn defaults
- Calibrates probabilities via OOF isotonic regression or Platt scaling
- Exports to ONNX via `save(directory)`

### Pooler

`InnerClassifierPooler` (`poolers/classification/inner_pooler.py`) learns a per-sample gating weight matrix `(n, n_heads)` using one `RidgeCV` per head. The oracle target is `softmax(log P(y_i | head_j))` — how well each head explains each training sample. At inference, `get_weights(X_prep)` returns the weight matrix; `predict_proba(R, X_prep)` returns the weighted ensemble prediction.

## Key design decisions

- **Zero-shot hyperparameters**: no CV grid search during training. Parameters are chosen from dataset profile statistics (n, p, sparsity, imbalance ratio) via rule-based presets in `base/*/params.py`.
- **ONNX-first**: training requires `[fit]` extras; inference requires only `onnxruntime`. The `Artifact` classes are the production inference path.
- **Calibration**: all classifiers calibrate OOF probabilities before the pooler sees them. Decision cutoffs are learned from balanced accuracy on OOF data, not hardcoded at 0.5.

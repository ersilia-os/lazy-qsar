# Ersilia's LazyQSAR

A Python library for building supervised QSAR (Quantitative Structure-Activity Relationship) models quickly, with minimal configuration. LazyQSAR automates descriptor computation, feature preprocessing, and model selection to produce robust ensemble models from chemical structures.

**Two entry points:**
- **`LazyClassifierQSAR`** — pass SMILES strings directly; built-in descriptors are computed automatically
- **`LazyClassifier`** — bring your own pre-computed descriptor arrays or HDF5 files

## Table of Contents

- [Installation](#installation)
- [Python API](#python-api)
  - [LazyClassifierQSAR (SMILES)](#lazyclassifierqsar-smiles)
  - [LazyClassifier (custom descriptors)](#lazyclassifier-custom-descriptors)
  - [Saving and loading](#saving-and-loading)
- [CLI](#cli)
- [How It Works](#how-it-works)
- [Base Models](#base-models)
- [Ersilia Model Hub integration](#ersilia-model-hub-integration)
- [Disclaimer](#disclaimer)

## Installation

Install from source:

```bash
git clone https://github.com/ersilia-os/lazy-qsar.git
cd lazy-qsar
pip install -e .
```

The base install includes only lightweight runtime dependencies (`numpy`, `onnxruntime`, etc.) — sufficient for loading and running pre-trained ONNX models without any ML packages.

Install optional extras depending on your use case:

| Extra | Command | Adds |
|-------|---------|------|
| `fit` | `pip install -e .[fit]` | scikit-learn, XGBoost, scipy, skl2onnx — required to train models |
| `descriptors` | `pip install -e .[descriptors]` | RDKit, FPSim2 — required for built-in molecular descriptors |
| `all` | `pip install -e .[all]` | Everything above |

The first time you use deep-learning descriptors (CDDD, Chemeleon, CLAMP), their checkpoints are downloaded automatically. To do this in advance:

```bash
lazyqsar setup --descriptors
```

## Python API

### LazyClassifierQSAR (SMILES)

Pass SMILES strings directly. Choose a descriptor mode:

| Mode | Descriptors | Notes |
|------|-------------|-------|
| `fast` | Morgan fingerprints | No deep-learning models, fastest |
| `slow` | CDDD, Chemeleon, CLAMP, Morgan, RDKit | Most thorough |

```python
from lazyqsar.qsar import LazyClassifierQSAR

model = LazyClassifierQSAR(mode="slow")
model.fit(smiles_list=smiles_train, y=y_train)
```

Available prediction methods:

| Method | Returns | Description |
|--------|---------|-------------|
| `predict_proba(smiles_list)` | `(N, 2)` | Calibrated class probabilities |
| `predict(smiles_list)` | `(N,)` | Binary labels at threshold 0.5 |
| `predict_logit(smiles_list)` | `(N, 2)` | Log-odds scores |
| `predict_rank(smiles_list)` | `(N, 2)` | Rank quantiles (0–1) |
| `predict_score(smiles_list)` | `(N, 2)` | Raw model scores |
| `predict_lift(smiles_list)` | `(N, 2)` | Probability / population prior |

### LazyClassifier (custom descriptors)

Pass your own descriptor arrays or HDF5 files. We recommend the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia) for descriptor computation — its `.h5` output format is supported natively.

```python
from lazyqsar.agnostic import LazyClassifier

# From a NumPy array
model = LazyClassifier()
model.fit(X=X_train, y=y_train)
y_hat = model.predict_proba(X=X_test)[:, 1]

# From an Ersilia .h5 file
model.fit(h5_file="descriptors.h5", y=y_train)
y_hat = model.predict_proba(h5_file="descriptors.h5")[:, 1]
```

The same prediction methods listed above are available, using `X=` instead of `smiles_list=`.

### Saving and loading

Models are saved as ONNX files, so inference only requires `numpy` and `onnxruntime` — no scikit-learn or XGBoost at prediction time.

```python
model.save(model_dir)          # directory
model.save("my_model.zip")     # or zip archive
```

```python
model = LazyClassifierQSAR.load(model_dir)
y_hat = model.predict_proba(smiles_list=smiles_test)[:, 1]

model = LazyClassifier.load(model_dir)
y_hat = model.predict_proba(X=X_test)[:, 1]
```

## CLI

All commands are available through the `lazyqsar` entry point.

**Setup:**

```bash
lazyqsar setup --fit          # sklearn, xgboost, scipy, skl2onnx, onnxmltools
lazyqsar setup --descriptors  # rdkit, FPSim2, Chemeleon / CDDD / CLAMP checkpoints
lazyqsar setup --fit --descriptors
```

**Fit:**

The `--input` directory must contain one CSV per task, with SMILES in the first column and binary labels (0/1) in the second column, with a header row.

```bash
lazyqsar fit --task classification --input $DATA_DIR --output $MODEL_DIR --mode slow
```

Pass `--models_txt` to train a subset of tasks (one CSV stem per line); without it, all CSVs in the directory are used.

**Predict:**

```bash
lazyqsar predict --input $INPUT_CSV --model $MODEL_DIR --output $OUTPUT_CSV
```

The output CSV contains one predicted probability column per task, ordered alphabetically by task name.

## How It Works

LazyQSAR builds an ensemble for each descriptor set through four steps:

1. **Portfolio selection** — the dataset is profiled (sample count, dimensionality, sparsity, class imbalance) and a rule-based selector decides which heads to train. The default portfolio is XGBoost + Random Forest; Logistic Regression is added automatically for small, high-dimensional, or low-prevalence datasets.

2. **Preprocessing** — a scaler (`StandardScaler`, `RobustScaler`, `MaxAbsScaler`, or `PowerTransformer`) and an optional correlation-based feature reducer are selected automatically from dataset statistics.

3. **Heads** — each selected head (Logistic Regression, XGBoost, Random Forest) is fitted on preprocessed features. For severely imbalanced datasets, balanced sub-batches are used and the batch predictions are averaged.

4. **Pooling and export** — head predictions are combined via a learned gating network (`InnerClassifierPooler`). The full pipeline is exported to ONNX for dependency-free inference.

When using `LazyClassifierQSAR`, a separate ensemble is trained per descriptor type and their predictions are combined via an AUC-weighted ensemble that accounts for per-sample prediction confidence.

## Base Models

The components under `lazyqsar/base/` can be used independently of the full pipeline:

| Module | Description |
|--------|-------------|
| [`lazyqsar.base.preprocessing`](lazyqsar/base/preprocessing/) | Automatic scaler and feature reducer selection |
| [`lazyqsar.base.xgboost`](lazyqsar/base/xgboost/README.md) | Automatic XGBoost hyperparameter selection with portfolio comparison |
| [`lazyqsar.base.linear`](lazyqsar/base/linear/README.md) | Automatic linear model selection (logistic/ridge/SGD) |
| [`lazyqsar.base.randomforest`](lazyqsar/base/randomforest/README.md) | Random Forest classifier with zero-shot hyperparameter selection |

## Ersilia Model Hub integration

LazyQSAR models can be used inside an [Ersilia Model Hub template](https://github.com/ersilia-os/eos-template). See [eos1lb5](https://github.com/ersilia-os/eos1lb5) for an example.

`lazyqsar fit` produces a `checkpoints` folder with one sub-directory per task and per descriptor type:

```text
checkpoints/
└── task1/
    ├── cddd/
    │   ├── featurizer.json
    │   ├── metadata.json
    │   └── batch_0/
    │       ├── preprocessor.onnx
    │       ├── xgboost.onnx
    │       └── pooler.json
    ├── chemeleon/   (same structure)
    ├── clamp/       (same structure)
    ├── morgan/      (same structure)
    └── rdkit/       (same structure)
```

`fast` mode produces only a `morgan/` subdirectory per task.

The `code/main.py` inference script:

```python
import os, sys
from lazyqsar.api.classifier_predict import predict

root = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.abspath(os.path.join(root, "..", "checkpoints"))
predict(model_dir=checkpoints_dir, input_csv=sys.argv[1], output_csv=sys.argv[2])
```

## Disclaimer

This library is intended for quick QSAR modeling. For a more complete automated QSAR pipeline, refer to [Zaira Chem](https://github.com/ersilia-os/zaira-chem).

Learn about the [Ersilia Open Source Initiative](https://ersilia.io).

# Ersilia's LazyQSAR

A Python library for building supervised QSAR (Quantitative Structure-Activity Relationship) models quickly, with minimal configuration. LazyQSAR automates descriptor computation, feature preprocessing, and model selection to produce robust ensemble models from chemical structures.

**Two usage modes:**
- **SMILES-based:** pass molecule SMILES strings directly; built-in descriptors are computed automatically
- **Descriptor-agnostic:** bring your own pre-computed descriptor arrays or HDF5 files

## Table of Contents

- [Installation](#installation)
- [Python API](#use-as-a-python-api)
  - [Binary Classification (SMILES)](#binary-classification)
  - [Binary Classification (Custom Descriptors)](#custom-descriptors)
  - [Saving and Loading Models](#saving-and-loading-models)
  - [Tests and Benchmarks](#tests-and-benchmarks)
- [CLI](#use-as-a-cli)
- [Base Models](#base-models)
- [How It Works](#how-it-works)
- [Use in an Ersilia Model Hub template](#use-in-an-ersilia-model-hub-template)
- [Disclaimer](#disclaimer)
- [About Us](#about-us)

## Installation

Install LazyQSAR from source:

```bash
git clone https://github.com/ersilia-os/lazy-qsar.git
cd lazy-qsar
python -m pip install -e .
```

The base install (`pip install -e .`) includes only lightweight runtime dependencies (`numpy`, `onnxruntime`, etc.) and is sufficient for **loading and running pre-trained ONNX models** without any ML packages.

Install optional extras depending on your use case:

| Extra | Command | Adds |
|-------|---------|------|
| `fit` | `pip install -e .[fit]` | scikit-learn, XGBoost, scipy, skl2onnx — required to train models |
| `descriptors` | `pip install -e .[descriptors]` | RDKit, FPSim2 — required to compute built-in molecular descriptors |
| `all` | `pip install -e .[all]` | Everything above |

The first time you run `LazyClassifierQSAR` with deep-learning descriptors, it will download the Chemeleon and CDDD model checkpoints. To complete this setup in advance, run:

```bash
lazyqsar setup --descriptors
```

## Use as a Python API

### Binary Classification

#### Built-in descriptors

Instantiate `LazyClassifierQSAR` with a mode of choice:

| Mode | Descriptors used | Speed |
|------|-----------------|-------|
| `fast` | Morgan fingerprints, RDKit | Fastest, no deep-learning descriptors |
| `default` | CDDD, Chemeleon, RDKit | Balanced |
| `slow` | CDDD, Chemeleon, Morgan, RDKit | Most thorough |

```python
from lazyqsar.qsar import LazyClassifierQSAR

model = LazyClassifierQSAR(mode="default")
model.fit(smiles_list=smiles_train, y=y_train)
y_hat = model.predict_proba(smiles_list=smiles_test)[:, 1]
```

#### Custom descriptors

Pre-calculate your own descriptors and pass them directly. We recommend the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia) for this — its `.h5` output format is supported natively. Alternatively, pass descriptors as a NumPy array.

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

### Saving and loading models

Models are saved as ONNX files by default, so inference only requires the ONNX runtime (no scikit-learn or XGBoost dependency at prediction time).

```python
# Save after training
model.save(model_dir)

# Load for inference
from lazyqsar.agnostic import LazyClassifier

model = LazyClassifier.load(model_dir)
y_hat = model.predict_proba(X=X)[:, 1]
```

You can also save and load as a `.zip` archive:

```python
model.save("my_model.zip")
model = LazyClassifier.load("my_model.zip")
```

The same save/load interface applies to `LazyClassifierQSAR`:

```python
from lazyqsar.qsar import LazyClassifierQSAR

model = LazyClassifierQSAR(mode="default")
model.fit(smiles_list=smiles_train, y=y_train)
model.save(model_dir)

model = LazyClassifierQSAR.load(model_dir)
y_hat = model.predict_proba(smiles_list=smiles_test)[:, 1]
```

### Tests and Benchmarks

#### Running tests

Run the full test suite with pytest:

```bash
pytest tests/
```

The `tests/` folder also contains an integration script that can be run directly:

```bash
python tests/test_binary_classification.py
python tests/test_binary_classification.py --agnostic
```

Additional flags:

| Flag | Description |
|------|-------------|
| `--mode fast\|default\|slow` | Select descriptor mode |
| `--agnostic` | Use descriptor-agnostic `LazyClassifier` |
| `--no-onnx` | Skip ONNX conversion |
| `--no-zip` | Skip ZIP archive save/load |
| `--clean` | Remove temporary files after the run |

#### Benchmarking

The [benchmark repository](https://github.com/ersilia-os/zaira-chem-tdc-benchmark) contains performance results for the default estimators and descriptors on the TDCommons ADMET dataset.

## Use as a CLI

All commands are available through the single `lazyqsar` entry point.

**Setup:**

Install optional dependencies after the base `pip install`:

```bash
lazyqsar setup --fit          # sklearn, xgboost, scipy, skl2onnx, onnxmltools
lazyqsar setup --descriptors  # rdkit, FPSim2 + downloads Chemeleon / CDDD checkpoints
lazyqsar setup --fit --descriptors  # both
```

**Fit:**

The `--input` directory must contain one CSV per task. Each CSV must have SMILES in the first column and binary labels (0/1) in the second column, with a header row.

```bash
lazyqsar fit --task classification --input $DATA_DIR --output $MODEL_DIR --mode default
```

Optionally pass `--models_txt` listing task names (CSV stems) to train, one per line. Without it, all CSVs in the directory are used.

```bash
lazyqsar fit --task classification --input $DATA_DIR --output $MODEL_DIR --models_txt models.txt
```

**Predict:**

```bash
lazyqsar predict --input $INPUT_CSV --model $MODEL_DIR --output $OUTPUT_CSV
```

The output CSV contains one predicted probability column per task. Optionally use `--models_txt` to predict only a subset of tasks.

## Base Models

LazyQSAR bundles three self-contained ML components under `lazyqsar/base/`. Each can be used independently of the QSAR pipeline:

| Module | Description | README |
|--------|-------------|--------|
| `lazyqsar.base.preprocessing` | Automatic scaler and feature reducer selection | [base/preprocessing](lazyqsar/base/preprocessing/README.md) |
| `lazyqsar.base.xgboost` | Automatic XGBoost hyperparameter selection with portfolio comparison | [base/xgboost](lazyqsar/base/xgboost/README.md) |
| `lazyqsar.base.linear` | Automatic linear model and feature selection (logistic/ridge/SGD) | [base/linear](lazyqsar/base/linear/README.md) |
| `lazyqsar.base.randomforest` | Random Forest classifier wrapper | — |

## How It Works

LazyQSAR builds an ensemble for each descriptor set through four steps:

1. **Portfolio selection** — the dataset is profiled (sample count, dimensionality, sparsity, class imbalance) and a rule-based selector decides which heads to train. The default portfolio is XGBoost + Random Forest; Logistic Regression is added automatically for small, high-dimensional, or low-prevalence datasets.

2. **Preprocessing** — a `BasePreprocessor` selects a scaler (`StandardScaler`, `RobustScaler`, `MaxAbsScaler`, or `PowerTransformer`) and optionally applies a correlation-based feature reducer, both chosen automatically from data statistics.

3. **Heads** — each selected head (Logistic Regression, XGBoost, Random Forest) is fitted on the preprocessed features. For severely imbalanced datasets the training data is split into balanced batches and one model ensemble is built per batch.

4. **Pooling and export** — head predictions are combined via a learned gating network (`InnerClassifierPooler`). The full pipeline (preprocessor + heads + pooler) is exported to ONNX, enabling dependency-free inference with only `numpy` and `onnxruntime`.

## Use in an Ersilia Model Hub template

LazyQSAR models can be used inside an [Ersilia Model Hub template](https://github.com/ersilia-os/eos-template) structure. See [eos1lb5](https://github.com/ersilia-os/eos1lb5) for an example.

`lazyqsar fit` produces a `checkpoints` folder with the following structure (one sub-directory per task, one per descriptor type):

```text
checkpoints/
├── task1/
│   ├── cddd/
│   │   ├── featurizer.json
│   │   ├── metadata.json
│   │   └── batch_0/
│   │       ├── preprocessor.onnx
│   │       ├── xgboost.onnx
│   │       └── pooler.json
│   ├── chemeleon/
│   │   └── (same structure)
│   └── rdkit/
│       └── (same structure)
└── task2/
    └── (same structure)
```

The `code/main.py` script should look like this:

```python
import os
import sys

from lazyqsar.api.classifier_predict import predict

root = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.abspath(os.path.join(root, "..", "checkpoints"))

input_file = sys.argv[1]
output_file = sys.argv[2]

predict(model_dir=checkpoints_dir, input_csv=input_file, output_csv=output_file)
```

Note that output columns are ordered alphabetically by task name. For a more controlled approach, see [eos1lb5](https://github.com/ersilia-os/eos1lb5).

## Disclaimer

This library is intended for quick QSAR modeling. For a more complete automated QSAR pipeline, refer to [Zaira Chem](https://github.com/ersilia-os/zaira-chem).

## About Us

Learn about the [Ersilia Open Source Initiative](https://ersilia.io)!

# Ersilia's LazyQSAR

A Python library for building supervised binary QSAR (Quantitative Structure-Activity Relationship) models quickly, with minimal configuration. LazyQSAR automates descriptor computation, feature selection, and hyperparameter tuning to produce robust ensemble models from chemical structures.

**Two usage modes:**
- **SMILES-based:** pass molecule SMILES strings directly; descriptors are computed automatically
- **Descriptor-agnostic:** bring your own pre-computed descriptor arrays or HDF5 files

## Table of Contents

- [Installation](#installation)
- [Python API](#use-as-a-python-api)
  - [Binary Classification (SMILES)](#binary-classification)
  - [Binary Classification (Custom Descriptors)](#custom-descriptors)
  - [Saving and Loading Models](#saving-and-loading-models)
  - [Tests and Benchmarks](#tests-and-benchmarks)
- [CLI](#use-as-a-cli)
- [How It Works](#how-it-works)
- [Disclaimer](#disclaimer)
- [About Us](#about-us)

## Installation

Install LazyQSAR from source:

```bash
git clone https://github.com/ersilia-os/lazy-qsar.git
cd lazy-qsar
python -m pip install -e .
```

To use the built-in LazyQSAR descriptors, install the optional dependencies:

```bash
python -m pip install -e .[descriptors]
```

This enables descriptor (featurizer) calculation. The first time you run LazyQSAR with deep-learning descriptors, it will download the Chemeleon and CDDD model checkpoints. To complete this setup in advance, run:

```bash
lazyqsar-setup
```

## Use as a Python API

### Binary Classification

LazyQSAR's binary classifier can run either with built-in descriptors (takes SMILES as input) or with custom pre-computed descriptors.

#### Built-in descriptors

Instantiate `LazyBinaryQSAR` with a mode of choice:

| Mode | Descriptors used | Optuna trials | Speed |
|------|-----------------|--------------|-------|
| `fast` | RDKit, Morgan fingerprints | 1 | Fastest, no deep-learning descriptors |
| `default` | Chemeleon, RDKit, CDDD | 10 | Balanced |
| `slow` | Chemeleon, Morgan, RDKit, CDDD | 30 | Most thorough |

```python
from lazyqsar.qsar import LazyBinaryQSAR

model = LazyBinaryQSAR(mode="default")
model.fit(smiles_list=smiles_train, y=y_train)
y_hat = model.predict_proba(smiles_list=smiles_test)[:, 1]
```

#### Custom descriptors

Pre-calculate your own descriptors and pass them directly. We recommend the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia) for this — its `.h5` output format is supported natively. Alternatively, pass descriptors as a NumPy array.

```python
from lazyqsar.agnostic import LazyBinaryClassifier

# From a NumPy array
model = LazyBinaryClassifier(mode="default")
model.fit(X=X_train, y=y_train)
y_hat = model.predict_proba(X=X_test)[:, 1]

# From an Ersilia .h5 file
model.fit(h5_file="descriptors.h5", y=y_train)
y_hat = model.predict_proba(h5_file="descriptors.h5")[:, 1]
```

### Saving and loading models

Models are saved as ONNX files by default, so inference only requires the ONNX runtime (no scikit-learn dependency at prediction time).

```python
# Save after training
model.save(model_dir)

# Load for inference (auto-detects ONNX or raw format)
from lazyqsar.agnostic import LazyBinaryClassifier

model = LazyBinaryClassifier.load(model_dir)
y_hat = model.predict_proba(X=X)[:, 1]
```

You can also save and load as a `.zip` archive:

```python
model.save("my_model.zip")
model = LazyBinaryClassifier.load("my_model.zip")
```

The same save/load interface applies to `LazyBinaryQSAR`:

```python
from lazyqsar.qsar import LazyBinaryQSAR

model = LazyBinaryQSAR(mode="default")
model.fit(smiles_list=smiles_train, y=y_train)
model.save(model_dir)

model = LazyBinaryQSAR.load(model_dir)
y_hat = model.predict_proba(smiles_list=smiles_test)[:, 1]
```

### Tests and benchmarks

#### Quick testing

The `tests/` folder contains scripts for quickly verifying that the code works. The Bioavailability dataset is used as an example.

```bash
python tests/test_binary_classification.py
python tests/test_binary_classification.py --agnostic
```

Additional flags:

| Flag | Description |
|------|-------------|
| `--mode fast\|default\|slow` | Select descriptor mode |
| `--agnostic` | Use descriptor-agnostic `LazyBinaryClassifier` |
| `--no-onnx` | Skip ONNX conversion |
| `--no-zip` | Skip ZIP archive save/load |
| `--clean` | Remove temporary files after the run |

#### Benchmarking

The [benchmark repository](https://github.com/ersilia-os/zaira-chem-tdc-benchmark) contains performance results for the default estimators and descriptors on the TDCommons ADMET dataset.

## Use as a CLI

The CLI expects a `data_dir` containing one CSV file per task. Each CSV must have SMILES in the first column and binary labels (0/1) in the second column, with a header row.

**Fit:**

```bash
lazyqsar-binary-fit --data_dir $DATA_DIR --model_dir $MODEL_DIR --mode default
```

Optionally, pass a `--models_txt` file listing which tasks (CSV filenames without extension) to train, one per line. Without it, all CSVs in the directory are used.

```bash
lazyqsar-binary-fit --data_dir $DATA_DIR --model_dir $MODEL_DIR --models_txt models.txt
```

**Predict:**

```bash
lazyqsar-binary-predict --input_csv $INPUT_CSV --model_dir $MODEL_DIR --output_csv $OUTPUT_CSV
```

The output CSV contains the input SMILES and one predicted probability column per task. Optionally use `--models_txt` to run predictions only for a subset of tasks.

## How It Works

LazyQSAR builds a weighted ensemble of up to 10 model variants per descriptor set:

1. **Preprocessing** — missing value imputation, variance filtering, and scaling (StandardScaler for dense data, TF-IDF for sparse fingerprints)
2. **Feature selection** — univariate (F-test) and model-based (permutation importance) selection pipelines
3. **Latent variables** — optional PCA-based dimensionality reduction
4. **Classifiers** — Logistic Regression, Linear SVM, Extra Trees, and MLP (PyTorch), each tuned via [Optuna](https://optuna.org)
5. **Ensemble** — predictions are weighted by individual model ROC-AUC scores on out-of-fold validation

All scikit-learn components are exported to ONNX for lightweight, dependency-free inference.

## Use in an Ersilia Model Hub template

LazyQSAR models can be used inside an [Ersilia Model Hub template](https://github.com/ersilia-os/eos-template) structure. See [eos1lb5](https://github.com/ersilia-os/eos1lb5) for an example.

Given a `checkpoints` folder with the following structure:

```text
checkpoints/
├── task1/
│   ├── cddd/
│   │   ├── featurizer.json
│   │   └── model.onnx
│   ├── chemeleon/
│   │   ├── featurizer.json
│   │   └── model.onnx
│   └── rdkit/
│       ├── featurizer.json
│       └── model.onnx
└── task2/
    ├── cddd/
    ├── chemeleon/
    └── rdkit/
```

The `code/main.py` script should look like this:

```python
import os
import sys
import csv

from lazyqsar.api.binary_qsar_predict import predict

root = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.abspath(os.path.join(root, "..", "checkpoints"))

input_file = sys.argv[1]
output_file = sys.argv[2]

predict(model_dir=checkpoints_dir, input_csv=input_file, output_csv=output_file)
```

Note that the order of the columns is alphabetical in the case presented. For a more controlled approach, look into the [eos1lb5](https://github.com/ersilia-os/eos1lb5) repository for an example.

## Disclaimer

This library is intended for quick QSAR modeling. For a more complete automated QSAR pipeline, refer to [Zaira Chem](https://github.com/ersilia-os/zaira-chem).

## About Us

Learn about the [Ersilia Open Source Initiative](https://ersilia.io)!

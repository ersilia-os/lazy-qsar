This tool has been financed by Project PID2023-148309OA-I00 funded by MICIU/AEI/10.13039/501100011033 and by ERDF, EU.

<img src="https://raw.githubusercontent.com/ersilia-os/ersilia/master/assets/miciu_cofinanciado.jpg" width="300">

# Ersilia's LazyQSAR

A Python library for building supervised QSAR models quickly, with minimal configuration. LazyQSAR automates chemical descriptor computation, and model selection to produce robust models for property and activity prediction.

**Two entry points:**
- **`LazyClassifierQSAR`**: pass SMILES strings directly; built-in descriptors are computed automatically
- **`LazyClassifier`**: bring your own pre-computed descriptor arrays

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

We recommend installation from source:

```bash
git clone https://github.com/ersilia-os/lazy-qsar.git
cd lazy-qsar
pip install -e .
```

The base install includes only lightweight runtime dependencies (`numpy`, `onnxruntime`, etc.), sufficient for loading and running pre-trained ONNX models without any ML and chemistry-related packages (RDKit). Therefore, the base install assumes descriptors are provided by the user.

You can install optional extras depending on your use case:

| Extra | Command | Adds |
|-------|---------|------|
| `fit` | `pip install -e .[fit]` | Training dependencies (scikit-learn, XGBoost, scipy, ONNX conversion tools) |
| `descriptors` | `pip install -e .[descriptors]` | Built-in molecular descriptors (RDKit, FPSim2, deep-learning models) |
| `all` | `pip install -e .[all]` | Everything above |

> CPU-only deployments where pip would otherwise pull the CUDA torch wheel (~3 GB) can pass `--cpu-torch` to force-reinstall torch from PyTorch's CPU index: `lazyqsar setup --descriptors --cpu-torch`.

The first time you use deep-learning descriptors (Chemeleon, CLAMP, CDDD), their checkpoints are downloaded automatically. To do this in advance:

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

model = LazyClassifierQSAR(mode="slow") # default is "slow"
model.fit(smiles_list=smiles_train, y=y_train)

ranks = model.predict_rank(smiles_list=smiles_test)[:, 1]  # percentile rank vs training set
```

Other prediction methods include `predict_proba`, `predict`, `predict_lift`, and more — see [docs/internals.md](docs/internals.md#part-8-prediction-methods) for the full list.

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

Models are saved as ONNX files, so inference only requires `numpy` and `onnxruntime`, i.e. no scikit-learn or XGBoost at prediction time. Metadata is stored in JSON format.

To save models:

```python
model.save(model_dir)          # directory
model.save("my_model.zip")     # or zip archive
```

And to load them:

```python
model = LazyClassifierQSAR.load(model_dir)
y_hat = model.predict_proba(smiles_list=smiles_test)[:, 1]

model = LazyClassifier.load(model_dir)
y_hat = model.predict_proba(X=X_test)[:, 1]
```

For multi-endpoint prediction across multiple model directories, see [Ersilia Model Hub integration](#ersilia-model-hub-integration).

## CLI

All commands are available through the `lazyqsar` entry point.

**Fit:**

The `--input` directory must contain one CSV per task, with SMILES in the first column and binary labels (0/1) in the second column, with a header row.

```bash
lazyqsar fit --task classification --input $DATA_DIR --output $MODEL_DIR --mode slow
```

Pass `--models_txt` to train a subset of tasks (one CSV stem per line); without it, all CSVs in the directory are used.

**Predict:**

```bash
lazyqsar predict --input $INPUT_CSV --model $MODEL_DIR --output $OUTPUT_CSV [--models_txt FILE] [--predict_type TYPE]
```

The output CSV contains one column per task, ordered alphabetically by task name, or filtered and ordered by `--models_txt` at predict time. `--predict_type` controls the output format: `proba` (default), `rank`, `logit`, `lift`, `score`, or `binary`.

## How it works

LazyQSAR builds an ensemble for each descriptor set through four steps:

1. **Portfolio selection**: the dataset is profiled (sample count, dimensionality, sparsity, class imbalance) and a rule-based selector decides which heads to train. The default portfolio is XGBoost + Random Forest; Linear Models and Support Vector Machines are added automatically for small, high-dimensional, or low-prevalence datasets.

2. **Preprocessing**: a scaler (`StandardScaler`, `RobustScaler`, `MaxAbsScaler`, or `PowerTransformer`) and an optional correlation-based feature reducer are selected automatically from dataset statistics.

3. **Heads**: each selected head is fitted on preprocessed features. For severely imbalanced datasets, balanced sub-batches are used and the batch predictions are averaged.

4. **Pooling**: head predictions are combined via a learned gating network (`InnerClassifierPooler`). When using `LazyClassifierQSAR`, a separate ensemble is trained per descriptor type and their predictions are combined via an AUC-weighted ensemble that accounts for per-sample prediction confidence.

The full pipeline is exported to ONNX, so inference requires only `numpy` and `onnxruntime`.

## Base Models

The components under `lazyqsar/base/` can be used independently of the full pipeline:

| Module | Description |
|--------|-------------|
| [`lazyqsar.base.preprocessing`](lazyqsar/base/preprocessing/) | Automatic scaler and feature reducer selection |
| [`lazyqsar.base.xgboost`](lazyqsar/base/xgboost/README.md) | Automatic XGBoost hyperparameter selection with portfolio comparison |
| [`lazyqsar.base.linear`](lazyqsar/base/linear/README.md) | Automatic linear model selection (logistic/ridge/SGD) |
| [`lazyqsar.base.randomforest`](lazyqsar/base/randomforest/README.md) | Random Forest classifier with zero-shot hyperparameter selection |
| [`lazyqsar.base.svc`](lazyqsar/base/svc/) | Support Vector Classifier with automatic kernel and C selection |

## Ersilia Model Hub integration

LazyQSAR models can be used inside an [Ersilia Model Hub template](https://github.com/ersilia-os/eos-template). See [eos3dys](https://github.com/ersilia-os/eos3dys) for an example.

Basically, `lazyqsar fit` can be used to produce a `checkpoints` folder with one sub-directory per task and per descriptor type:

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

The `code/main.py` inference script:

```python
import os, sys
from lazyqsar.api.classifier_predict import predict
from ersilia_pack_utils.core import read_smiles, write_out

root = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.abspath(os.path.join(root, "..", "checkpoints"))
_, smiles_list = read_smiles(input_file)
outputs, header = predict(model_dir=checkpoints_dir, smiles=smiles_list, predict_type="rank")
write_out(outputs, header, output_file, np.float32)
```

This function computes descriptors once per descriptor type and reuses them across all tasks, making it suitable for scoring large compound libraries. `predict_type` controls the output format and is available in both the Python API and the CLI (`--predict_type`).

`model_dir` also accepts a `dict[str, str]` mapping **column names to model directories**, for scoring multiple targets stored under separate paths — see [docs/internals.md](docs/internals.md) for details.

## Disclaimer

This library is intended for quick QSAR modeling. For a more complete automated QSAR pipeline, refer to [ZairaChem](https://github.com/ersilia-os/zaira-chem).

ZairaChem's version, with an earlier version of LazyQSAR, was presented in this article:

```
@article{Turon2023,
  author = {Turon, G. and Hlozek, J. and Woodland, J.G. and et al.},
  title = {First fully-automated AI/ML virtual screening cascade implemented at a drug discovery centre in Africa},
  journal = {Nat Commun},
  volume = {14},
  pages = {5736},
  year = {2023},
  doi = {10.1038/s41467-023-41512-2},
  url = {https://doi.org/10.1038/s41467-023-41512-2}
}
```


## About the Ersilia Open Source Initiative

The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization with the mission to equip laboratories, universities, and clinics in the Global South with AI/ML tools for infectious disease research. We work on the principles of open science, decolonized research, and egalitarian access to knowledge and research outputs. You can support Ersilia by clicking here.

# Ersilia's LazyQSAR

A library to build supervised QSAR models for chemistry quickly.

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

This will enable descriptor (featurizer) calculation. The first time you run LazyQSAR, it will download the Chemeleon and CDDD model checkpoints. To complete this setup in advance, run:

```bash
lazyqsar-setup
```

## Use as a Python API

### Binary Classification

LazyQSAR's binary classifier can run either with built-in descriptors (takes SMILES as input) or with custom pre-computed descriptors.

#### Built-in descriptors

Instantiate `LazyBinaryQSAR` with a mode of choice:

| Mode | Descriptors used | Speed |
|------|-----------------|-------|
| `fast` | RDKit, Morgan fingerprints | Fastest, no deep-learning descriptors |
| `default` | Chemeleon, RDKit, CDDD | Balanced |
| `slow` | Chemeleon, Morgan, RDKit, CDDD | Most thorough |

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

Models are saved as ONNX files by default, so inference only requires the ONNX runtime.

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

## Disclaimer

This library is intended for quick QSAR modeling. For a more complete automated QSAR pipeline, refer to [Zaira Chem](https://github.com/ersilia-os/zaira-chem).

## About us

Learn about the [Ersilia Open Source Initiative](https://ersilia.io)!

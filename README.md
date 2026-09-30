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
- [Running the tests](#running-the-tests)
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
| `fit` | `pip install -e ".[fit]"` | Training dependencies (scikit-learn, XGBoost, scipy, ONNX conversion tools) and `eosvc`, which is what fetches the reference library — so `rank` needs this extra even if you bring your own descriptors |
| `descriptors` | `pip install -e ".[descriptors]"` | Built-in molecular descriptors (RDKit, FPSim2, deep-learning models) |
| `all` | `pip install -e ".[all]"` | Everything above |

> CPU-only deployments where pip would otherwise pull the CUDA torch wheel (~3 GB) can pass `--cpu-torch` to force-reinstall torch from PyTorch's CPU index: `lazyqsar setup --descriptors --cpu-torch`.

The first time you use deep-learning descriptors (Chemeleon, CLAMP, CDDD), their checkpoints are downloaded automatically. To do this in advance:

```bash
lazyqsar setup --descriptors
```

Fitting also needs the reference library that `rank` is reported against. It is fetched on
first use, or in advance:

```bash
lazyqsar setup --reference                    # all five descriptors, 267 MB
lazyqsar setup --reference --only morgan      # fast mode needs only this, 6.5 MB
```

It is deliberately **not** included in `--descriptors`. Inspect or manage it with:

| command | |
|---|---|
| `lazyqsar reference status` | what is cached, and how big |
| `lazyqsar reference fetch [--only LIST] [--force]` | download it |
| `lazyqsar reference verify` | check what is cached is well formed |
| `lazyqsar reference smiles --output ref.csv` | the molecule list, which is all a bring-your-own-descriptor caller needs |

`LAZYQSAR_HOME` moves the cache (checkpoints and reference together);
`LAZYQSAR_REFERENCE_DIR` points at a prepared copy; `LAZYQSAR_REFERENCE_OFFLINE=1` refuses
to fetch rather than reaching for the network.

`LAZYQSAR_REFERENCE_N` selects the reference *tier* — how many molecules the library holds.
It changes what `rank` means, and the value is stamped into every checkpoint as
`pooled_ranker.library.n`, so two models fitted under different tiers are not comparable
on `rank` even though both report a number in [0, 1]. Leave it alone unless you know why
you are changing it.

`LAZYQSAR_FIT_SCRATCH` redirects where a fit stages its descriptor matrices. Worth setting
on a node with a small `/tmp`: a slow-mode fit over many tasks stages one matrix per
descriptor over the union of every task, which for 175,000 compounds across five
descriptors is several gigabytes.

## When a fit or a prediction refuses

The library fails loudly in a few places rather than returning a number it cannot stand
behind. Each of these is a deliberate refusal, not a bug:

| What you see | What it means | What to do |
|---|---|---|
| `predict_rank needs a reference library` | The checkpoint carries no reference — either it predates 3.6, or it was fitted through `LazyClassifier` without `reference_X=`. | Refit with 3.6+, or pass `reference_X=`. `proba`, `logit`, `lift`, `score` and `binary` still work. |
| `No reference matrix for ... and fetching is disabled` / `eosvc ... is not installed` | Checked *before* training so a long fit is not lost to it. | `lazyqsar setup --reference`, or point `LAZYQSAR_REFERENCE_DIR` at a copy, or `pip install "lazyqsar[fit]"` for `eosvc`. |
| `class N has a single member` / `every label is N` | Stratified splitting cannot work, so no fold count exists. Raised from the labels before any descriptor is computed. | Add examples of the minority class, or stop treating the task as classification. |
| `Invalid SMILES at position(s): ...` at fit | Training on a molecule that cannot be featurized is meaningless, so fit raises rather than guessing. Note an empty cell counts as invalid. | Clean the input. At *predict* time the same rows are NaN'd instead, and the rest of the library still scores. |
| `This checkpoint's reference rank was built from [...] but [...] missing` | Descriptor directories have been removed from the checkpoint, so it would be scored with fewer descriptors than its reference describes. | Restore the directories or refit. |
| A warning that the cutoff `admits X% of the reference library, not 1%` | The reference's probabilities are too concentrated for the anchors to separate, so this model's `rank` is not comparable with others. | Check `LAZYQSAR_REFERENCE_N` and whether the model saturates. |
| A warning that the actives anchor was dropped | The model's known actives do not reach the top 0.1% of drug-like space. Informative, not an error — see the changelog's Known limitations. | Nothing; the scale runs straight to certainty above the last reference anchor. |

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

ranks = model.predict_rank(smiles_list=smiles_test)[:, 1]  # position against 50,000 drug-like reference molecules; 0.65 = beats 99%
```

Other prediction methods are `predict_proba`, `predict_logit`, `predict_score`, `predict_lift` and `predict` (binary labels). All six share one implementation with the CLI, so a checkpoint gives the same answer through either entry point.

> `predict_rank` positions a molecule against a **fixed reference library** of 50,000
> drug-like molecules, anchored on that library's **upper tail**. Each step of rank is a 10x
> shrink of the tail:
>
> | rank | means |
> |---|---|
> | 0.25 | beats half of drug-like chemical space |
> | 0.50 | beats 90% -- the top tenth |
> | **0.65** | beats 99% -- **the decision cutoff** |
> | 0.75 | beats 99.9% |
> | 0.95 | at the top of what this model's known actives reach |
>
> Within one model it is a monotone view of `proba` -- they order molecules identically, so
> any ordering-only metric (AUROC, AUPRC, BEDROC) gives the same answer from either.
>
> **The axis is spent on the top of the list**, because that is a bioactivity model's
> product. Measured across six antimicrobial models, the top 1% of a screened library sits
> between the reference's p98.9 and p100; anchoring on quartiles instead gave that top 1% a
> mean span of 0.069 of the axis against 0.215 here, and on one model the 113 best-scoring
> compounds shared a span of 0.003 -- effectively one value.
>
> **The cost is the bottom.** Roughly a third of a generic library can land below 0.25, so
> `rank` says little about *how* inactive something is. Do not read low ranks
> quantitatively.
>
> Above the reference's p99.9 the library is too sparse to resolve anything, so the last
> stretch is pinned on the model's own out-of-fold actives -- `0.95` is their 95th
> percentile. When a model's actives do not even reach p99.9 that anchor is dropped and
> reported, which is a statement about the model: its actives look like generic chemistry.
>
> The top of the scale therefore does not distinguish a strong model from a weak one. That
> signal lives in `oof_diagnostics.screening_auc` and `sensitivity_at_cutoff`, both reported
> in every checkpoint.
>
> And a rank says nothing on its own about model skill -- a random model still puts 1% of the
> reference above the cutoff, because the cutoff is *defined* as 1%. Every output is a
> monotone transform of one probability, so none of them can separate a false positive from a
> true positive that scores the same. Report `proba`, `lift` and the out-of-fold AUC
> alongside it.

### LazyClassifier (custom descriptors)

Pass your own descriptor arrays or HDF5 files. We recommend the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia) for descriptor computation — its `.h5` output format is supported natively.

```python
from lazyqsar.agnostic import LazyClassifier

# From a NumPy array
model = LazyClassifier()
model.fit(X=X_train, y=y_train)
y_hat = model.predict_proba(X=X_test)[:, 1]

# `rank` needs a reference library. This entry point never sees the molecules, so it
# cannot featurize one -- pass the descriptors of the reference set yourself:
from lazyqsar.reference import reference_smiles

X_ref = my_featurizer(reference_smiles())        # same featurizer, same order
model.fit(X=X_train, y=y_train, reference_X=X_ref)   # or reference_h5_file="ref.h5"
ranks = model.predict_rank(X=X_test)[:, 1]

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

The output CSV contains one column per task, ordered alphabetically by task name, or filtered and ordered by `--models_txt` at predict time. `--predict_type` controls the output format:

| type | meaning |
|------|---------|
| `proba` (default) | calibrated probability of the positive class |
| `rank` | position against the 50,000-molecule reference library, on its upper tail: 0.50 is the top 10%, 0.65 the top 1%, 0.75 the top 0.1% |
| `logit` | log-odds of the calibrated probability |
| `lift` | probability divided by the training-set positive rate |
| `score` | the pre-calibration scale, read off the calibrated probability |
| `binary` | 0/1 label, thresholded at the decision cutoff -- `rank >= 0.65`, i.e. beats 99% of drug-like space. Checkpoints without a reference-rank cutoff keep `proba >= 0.5` |

All six rank molecules identically — they are different scales on one quantity, so sorting
by any of them gives the same order. `score` reports what the model looked like before
calibration; it reaches that scale through a monotone map stored in the checkpoint rather
than by pooling the raw per-descriptor scores, which would not agree with `proba` about the
order. Checkpoints fitted before v3.5.0 carry no map and keep the older behaviour, where
`score` could disagree.

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
| [`lazyqsar.base.preprocessing`](lazyqsar/base/preprocessing/README.md) | Automatic scaler and feature reducer selection |
| [`lazyqsar.base.xgboost`](lazyqsar/base/xgboost/README.md) | Automatic XGBoost hyperparameter selection with portfolio comparison |
| [`lazyqsar.base.linear`](lazyqsar/base/linear/README.md) | Automatic linear model selection (logistic/ridge/SGD) |
| [`lazyqsar.base.randomforest`](lazyqsar/base/randomforest/README.md) | Random Forest classifier with zero-shot hyperparameter selection |
| [`lazyqsar.base.svc`](lazyqsar/base/svc/README.md) | Support Vector Classifier with automatic kernel and C selection |

## Running the tests

```bash
pip install -e ".[fit,test]"
pytest
```

The suite is organised by what a test needs installed, so it runs — and reports honestly —
in whichever environment you have. A test whose tier is missing is skipped with a message
naming the absent module, never an error.

| Selection | Needs | Covers |
|---|---|---|
| `pytest -m "not fit and not chem"` | the base install | the inference path, the ensemble arithmetic, the registry and the CLI surface |
| `pytest` | `.[fit]` plus `rdkit` | the above, plus fitting, ONNX export, the pipeline, and Morgan and RDKit descriptors |

Those are the only two selections, and CI runs both as parallel jobs on Python 3.12, so a
pull request finishes in a few minutes. There is no third, larger one: `pip install
".[all]"` pulls in torch, chemprop and chemeleon but adds no tests, because nothing in the
suite executes them.

The base selection is the contract Ersilia Model Hub templates depend on: it must pass on
the core dependencies alone — `numpy`, `onnxruntime`, `pandas`, `h5py`, `psutil`, `rich`
and `loguru` — with no `scikit-learn`, `xgboost` or RDKit anywhere on the path. Add
`-n auto --dist loadfile` for the heavier tier, and `--durations=15` to see where the time
goes.

### What is not covered

Worth stating plainly, because a green suite does not mean these are exercised:

- **Three of the five descriptors have no execution coverage.** Only `morgan` and `rdkit`
  are ever actually run. `cddd`, `clamp` and `chemeleon` appear in tests only as names —
  in stubbed registries, CLI argument validation and import-purity checks — so a change to
  any of them can break `slow` mode without reddening anything. Test them by hand against
  real molecules before changing them.
- **The SVC ONNX export.** The score-column sign and the per-head export-versus-fit
  comparison are not checked. A converter regression there is silent.
- **Python 3.11 and 3.13.** Supported per `requires-python`, exercised by nothing.

Install the `fit` extra from the pins rather than whatever is already in your environment:
`scikit-learn` in particular is pinned to `1.9.1`, and descriptor selection differs enough
between minor versions to change which descriptors a portfolio keeps.

Test data lives in `tests/data/` and is documented in `tests/data/README.md`.

## Ersilia Model Hub integration

LazyQSAR models can be used inside an [Ersilia Model Hub template](https://github.com/ersilia-os/eos-template). See [eos3dys](https://github.com/ersilia-os/eos3dys) for an example.

Basically, `lazyqsar fit` can be used to produce a `checkpoints` folder with one sub-directory per task and per descriptor type:

```text
checkpoints/
└── task1/
    ├── metadata.json          task-level: active mask, AUCs, cutoff, reference knots
    ├── chemeleon/
    │   ├── featurizer.json
    │   ├── metadata.json
    │   ├── applicability_domain/
    │   │   └── applicability_domain.onnx
    │   └── batch_0/
    │       ├── preprocessor.onnx
    │       ├── xgboost.onnx
    │       └── pooler.json
    ├── clamp/     (same structure)
    └── morgan/    (same structure)
```

Slow mode screens five descriptors and typically keeps **two or three** — the portfolio
prunes the rest, and a pruned descriptor's directory is not written at all, so a real
checkpoint is smaller than the descriptor list suggests. Which survived is recorded in the
task-level `metadata.json` under `active_descriptors`.

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

Descriptors are computed once per descriptor type and shared across every task, and molecules are scored in blocks whose working set is released before the next block starts, so peak memory does not grow with the size of the input. The block size is derived from a fixed working-set budget of roughly 1 GB and the number of endpoints being scored, and is always a whole number of chunks — which is what keeps the output identical however the input is divided — so there is no knob for it. `LAZYQSAR_PREDICT_CHUNK` (default 1000) sets the featurization and inference batch size within a block.

`model_dir` also accepts a `dict[str, str]` mapping **column names to model directories**, for scoring multiple targets stored under separate paths. Column names and their order are preserved exactly as given.

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

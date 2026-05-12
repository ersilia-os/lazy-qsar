# dev/

This directory contains development-time artefacts for the `lazyqsar` package. It is **not** part of the published package and is not installed by `pip install lazyqsar`.

```
dev/
├── benchmarks/   scripts and notebooks to evaluate model performance
├── data/         raw datasets used by smoke tests and benchmarks
├── smoke/        end-to-end smoke tests for quick sanity checks
└── tests/        unit and integration tests (run with pytest)
```

> **Reproducibility notice:** Scripts and notebooks in `dev/` are not guaranteed to be reproducible. They may depend on external datasets, specific hardware, optional dependencies, or produce stochastic outputs. See each subdirectory README for guidance on obtaining required data and expected environment.

## Running tests

```bash
# Install development dependencies
pip install -e .[fit]

# Run the full test suite from the repo root
pytest

# Run a single test file
pytest dev/tests/test_classifier_unit.py -v
```

## Running smoke tests

```bash
python dev/smoke/smoke_classifier.py           # 500 samples, 100 features
python dev/smoke/smoke_classifier.py 1000 200  # custom size
```

Smoke tests for QSAR workflows additionally require `dev/data/` to be populated — see [dev/data/README.md](data/README.md).

# Lazy QSAR

A library to build fast QSAR models

## Installation

```bash
git clone https:/github.com/ersilia-os/lazy-qsar.git
cd lazy-qsar
python -m pip install -e .
```

## Usage

```python
import lazyqsar as lq

# get an example
smiles, y = lq.examples.binary_classification()

# train model
model = lq.MorganBinaryClassifier()
model.fit(smiles, y)

# make predictions
y_hat = model.predict(smiles)
```

## Disclaimer

This library is only intended for quick-and-dirty QSAR modeling.
For a more complete automated QSAR modeling, please refer to [Zaira Chem](https://github.com/ersilia-os/zaira-chem)

## About us

Learn about the [Ersilia Open Source Initiative](https://ersilia.io)!

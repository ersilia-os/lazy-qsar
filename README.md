# Lazy QSAR

A library to build QSAR models fastly

## Installation

```bash
git clone https://github.com/ersilia-os/lazy-qsar.git
cd lazy-qsar
python -m pip install -e .
```

## Usage

### Get an example

You can find example data in the fantastic [Therapeutic Data Commons](https://tdcommons.ai) portal.

```python
# fetch an example from TDC
from tdc.single_pred import Tox
data = Tox(name = 'hERG')
split = data.get_split()
```
Here we are selecting the hERG blockade toxicity dataset. Let's refactor data for convenience.

```python
# refactor fetched data in a convenient format
smiles_train = list(split["train"]["Drug"])
y_train = list(split["train"]["Y"])
smiles_valid = list(split["valid"]["Drug"])
y_valid = list(split["valid"]["Y"])
```

### Build a model

Now we can train (and validate) a model based on Morgan fingerprints.

```python
import lazyqsar as lq

# train
model = lq.MorganBinaryClassifier()
model.fit(smiles_train, y_train)

# validate
from sklearn.metrics import roc_curve, auc
y_hat = model.predict_proba(smiles_valid)[:,1]
fpr, tpr, _ = roc_curve(y_valid, y_hat)
print("AUROC", auc(fpr, tpr))
```

## Disclaimer

This library is only intended for quick-and-dirty QSAR modeling.
For a more complete automated QSAR modeling, please refer to [Zaira Chem](https://github.com/ersilia-os/zaira-chem)

## About us

Learn about the [Ersilia Open Source Initiative](https://ersilia.io)!

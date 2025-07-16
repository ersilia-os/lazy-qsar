# Ersilia's LazyQSAR

A library to build QSAR models fastly.

## Installation

Install LazyQSAR from source:

```bash
git clone https://github.com/ersilia-os/lazy-qsar.git
cd lazy-qsar
python -m pip install -e .
```

## Usage

1. Choose one of the available descriptors of small molecules.
2. Fit a model using AutoML. LazyQSAR will search several hyperparametrs.
3. Get the validation of the model on the test set.

### Example for binary classifications

#### Get the data

You can find example data in the fantastic [Therapeutic Data Commons](https://tdcommons.ai) portal.

```python
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

#### Build a model

Now we can train a model based on Morgan fingerprints.

```python
import lazyqsar as lq

model = lq.LazyBinaryQSAR(descriptor_type="morgan", model_type="random_forest") 
model.fit(smiles_list=smiles_train, y=y_train)
model.save_model(model_dir="my_model")
```

#### Validate its performance

```python
from sklearn.metrics import roc_curve, auc
y_hat = model.predict_proba(smiles_valid)[:,1]
fpr, tpr, _ = roc_curve(y_valid, y_hat)
print("AUROC", auc(fpr, tpr))
```

### Example for Regressions
_In the current version of LazyQSAR regression is not yet implemented..._

#### Get the data
You can find example data in the fantastic [Therapeutic Data Commons](https://tdcommons.ai) portal.

```python
from tdc.single_pred import Tox
data = Tox(name = 'LD50_Zhu')
split = data.get_split()
```
Here we are selecting the Acute Toxicity dataset. Let's refactor data for convenience.

```python
# refactor fetched data in a convenient format
smiles_train = list(split["train"]["Drug"])
y_train = list(split["train"]["Y"])
smiles_valid = list(split["valid"]["Drug"])
y_valid = list(split["valid"]["Y"])
```

## Benchmark
The pipeline has been validated using the Therapeutic Data Commons ADMET datasets. More information about its results can be found in the /benchmark folder.

## Disclaimer

This library is only intended for quick-and-dirty QSAR modeling. For a more complete automated QSAR modeling, please refer to [Zaira Chem](https://github.com/ersilia-os/zaira-chem)

## About us

Learn about the [Ersilia Open Source Initiative](https://ersilia.io)!

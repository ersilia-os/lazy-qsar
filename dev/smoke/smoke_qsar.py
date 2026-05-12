#!/usr/bin/env python
"""
Smoke test: LazyClassifierQSAR fast mode on the bioavailability dataset.
"""

import csv
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from lazyqsar.utils.logging import logger
from lazyqsar.qsar import LazyClassifierQSAR

logger.set_verbosity(True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data")


def _load(split):
    path = os.path.join(DATA_DIR, f"bioavailability_ma_{split}.csv")
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        smiles, labels = [], []
        for row in reader:
            smiles.append(row[0])
            labels.append(int(row[1]))
    return smiles, np.array(labels)


smiles_train, y_train = _load("train")
smiles_test, y_test = _load("test")

model = LazyClassifierQSAR(mode="slow")
model.fit(smiles_train, y_train)

proba = model.predict_proba(smiles_test)
auc = roc_auc_score(y_test, proba[:, 1])

print("\nSolo AUCs per descriptor:")
for i, name in enumerate(model.descriptor_types):
    X = model._transform_cached(i, smiles_test)
    solo = roc_auc_score(y_test, model.models[i].predict_proba(X=X)[:, 1])
    print(f"  {name:<12} {solo:.4f}")

print(f"\nCombined AUC = {auc:.4f}")
assert auc > 0.6, f"AUC too low: {auc:.4f}"
print("PASSED")

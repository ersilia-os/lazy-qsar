#!/usr/bin/env python
"""
Smoke test: all 4 descriptor types + applicability domain (bioavailability dataset).
"""

import csv
import os
from lazyqsar.applicability import ApplicabilityDomain
from lazyqsar.utils.logging import logger

logger.set_verbosity(True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data")


def _load_smiles(split):
    path = os.path.join(DATA_DIR, f"bioavailability_ma_{split}.csv")
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        return [row[0] for row in reader]


smiles_train = _load_smiles("train")
smiles_test = _load_smiles("test")

descriptors = {}

from lazyqsar.descriptors import MorganFingerprint, RDKitDescriptor

descriptors["morgan"] = MorganFingerprint()
descriptors["rdkit"] = RDKitDescriptor()

try:
    from lazyqsar.descriptors.chemeleon import ChemeleonDescriptor

    descriptors["chemeleon"] = ChemeleonDescriptor()
except ImportError as e:
    print(f"chemeleon skipped: {e}")

try:
    from lazyqsar.descriptors.cddd import ContinuousDataDrivenDescriptor

    descriptors["cddd"] = ContinuousDataDrivenDescriptor()
except ImportError as e:
    print(f"cddd skipped: {e}")

for name, desc in descriptors.items():
    X_train = desc.transform(smiles_train)
    X_test = desc.transform(smiles_test)
    ad = ApplicabilityDomain()
    ad.fit(X_train)
    scores = ad.score(X_test)
    print(
        f"[{name}] X={X_train.shape}  test AD mean={scores.mean():.3f}  min={scores.min():.3f}  max={scores.max():.3f}"
    )
    assert scores.shape == (len(smiles_test),)
    assert 0.0 <= scores.min() and scores.max() <= 1.0

print("PASSED")

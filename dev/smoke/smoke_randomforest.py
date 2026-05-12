#!/usr/bin/env python
"""
Smoke test for BaseRFClassifier, including ONNX roundtrip.

Usage:
    python smoke_randomforest.py
"""

import tempfile

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lazyqsar.base.randomforest import BaseRFArtifact, BaseRFClassifier
from lazyqsar.utils.logging import logger

logger.set_verbosity(True)


# -------------------------------------------------------------------
# Synthetic binary classification data
# -------------------------------------------------------------------
rng = np.random.default_rng(42)
n_samples = 2000
n_features = 50

X = rng.standard_normal((n_samples, n_features)).astype("float32")
signal = (
    1.2 * X[:, 0]
    - 0.9 * X[:, 1]
    + 0.7 * X[:, 2] * X[:, 3]
    + 0.4 * (X[:, 4] > 0).astype(float)
    + rng.normal(0, 0.6, size=n_samples)
)
y = (signal > np.median(signal)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(
    f"\nRandomForest smoke test  —  train: {len(y_train):,}  "
    f"test: {len(y_test):,}  features: {X_train.shape[1]:,}  "
    f"pos_rate: {y_train.mean():.1%}\n"
)


# -------------------------------------------------------------------
# Fit
# -------------------------------------------------------------------
clf = BaseRFClassifier(calibrated=True)
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)
score = clf.predict_score(X_test)
rank = clf.predict_rank(X_test)
logit = clf.predict_logit(X_test)
preds = clf.predict(X_test)
test_auc = roc_auc_score(y_test, proba[:, 1])

print("── RF outputs " + "─" * 52)
print(f"  predict_proba shape : {proba.shape}")
print(f"  predict_score shape : {score.shape}")
print(f"  predict_rank shape  : {rank.shape}")
print(f"  predict_logit shape : {logit.shape}")
print(f"  predict shape       : {preds.shape}")
print(f"  test AUC            : {test_auc:.4f}")
print(f"  decision_cutoff     : {clf.decision_cutoff_:.4f}")
print(f"  cutoff_source       : {clf.decision_cutoff_source_}")
print()


# -------------------------------------------------------------------
# ONNX roundtrip
# -------------------------------------------------------------------
print("── ONNX roundtrip " + "─" * 48)
with tempfile.TemporaryDirectory() as tmp:
    clf.save(tmp, onnx=True)
    artifact = BaseRFArtifact.load(tmp)
    proba_onnx = artifact.run(X_test)

p1_mem = proba[:, 1]
p1_onnx = proba_onnx[:, 1]

pearson_r, _ = pearsonr(p1_mem, p1_onnx)
spearman_r, _ = spearmanr(p1_mem, p1_onnx)
max_diff = float(np.abs(p1_mem - p1_onnx).max())
mean_diff = float(np.abs(p1_mem - p1_onnx).mean())

print(f"  Pearson  r    : {pearson_r:.6f}")
print(f"  Spearman r    : {spearman_r:.6f}")
print(f"  mean |diff|   : {mean_diff:.6f}")
print(f"  max  |diff|   : {max_diff:.6f}")

assert pearson_r > 0.999, f"ONNX roundtrip broken: Pearson r = {pearson_r:.6f}"
assert spearman_r > 0.999, f"ONNX roundtrip broken: Spearman r = {spearman_r:.6f}"

print("\nSmoke test PASSED\n")

#!/usr/bin/env python
"""
Smoke test on a large ChEMBL bioactivity dataset (Morgan fingerprints).

Dataset: chembl4649948  —  n=86,589  pos_rate~1.5%  p=2048
Data loaded directly from the zeroshot-xgboost repository.

Compares LazyClassifier vs RandomForest and LogisticRegression baselines.

Usage:
    python smoke_chembl_large.py
"""

import os
import time
import tempfile
import csv

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lazyqsar.utils.logging import logger

logger.set_verbosity(True)

from lazyqsar.agnostic import LazyClassifier

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
DATASET = "chembl4649948_smiles_activity"
DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "zeroshot-xgboost",
    "data",
    "large",
    "binary",
    DATASET,
)

X_all = np.load(os.path.join(DATA_DIR, "morgan_descriptor.npy")).astype("float32")
with open(os.path.join(DATA_DIR, "data.csv")) as f:
    rows = list(csv.reader(f))
y_all = np.array([int(r[-1]) for r in rows[1:]], dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

print(f"\n{DATASET}")
print(f"  Total: {len(y_all):,}  →  train: {len(y_train):,}  test: {len(y_test):,}")
print(
    f"  features: {X_train.shape[1]:,}  pos_rate: {y_train.mean():.2%}  "
    f"n_pos_train: {y_train.sum():,}\n"
)

# -------------------------------------------------------------------
# Baseline: vanilla Random Forest
# -------------------------------------------------------------------
print("── Random Forest (baseline) " + "─" * 40)
t0 = time.perf_counter()
rf = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_time = time.perf_counter() - t0
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print(f"  AUC = {rf_auc:.4f}   fit time = {rf_time:.1f}s\n")

# -------------------------------------------------------------------
# LazyClassifier
# -------------------------------------------------------------------
print("── LazyClassifier " + "─" * 50)
t0 = time.perf_counter()
clf = LazyClassifier()
clf.fit(X=X_train, y=y_train)
lazy_time = time.perf_counter() - t0

proba = clf.predict_proba(X=X_test)
lazy_auc = roc_auc_score(y_test, proba[:, 1])

# -------------------------------------------------------------------
# ONNX roundtrip check
# -------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    model_dir = os.path.join(tmp, "model")
    clf.save(model_dir)
    artifact = LazyClassifier.load(model_dir)
    proba_onnx = artifact.predict_proba(X=X_test[:500])

from scipy.stats import pearsonr

pearson_r, _ = pearsonr(proba[:500, 1], proba_onnx[:, 1])
max_diff = float(np.abs(proba[:500, 1] - proba_onnx[:, 1]).max())

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
print("\n" + "═" * 60)
print(f"{'Method':<30}  {'AUC':>8}  {'Time (s)':>10}")
print("─" * 60)
print(f"{'RandomForest (n=100, balanced)':<30}  {rf_auc:>8.4f}  {rf_time:>10.1f}")
print(f"{'LazyClassifier':<30}  {lazy_auc:>8.4f}  {lazy_time:>10.1f}")
delta = lazy_auc - rf_auc
sign = "+" if delta >= 0 else ""
print(f"\n  Δ AUC (Lazy − RF) = {sign}{delta:.4f}")
print(f"\n  ONNX roundtrip  Pearson r = {pearson_r:.6f}  max|diff| = {max_diff:.4f}")
print("═" * 60 + "\n")

assert pearson_r > 0.999, f"ONNX roundtrip broken: Pearson r = {pearson_r:.6f}"
print("Smoke test PASSED\n")

# -------------------------------------------------------------------
# Calibration plot
# -------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

rf_prob = rf.predict_proba(X_test)[:, 1]
lazy_prob = proba[:, 1]

fig, (ax_cal, ax_hist) = plt.subplots(
    2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
)

ax_cal.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
for prob, label, color in [
    (rf_prob, f"RandomForest  (AUC={rf_auc:.3f})", "#e15759"),
    (lazy_prob, f"LazyClassifier (AUC={lazy_auc:.3f})", "#4e79a7"),
]:
    frac_pos, mean_pred = calibration_curve(
        y_test, prob, n_bins=10, strategy="quantile"
    )
    ax_cal.plot(mean_pred, frac_pos, "o-", label=label, color=color, lw=1.5, ms=5)

ax_cal.set_ylabel("Fraction of positives")
ax_cal.set_title(f"Calibration — {DATASET[:12]} (test, pos_rate={y_test.mean():.1%})")
ax_cal.legend(fontsize=9)
ax_cal.set_ylim(-0.05, 1.05)
ax_cal.grid(True, alpha=0.3)

ax_hist.hist(rf_prob, bins=30, range=(0, 1), alpha=0.5, color="#e15759", label="RF")
ax_hist.hist(lazy_prob, bins=30, range=(0, 1), alpha=0.5, color="#4e79a7", label="Lazy")
ax_hist.set_xlabel("Mean predicted probability")
ax_hist.set_ylabel("Count")
ax_hist.legend(fontsize=8)
ax_hist.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "calibration_chembl_large.png")
plt.savefig(out_path, dpi=150)
print(f"Calibration plot saved to {out_path}\n")

# -------------------------------------------------------------------
# Test-set score swarm plot
# -------------------------------------------------------------------
rng = np.random.default_rng(42)
fig, ax = plt.subplots(figsize=(6, 5))

inactive_mask = y_test == 0
active_mask = y_test == 1

x_inactive = rng.normal(loc=0.0, scale=0.06, size=int(inactive_mask.sum()))
x_active = rng.normal(loc=1.0, scale=0.06, size=int(active_mask.sum()))

ax.scatter(
    x_inactive,
    lazy_prob[inactive_mask],
    s=12,
    alpha=0.35,
    color="#e15759",
    edgecolors="none",
    label=f"Inactive (n={inactive_mask.sum():,})",
)
ax.scatter(
    x_active,
    lazy_prob[active_mask],
    s=12,
    alpha=0.45,
    color="#4e79a7",
    edgecolors="none",
    label=f"Active (n={active_mask.sum():,})",
)

ax.set_xticks([0, 1], ["Inactive", "Active"])
ax.set_xlim(-0.35, 1.35)
ax.set_ylim(-0.02, 1.02)
ax.set_ylabel("Predicted probability")
ax.set_title(f"LazyClassifier test-set scores — {DATASET[:12]}")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
swarm_path = os.path.join(os.path.dirname(__file__), "swarm_chembl_large.png")
plt.savefig(swarm_path, dpi=150)
print(f"Swarm plot saved to {swarm_path}\n")

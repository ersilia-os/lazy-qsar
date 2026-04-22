#!/usr/bin/env python
"""
Smoke test on the AMES mutagenicity dataset (pre-computed Morgan fingerprints).

Compares LazyClassifier vs vanilla RandomForest on the same features.

Usage:
    python smoke_ames.py
"""

import os
import time
import tempfile
import csv

import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from xgboost import XGBClassifier

from lazyqsar.utils.logging import logger
from lazyqsar.utils.metrics import composite_metrics

logger.set_verbosity(True)

from lazyqsar.agnostic import LazyClassifier

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data")


def _load(split):
    with h5py.File(os.path.join(DATA_DIR, f"ames_{split}.h5"), "r") as f:
        X = f["Values"][:]
    with open(os.path.join(DATA_DIR, f"ames_{split}.csv")) as f:
        reader = csv.reader(f)
        next(reader)
        y = np.array([int(row[1]) for row in reader], dtype=int)
    return X.astype("float32"), y


X_train, y_train = _load("train")
X_test, y_test = _load("test")

print(
    f"\nAMES mutagenicity  —  train: {len(y_train):,}  test: {len(y_test):,}  "
    f"features: {X_train.shape[1]:,}  pos_rate: {y_train.mean():.1%}\n"
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
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_metrics = composite_metrics(y_test, rf_prob)
rf_auc = rf_metrics["auroc"]
print(
    f"  AUC = {rf_metrics['auroc']:.4f}   "
    f"AUPR = {rf_metrics['aupr']:.4f}   "
    f"BEDROC = {rf_metrics['bedroc']:.4f}   "
    f"Composite = {rf_metrics['composite']:.4f}   "
    f"fit time = {rf_time:.1f}s\n"
)

# -------------------------------------------------------------------
# Baseline: vanilla Logistic Regression
# -------------------------------------------------------------------
print("── Logistic Regression (baseline) " + "─" * 34)
t0 = time.perf_counter()
scaler = MaxAbsScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
lr = LogisticRegression(
    C=0.1,
    solver="saga",
    penalty="l1",
    class_weight="balanced",
    max_iter=10_000,
    random_state=42,
)
lr.fit(X_train_sc, y_train)
lr_time = time.perf_counter() - t0
lr_prob = lr.predict_proba(X_test_sc)[:, 1]
lr_metrics = composite_metrics(y_test, lr_prob)
lr_auc = lr_metrics["auroc"]
print(
    f"  AUC = {lr_metrics['auroc']:.4f}   "
    f"AUPR = {lr_metrics['aupr']:.4f}   "
    f"BEDROC = {lr_metrics['bedroc']:.4f}   "
    f"Composite = {lr_metrics['composite']:.4f}   "
    f"fit time = {lr_time:.1f}s\n"
)

# -------------------------------------------------------------------
# Baseline: vanilla XGBoost
# -------------------------------------------------------------------
print("── XGBoost (baseline) " + "─" * 45)
t0 = time.perf_counter()
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)
xgb.fit(X_train, y_train)
xgb_time = time.perf_counter() - t0
xgb_prob = xgb.predict_proba(X_test)[:, 1]
xgb_metrics = composite_metrics(y_test, xgb_prob)
xgb_auc = xgb_metrics["auroc"]
print(
    f"  AUC = {xgb_metrics['auroc']:.4f}   "
    f"AUPR = {xgb_metrics['aupr']:.4f}   "
    f"BEDROC = {xgb_metrics['bedroc']:.4f}   "
    f"Composite = {xgb_metrics['composite']:.4f}   "
    f"fit time = {xgb_time:.1f}s\n"
)

# -------------------------------------------------------------------
# LazyClassifier
# -------------------------------------------------------------------
print("── LazyClassifier " + "─" * 50)
t0 = time.perf_counter()
clf = LazyClassifier()
clf.fit(X=X_train, y=y_train)
lazy_time = time.perf_counter() - t0

proba = clf.predict_proba(X=X_test)
lazy_prob = proba[:, 1]
lazy_metrics = composite_metrics(y_test, lazy_prob)
lazy_auc = lazy_metrics["auroc"]

# -------------------------------------------------------------------
# ONNX roundtrip check
# -------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    model_dir = os.path.join(tmp, "model")
    clf.save(model_dir)
    artifact = LazyClassifier.load(model_dir)
    proba_onnx = artifact.predict_proba(X=X_test)

from scipy.stats import pearsonr

pearson_r, _ = pearsonr(proba[:, 1], proba_onnx[:, 1])
max_diff = float(np.abs(proba[:, 1] - proba_onnx[:, 1]).max())

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
print("\n" + "═" * 60)
print(
    f"{'Method':<30}  {'AUC':>8}  {'AUPR':>8}  {'BEDROC':>8}  {'Cons.':>8}  {'Time (s)':>10}"
)
print("─" * 60)
print(
    f"{'RandomForest (n=100, balanced)':<30}  "
    f"{rf_metrics['auroc']:>8.4f}  {rf_metrics['aupr']:>8.4f}  "
    f"{rf_metrics['bedroc']:>8.4f}  {rf_metrics['composite']:>8.4f}  {rf_time:>10.1f}"
)
print(
    f"{'LogisticRegression (L1, C=0.1)':<30}  "
    f"{lr_metrics['auroc']:>8.4f}  {lr_metrics['aupr']:>8.4f}  "
    f"{lr_metrics['bedroc']:>8.4f}  {lr_metrics['composite']:>8.4f}  {lr_time:>10.1f}"
)
print(
    f"{'XGBoost (default-ish)':<30}  "
    f"{xgb_metrics['auroc']:>8.4f}  {xgb_metrics['aupr']:>8.4f}  "
    f"{xgb_metrics['bedroc']:>8.4f}  {xgb_metrics['composite']:>8.4f}  {xgb_time:>10.1f}"
)
print(
    f"{'LazyClassifier':<30}  "
    f"{lazy_metrics['auroc']:>8.4f}  {lazy_metrics['aupr']:>8.4f}  "
    f"{lazy_metrics['bedroc']:>8.4f}  {lazy_metrics['composite']:>8.4f}  {lazy_time:>10.1f}"
)
delta = lazy_auc - rf_auc
sign = "+" if delta >= 0 else ""
print(f"\n  Δ AUC (Lazy − RF) = {sign}{delta:.4f}")
delta_composite = lazy_metrics["composite"] - rf_metrics["composite"]
sign_composite = "+" if delta_composite >= 0 else ""
print(f"  Δ Composite (Lazy − RF) = {sign_composite}{delta_composite:.4f}")
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

fig, (ax_cal, ax_hist) = plt.subplots(
    2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
)

# Reliability diagram
ax_cal.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
for prob, label, color in [
    (rf_prob, f"RandomForest  (AUC={rf_auc:.3f})", "#e15759"),
    (lr_prob, f"LogisticRegr.  (AUC={lr_auc:.3f})", "#59a14f"),
    (xgb_prob, f"XGBoost       (AUC={xgb_auc:.3f})", "#f28e2b"),
    (lazy_prob, f"LazyClassifier (AUC={lazy_auc:.3f})", "#4e79a7"),
]:
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10, strategy="uniform")
    ax_cal.plot(mean_pred, frac_pos, "o-", label=label, color=color, lw=1.5, ms=5)

ax_cal.set_ylabel("Fraction of positives")
ax_cal.set_title("Calibration — AMES mutagenicity (test set)")
ax_cal.legend(fontsize=9)
ax_cal.set_ylim(-0.05, 1.05)
ax_cal.grid(True, alpha=0.3)

# Prediction histogram
ax_hist.hist(rf_prob, bins=20, range=(0, 1), alpha=0.5, color="#e15759", label="RF")
ax_hist.hist(lr_prob, bins=20, range=(0, 1), alpha=0.5, color="#59a14f", label="LR")
ax_hist.hist(xgb_prob, bins=20, range=(0, 1), alpha=0.5, color="#f28e2b", label="XGB")
ax_hist.hist(lazy_prob, bins=20, range=(0, 1), alpha=0.5, color="#4e79a7", label="Lazy")
ax_hist.set_xlabel("Mean predicted probability")
ax_hist.set_ylabel("Count")
ax_hist.legend(fontsize=8)
ax_hist.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "calibration_ames.png")
plt.savefig(out_path, dpi=150)
print(f"Calibration plot saved to {out_path}\n")

# -------------------------------------------------------------------
# Test-set swarm plots: raw score, proba, logit, lift, rank
# -------------------------------------------------------------------
import matplotlib.patheffects as pe

score = clf.predict_score(X=X_test)[:, 1]
logit = clf.predict_logit(X=X_test)[:, 1]
lift = clf.predict_lift(X=X_test)[:, 1]
rank = clf.predict_rank(X=X_test)[:, 1]

panels = [
    ("Raw score", score, None),
    ("Proba", lazy_prob, None),
    ("Logit", logit, None),
    ("Lift", lift, None),
    ("Rank", rank, (0, 1)),
]

rng = np.random.default_rng(42)
fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), sharey=False)

for ax, (title, vals, ylim) in zip(axes, panels):
    for label, color in [(0, "#e15759"), (1, "#4e79a7")]:
        v = vals[y_test == label]
        jitter = rng.uniform(-0.15, 0.15, size=len(v))
        ax.scatter(
            np.full(len(v), label) + jitter, v, c=color, alpha=0.5, s=18, linewidths=0
        )
        ax.plot(
            [label - 0.25, label + 0.25],
            [np.median(v), np.median(v)],
            color=color,
            lw=2.5,
            path_effects=[pe.Stroke(linewidth=4, foreground="white"), pe.Normal()],
        )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["neg", "pos"])
    if ylim:
        ax.set_ylim(*ylim)

fig.suptitle(
    f"AMES mutagenicity — n_test={len(y_test)}  pos_rate={y_test.mean():.1%}  AUC={lazy_auc:.3f}",
    fontsize=11,
)
fig.tight_layout()
swarm_path = os.path.join(os.path.dirname(__file__), "swarm_ames.png")
fig.savefig(swarm_path, dpi=150)
print(f"Swarm plots saved to {swarm_path}\n")
plt.close(fig)

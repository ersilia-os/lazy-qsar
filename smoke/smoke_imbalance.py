#!/usr/bin/env python
"""
Smoke test for LazyClassifier on a moderately imbalanced dataset (2% positives).

Verifies:
  - No imbalance batching is triggered
  - All predict_* methods work on a single-batch fit
  - Composite metrics can be evaluated on the test set
  - ONNX roundtrip remains consistent

Usage:
    python smoke_imbalance.py
"""

import os
import tempfile

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split

from lazyqsar.agnostic import LazyClassifier
from lazyqsar.utils.logging import logger
from lazyqsar.utils.metrics import composite_metrics

logger.set_verbosity(True)


# -------------------------------------------------------------------
# Data: exactly 2% positives
# -------------------------------------------------------------------
rng = np.random.default_rng(42)
n_samples = 5_000
n_pos = 100
n_neg = n_samples - n_pos
n_features = 50

X_pos = (rng.standard_normal((n_pos, n_features)) + 1.25).astype("float32")
X_neg = rng.standard_normal((n_neg, n_features)).astype("float32")
X = np.vstack([X_pos, X_neg])
y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])

idx = rng.permutation(len(y))
X, y = X[idx], y[idx]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pos_rate = float(np.mean(y_train))
neg_pos_ratio = float(np.sum(y_train == 0) / np.sum(y_train == 1))

print("\nImbalance smoke test (2% positives)")
print(
    f"  train : {len(y_train):,}  positives: {int(y_train.sum())}  "
    f"negatives: {int((y_train == 0).sum())}  pos_rate: {pos_rate:.2%}  "
    f"neg/pos: {neg_pos_ratio:.1f}:1"
)
print(
    f"  test  : {len(y_test):,}   positives: {int(y_test.sum())}  "
    f"negatives: {int((y_test == 0).sum())}\n"
)


# -------------------------------------------------------------------
# Fit
# -------------------------------------------------------------------
clf = LazyClassifier()
clf.fit(X=X_train, y=y_train)


# -------------------------------------------------------------------
# Single-batch sanity check
# -------------------------------------------------------------------
pop_prior = clf._model.population_prior_
batch_priors = clf._model.batch_priors_
decision_cutoff = clf._model.decision_cutoff_

print("\n── Batching diagnostics " + "─" * 42)
print(f"  population_prior  : {pop_prior:.6f}")
print(f"  batch_priors      : {[f'{p:.4f}' for p in batch_priors]}")
print(f"  num_batches       : {len(batch_priors)}")
print(f"  decision_cutoff   : {decision_cutoff:.4f}")
print()

assert abs(pop_prior - 0.02) < 0.0025, f"population_prior drifted: {pop_prior:.6f}"
assert len(batch_priors) == 1, f"Expected one batch, got {len(batch_priors)}"
assert abs(batch_priors[0] - pop_prior) < 1e-9, (
    f"Single-batch prior should match population prior: "
    f"batch_prior={batch_priors[0]:.6f}, pop_prior={pop_prior:.6f}"
)


# -------------------------------------------------------------------
# predict_* on test set
# -------------------------------------------------------------------
proba = clf.predict_proba(X=X_test)
score = clf.predict_score(X=X_test)
logit = clf.predict_logit(X=X_test)
lift = clf.predict_lift(X=X_test)
rank = clf.predict_rank(X=X_test)
preds = clf.predict(X=X_test)

print("── predict_* shape and range checks " + "─" * 32)
for name, arr in [
    ("proba", proba),
    ("score", score),
    ("logit", logit),
    ("lift", lift),
    ("rank", rank),
    ("predict", preds),
]:
    if arr.ndim == 2:
        print(
            f"  {name:<10}  shape={arr.shape}  "
            f"min={arr.min():.4f}  max={arr.max():.4f}  "
            f"mean_pos={arr[y_test == 1, 1].mean():.4f}  "
            f"mean_neg={arr[y_test == 0, 1].mean():.4f}"
        )
    else:
        print(
            f"  {name:<10}  shape={arr.shape}  "
            f"min={arr.min():.4f}  max={arr.max():.4f}  "
            f"mean={arr.mean():.4f}"
        )

assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), "predict_proba not normalized"
assert np.allclose(score.sum(axis=1), 1.0, atol=1e-6), "predict_score not normalized"
assert rank[:, 1].min() >= 0.0 and rank[:, 1].max() <= 1.0, "rank out of [0,1]"
assert logit[y_test == 1, 1].mean() > logit[y_test == 0, 1].mean(), (
    "logit does not separate classes"
)
assert lift[y_test == 1, 1].mean() > lift[y_test == 0, 1].mean(), (
    "lift does not separate classes"
)


# -------------------------------------------------------------------
# Test metrics
# -------------------------------------------------------------------
metrics = composite_metrics(y_test, proba[:, 1])
print("\n── Test metrics " + "─" * 49)
print(f"  AUROC      : {metrics['auroc']:.4f}")
print(f"  AUPR       : {metrics['aupr']:.4f}")
print(f"  BEDROC     : {metrics['bedroc']:.4f}")
print(f"  Composite  : {metrics['composite']:.4f}")


# -------------------------------------------------------------------
# ONNX roundtrip
# -------------------------------------------------------------------
print("\n── ONNX roundtrip " + "─" * 50)
with tempfile.TemporaryDirectory() as tmp:
    model_dir = os.path.join(tmp, "model")
    clf.save(model_dir)
    artifact = LazyClassifier.load(model_dir)
    proba_onnx = artifact.predict_proba(X=X_test)

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


# -------------------------------------------------------------------
# Calibration plot
# -------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.calibration import calibration_curve

fig, (ax_cal, ax_hist) = plt.subplots(
    2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
)

ax_cal.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
frac_pos, mean_pred = calibration_curve(
    y_test, proba[:, 1], n_bins=10, strategy="uniform"
)
ax_cal.plot(
    mean_pred,
    frac_pos,
    "o-",
    label=f"LazyClassifier (AUROC={metrics['auroc']:.3f}, Composite={metrics['composite']:.3f})",
    color="#4e79a7",
    lw=1.5,
    ms=5,
)
ax_cal.set_ylabel("Fraction of positives")
ax_cal.set_title("Calibration — 2% imbalance smoke test (test set)")
ax_cal.legend(fontsize=9)
ax_cal.set_ylim(-0.05, 1.05)
ax_cal.grid(True, alpha=0.3)

ax_hist.hist(
    proba[:, 1], bins=20, range=(0, 1), alpha=0.7, color="#4e79a7", label="Lazy"
)
ax_hist.set_xlabel("Mean predicted probability")
ax_hist.set_ylabel("Count")
ax_hist.legend(fontsize=8)
ax_hist.grid(True, alpha=0.3)

plt.tight_layout()
calibration_path = os.path.join(os.path.dirname(__file__), "calibration_imbalance.png")
plt.savefig(calibration_path, dpi=150)
print(f"Calibration plot saved to {calibration_path}\n")
plt.close(fig)


# -------------------------------------------------------------------
# Test-set swarm plots: raw score, proba, logit, lift, rank
# -------------------------------------------------------------------
panels = [
    ("Raw score", score[:, 1], None),
    ("Proba", proba[:, 1], None),
    ("Logit", logit[:, 1], None),
    ("Lift", lift[:, 1], None),
    ("Rank", rank[:, 1], (0, 1)),
]

rng2 = np.random.default_rng(42)
fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), sharey=False)

for ax, (title, vals, ylim) in zip(axes, panels):
    for label, color in [(0, "#e15759"), (1, "#4e79a7")]:
        v = vals[y_test == label]
        jitter = rng2.uniform(-0.15, 0.15, size=len(v))
        ax.scatter(
            np.full(len(v), label) + jitter,
            v,
            c=color,
            alpha=0.5,
            s=18,
            linewidths=0,
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
    f"Imbalance 2% — n_test={len(y_test)}  pos_rate={y_test.mean():.2%}  "
    f"AUROC={metrics['auroc']:.3f}  Composite={metrics['composite']:.3f}",
    fontsize=11,
)
fig.tight_layout()
swarm_path = os.path.join(os.path.dirname(__file__), "swarm_imbalance.png")
fig.savefig(swarm_path, dpi=150)
print(f"Swarm plots saved to {swarm_path}\n")
plt.close(fig)

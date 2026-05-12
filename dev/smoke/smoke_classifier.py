#!/usr/bin/env python
"""
Smoke test for LazyClassifier — fit, predict, ONNX roundtrip.
Shows the per-step timing table (requires verbose mode).

Usage:
    python smoke_classifier.py           # 500 samples, 100 features
    python smoke_classifier.py 1000 200  # custom n_samples n_features
"""

import sys
import os
import tempfile

import numpy as np

from lazyqsar.utils.logging import logger

logger.set_verbosity(True)  # enable timing table + all banners

from lazyqsar.agnostic import LazyClassifier

# -------------------------------------------------------------------
# Data
# -------------------------------------------------------------------
n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 500
n_features = int(sys.argv[2]) if len(sys.argv) > 2 else 100

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=min(20, n_features // 5),
    n_redundant=min(10, n_features // 10),
    flip_y=0.05,
    random_state=42,
)
X = X.astype("float32")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(
    f"\nSmoke test  n_train={len(y_train)}  n_test={len(y_test)}  p={n_features}  pos={y_train.mean():.0%}\n"
)

# -------------------------------------------------------------------
# Fit  (timing table printed at the end when verbose=True)
# -------------------------------------------------------------------
clf = LazyClassifier(max_rounds=500)
clf.fit(X=X_train, y=y_train)

# -------------------------------------------------------------------
# Predict on held-out test set
# -------------------------------------------------------------------
proba = clf.predict_proba(X=X_test)
preds = clf.predict(X=X_test)
print(f"\npredict_proba shape : {proba.shape}")
print(f"predict shape       : {preds.shape}")
print(f"mean P(1) on test   : {proba[:, 1].mean():.4f}")

# -------------------------------------------------------------------
# ONNX save / load roundtrip (compare on test set — no calibration overfit)
# -------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    model_dir = os.path.join(tmp, "model")
    clf.save(model_dir)
    artifact = LazyClassifier.load(model_dir)  # loads ONNX artifact
    proba_onnx = artifact.predict_proba(X=X_test)

from scipy.stats import pearsonr, spearmanr

p1_mem = proba[:, 1]
p1_onnx = proba_onnx[:, 1]

pearson_r, _ = pearsonr(p1_mem, p1_onnx)
spearman_r, _ = spearmanr(p1_mem, p1_onnx)
max_diff = float(np.abs(p1_mem - p1_onnx).max())
mean_diff = float(np.abs(p1_mem - p1_onnx).mean())

print(f"\nONNX roundtrip — in-memory vs ONNX artifact (n_test={len(p1_mem)})")
print(f"  Pearson  r          : {pearson_r:.6f}")
print(f"  Spearman r          : {spearman_r:.6f}")
print(f"  mean |diff|         : {mean_diff:.4f}")
print(f"  max  |diff|         : {max_diff:.4f}")
print()

assert pearson_r > 0.999, f"Pearson r too low: {pearson_r:.6f}"
assert spearman_r > 0.999, f"Spearman r too low: {spearman_r:.6f}"

from sklearn.metrics import roc_auc_score

test_auc = roc_auc_score(y_test, p1_mem)
print(f"  test AUC (in-memory) : {test_auc:.4f}")
assert test_auc > 0.65, f"Test AUC too low ({test_auc:.4f}) — model may not be learning"

# -------------------------------------------------------------------
# Swarm plots: raw score, proba, logit, lift, rank  (vs true label)
# -------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    score = clf.predict_score(X=X_test)[:, 1]
    proba_ = clf.predict_proba(X=X_test)[:, 1]
    logit = clf.predict_logit(X=X_test)[:, 1]
    lift = clf.predict_lift(X=X_test)[:, 1]
    rank = clf.predict_rank(X=X_test)[:, 1]

    panels = [
        ("Raw score", score, None),
        ("Proba", proba_, None),
        ("Logit", logit, None),
        ("Lift", lift, None),
        ("Rank", rank, (0, 1)),
    ]

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), sharey=False)

    for ax, (title, vals, ylim) in zip(axes, panels):
        for label, color in [(0, "#4C72B0"), (1, "#DD8452")]:
            v = vals[y_test == label]
            jitter = rng.uniform(-0.15, 0.15, size=len(v))
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
        f"n_test={len(y_test)}  pos_rate={y_test.mean():.1%}  AUC={test_auc:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    plot_path = "smoke_swarm.png"
    fig.savefig(plot_path, dpi=120)
    print(f"\nSwarm plots saved → {plot_path}")
    plt.close(fig)
except ImportError:
    print("\n(matplotlib not available — skipping swarm plots)")

print("Smoke test PASSED\n")

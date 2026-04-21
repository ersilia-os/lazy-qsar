#!/usr/bin/env python
"""
Smoke test: LazyClassifierQSAR (fast mode) on the bioavailability_ma dataset.

Uses morgan + rdkit descriptors and logs which descriptor contributed most per
molecule via applicability-domain softmax weights.

To run with slow mode (chemeleon, morgan, rdkit, cddd), install the full
descriptor extras and ensure a compatible environment:
    pip install -e ".[all]"

Usage:
    python smoke/smoke_bioavailability_qsar.py
"""

import csv
import os
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from lazyqsar.utils.logging import logger
from lazyqsar.qsar import LazyClassifierQSAR, _softmax_weights

logger.set_verbosity(True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "data")
OUT_DIR = os.path.dirname(__file__)
MODE = "fast"  # change to "slow" when chemeleon/chemprop is working


# ── helpers ──────────────────────────────────────────────────────────────────


def _load_csv(split):
    path = os.path.join(DATA_DIR, f"bioavailability_ma_{split}.csv")
    smiles, labels = [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            smiles.append(row[0])
            labels.append(int(row[1]))
    return smiles, np.array(labels, dtype=int)


def _per_descriptor_breakdown(model, smiles_list):
    """
    Return per-descriptor predictions and AD scores without re-running
    descriptor computation (uses the internal feature cache).

    Returns
    -------
    desc_names : list[str]
    preds      : ndarray (D, B)  — P(y=1) per descriptor
    ad_scores  : ndarray (B, D)  — AD score per descriptor
    weights    : ndarray (B, D)  — softmax weights per molecule
    """
    preds, ad_scores = [], []
    for i, (mod, ad) in enumerate(zip(model.models, model.ad_models)):
        X = model._transform_cached(i, smiles_list)
        preds.append(mod.predict_proba(X=X)[:, 1])
        ad_scores.append(ad.score(X))

    P = np.stack(preds, axis=0)  # (D, B)
    A = np.stack(ad_scores, axis=0).T  # (B, D)
    W = _softmax_weights(A)  # (B, D)
    return model.descriptor_types, P, A, W


def _print_descriptor_table(desc_names, P, A, W, y_true, combined_auc):
    width = 78
    print("\n" + "═" * width)
    print(
        f"  {'Descriptor':<12}  {'AUC (solo)':>10}  {'AD mean±std':>14}  "
        f"{'Weight mean±std':>16}  {'Wins':>6}"
    )
    print("─" * width)
    for d, name in enumerate(desc_names):
        solo_auc = roc_auc_score(y_true, P[d])
        ad_mean, ad_std = float(A[:, d].mean()), float(A[:, d].std())
        w_mean, w_std = float(W[:, d].mean()), float(W[:, d].std())
        wins = int((W.argmax(axis=1) == d).sum())
        print(
            f"  {name:<12}  {solo_auc:>10.4f}  "
            f"{ad_mean:>6.3f}±{ad_std:<6.3f}  "
            f"{w_mean:>7.3f}±{w_std:<7.3f}  "
            f"{wins:>6}"
        )
    print("─" * width)
    print(f"  {'combined':<12}  {combined_auc:>10.4f}")
    print("═" * width + "\n")


def _print_winner_breakdown(desc_names, W):
    B = W.shape[0]
    winner_idx = W.argmax(axis=1)
    print("  AD-weight winner per molecule:")
    for d, name in enumerate(desc_names):
        n = int((winner_idx == d).sum())
        bar = "█" * int(round(n / B * 40))
        print(f"    {name:<12}  {n:>4} / {B}  ({n / B:>5.1%})  {bar}")
    print()


def _print_low_confidence_molecules(smiles, desc_names, W, A, n=5):
    """Molecules where all descriptors have low AD scores."""
    max_ad = A.max(axis=1)
    worst_idx = np.argsort(max_ad)[:n]
    print(f"  Top {n} molecules with lowest best-descriptor AD score (hardest cases):")
    for rank, idx in enumerate(worst_idx, 1):
        best_d = int(A[idx].argmax())
        scores_str = "  ".join(
            f"{name}={A[idx, d]:.3f}{'*' if d == best_d else ' '}"
            for d, name in enumerate(desc_names)
        )
        smi = smiles[idx][:50]
        print(f"    {rank}. [{max_ad[idx]:.3f}]  {smi:<50}  {scores_str}")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

print("\n" + "═" * 78)
print(f"  LazyClassifierQSAR  ·  mode={MODE}  ·  Bioavailability (Ma)")
print("═" * 78 + "\n")

smiles_train, y_train = _load_csv("train")
smiles_test, y_test = _load_csv("test")

print(f"  Train: {len(y_train):,} molecules  (pos={y_train.mean():.1%})")
print(f"  Test : {len(y_test):,} molecules   (pos={y_test.mean():.1%})\n")

# ── Fit ───────────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
model = LazyClassifierQSAR(mode=MODE)
model.fit(smiles_train, y_train)
fit_time = time.perf_counter() - t0
print(f"\n  Fit complete in {fit_time:.1f}s\n")

# ── Predict + per-descriptor breakdown ───────────────────────────────────────
t1 = time.perf_counter()
proba_combined = model.predict_proba(smiles_test)
pred_time = time.perf_counter() - t1
combined_auc = roc_auc_score(y_test, proba_combined[:, 1])

desc_names, P, A_test, W_test = _per_descriptor_breakdown(model, smiles_test)

print(f"  Prediction complete in {pred_time:.2f}s\n")
print("  ── Per-descriptor performance & AD statistics (test set) ──")
_print_descriptor_table(desc_names, P, A_test, W_test, y_test, combined_auc)

print("  ── AD-weight winner distribution ──")
_print_winner_breakdown(desc_names, W_test)

print("  ── Hard-case molecules (low AD score across all descriptors) ──")
_print_low_confidence_molecules(smiles_test, desc_names, W_test, A_test, n=5)

# ── ONNX save / load round-trip ───────────────────────────────────────────────
print("  ── ONNX save / load round-trip ──")
with tempfile.TemporaryDirectory() as tmp:
    model_dir = os.path.join(tmp, "model")
    t_save = time.perf_counter()
    model.save(model_dir)
    save_time = time.perf_counter() - t_save

    for desc in desc_names:
        ad_onnx = os.path.join(
            model_dir, desc, "applicability_domain", "applicability_domain.onnx"
        )
        assert os.path.isfile(ad_onnx), f"Missing AD ONNX for {desc}"
    print(
        f"  Saved in {save_time:.1f}s — AD ONNX present for all {len(desc_names)} descriptors"
    )

    t_load = time.perf_counter()
    artifact = LazyClassifierQSAR.load(model_dir)
    load_time = time.perf_counter() - t_load
    print(f"  Loaded in {load_time:.1f}s")

    proba_onnx = artifact.predict_proba(smiles_test)
    onnx_auc = roc_auc_score(y_test, proba_onnx[:, 1])
    max_diff = float(np.abs(proba_combined[:, 1] - proba_onnx[:, 1]).max())
    print(f"  ONNX AUC = {onnx_auc:.4f}  max|diff vs sklearn| = {max_diff:.4e}\n")

# ── Plots ─────────────────────────────────────────────────────────────────────
D = len(desc_names)
fig, axes = plt.subplots(1, 3, figsize=(5 * 3, 4))

ax = axes[0]
for d, name in enumerate(desc_names):
    ax.hist(A_test[:, d], bins=25, alpha=0.6, label=name, density=True)
ax.set_xlabel("AD score")
ax.set_ylabel("Density")
ax.set_title("AD score distributions (test set)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

ax = axes[1]
ax.boxplot([W_test[:, d] for d in range(D)], labels=desc_names, patch_artist=True)
ax.set_xlabel("Descriptor")
ax.set_ylabel("Softmax weight")
ax.set_title("AD-based weights per descriptor (test set)")
ax.grid(True, alpha=0.25, axis="y")

ax = axes[2]
solo_aucs = [roc_auc_score(y_test, P[d]) for d in range(D)]
colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"][:D]
bars = ax.bar(desc_names, solo_aucs, color=colors, alpha=0.8)
ax.axhline(
    combined_auc, color="black", lw=1.5, ls="--", label=f"Combined ({combined_auc:.3f})"
)
y_margin = 0.03
ax.set_ylim(max(0, min(solo_aucs) - y_margin), min(1.0, combined_auc + y_margin))
for bar, auc in zip(bars, solo_aucs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        auc + 0.002,
        f"{auc:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
ax.set_ylabel("AUC (ROC)")
ax.set_title("Solo vs combined AUC")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25, axis="y")

fig.suptitle(
    f"LazyClassifierQSAR {MODE} — Bioavailability (Ma)  "
    f"n_test={len(y_test)}  combined AUC={combined_auc:.3f}",
    fontsize=10,
)
fig.tight_layout()
plot_path = os.path.join(OUT_DIR, "smoke_bioavailability_qsar.png")
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"  Plot saved to {plot_path}\n")

# ── Final assertion ───────────────────────────────────────────────────────────
assert combined_auc > 0.6, f"Combined AUC too low: {combined_auc:.4f}"
print(f"  Smoke test PASSED  (combined AUC = {combined_auc:.4f})\n")

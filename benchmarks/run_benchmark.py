#!/usr/bin/env python
"""
Benchmark LazyClassifier (agnostic) and LazyClassifierQSAR against
baseline classifiers on the TDC ADMET binary classification suite.

Usage:
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --data-dir /path/to/tdc/binary
    python benchmarks/run_benchmark.py --mode agnostic --datasets bioavailability_ma ames
"""
import argparse
import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths and dataset list
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _REPO_ROOT.parent / "zeroshot-xgboost" / "data" / "tdc" / "binary"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "benchmarks" / "results"

DATASETS = [
    "ames",
    "bbb_martins",
    "bioavailability_ma",
    "carcinogens_lagunin",
    "clintox",
    "cyp1a2_veith",
    "cyp2c19_veith",
    "cyp2c9_substrate_carbonmangels",
    "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels",
    "cyp2d6_veith",
    "cyp3a4_substrate_carbonmangels",
    "cyp3a4_veith",
    "dili",
    "herg",
    "hia_hou",
    "pgp_broccatelli",
    "skin_reaction",
]

MODEL_ORDER = [
    "LazyClassifier",
    "LazyClassifier (no cal)",
    "LazyClassifier (ONNX)",
    "LR (default)",
    "XGB (default)",
    "RF (default)",
]

# Consistent color palette across all plots
_PALETTE = [
    "#2196F3",  # blue        — LazyClassifier
    "#9C27B0",  # purple      — LazyClassifier (no cal)
    "#03A9F4",  # light-blue  — LazyClassifier (ONNX)
    "#FF9800",  # orange      — LR
    "#F44336",  # red         — XGB
    "#4CAF50",  # green       — RF
]
MODEL_COLORS = {m: _PALETTE[i] for i, m in enumerate(MODEL_ORDER)}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dataset(path: Path):
    """Load a TDC .tab file → (smiles list, binary y array)."""
    df = pd.read_csv(path, sep="\t")
    smiles = df["Drug"].tolist()
    y = df["Y"].values.astype(int)
    return smiles, y


def compute_morgan(smiles_list, radius=2, n_bits=2048):
    """Compute Morgan fingerprints → float32 numpy array (n, n_bits)."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
        else:
            fp = gen.GetFingerprintAsNumPy(mol)
            fps.append(fp.astype(np.float32))
    return np.array(fps, dtype=np.float32)


# ---------------------------------------------------------------------------
# Calibration helper
# ---------------------------------------------------------------------------

def compute_ece(y_true, proba, n_bins=10):
    """Expected Calibration Error: weighted mean |actual_pos_rate - mean_pred_prob| per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - proba[mask].mean())
    return round(float(ece), 4)


# ---------------------------------------------------------------------------
# File tree printer
# ---------------------------------------------------------------------------

def _tree_lines(directory, prefix=""):
    """Yield lines for a file tree rooted at directory."""
    try:
        entries = sorted(os.listdir(directory))
    except PermissionError:
        return
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            yield f"{prefix}{connector}{entry}/"
            extension = "    " if i == len(entries) - 1 else "│   "
            yield from _tree_lines(path, prefix + extension)
        else:
            size_kb = os.path.getsize(path) / 1024
            yield f"{prefix}{connector}{entry}  ({size_kb:.1f} KB)"


def print_file_tree(directory, output_file=None):
    """Print (and optionally save) the file tree of a directory."""
    lines = [f"{os.path.basename(directory)}/"] + list(_tree_lines(directory))
    text = "\n".join(lines)
    print(f"\n{'─' * 60}")
    print("  Saved model directory structure:")
    print(f"{'─' * 60}")
    print(text)
    print(f"{'─' * 60}\n")
    if output_file:
        with open(output_file, "w") as f:
            f.write(text + "\n")


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

def _fit_predict(model, fit_fn, predict_fn):
    """Time fit and inference separately. Returns (proba, fit_time, infer_time)."""
    t0 = time.time()
    fit_fn(model)
    t_fit = round(time.time() - t0, 3)
    t1 = time.time()
    proba = predict_fn(model)
    t_infer = round(time.time() - t1, 4)
    return proba, t_fit, t_infer


def run_dataset(name, smiles, y, mode, output_dir, is_first_dataset=False):
    """
    Fit all requested models on one dataset.

    Returns:
        results  : list of metric dicts
        curves   : list of curve dicts
    """
    smiles_train, smiles_test, y_train, y_test = train_test_split(
        smiles, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = compute_morgan(smiles_train)
    X_test = compute_morgan(smiles_test)

    results = []
    curves = []

    def _record(model_name, proba, t_fit, t_infer, error=None,
                save_time=None, load_time=None):
        if error is not None:
            results.append({
                "dataset": name, "model": model_name,
                "roc_auc": np.nan, "pr_auc": np.nan, "ece": np.nan,
                "fit_time": np.nan, "infer_time": np.nan,
                "save_time": np.nan, "load_time": np.nan,
                "error": str(error)[:120],
            })
            return
        fpr, tpr, _ = roc_curve(y_test, proba)
        precision, recall, _ = precision_recall_curve(y_test, proba)
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
        results.append({
            "dataset": name, "model": model_name,
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "pr_auc": round(average_precision_score(y_test, proba), 4),
            "ece": compute_ece(y_test, proba),
            "fit_time": t_fit,
            "infer_time": t_infer,
            "save_time": save_time,
            "load_time": load_time,
        })
        curves.append({
            "dataset": name, "model": model_name,
            "fpr": fpr, "tpr": tpr,
            "precision": precision, "recall": recall,
            "cal_frac_pos": frac_pos, "cal_mean_pred": mean_pred,
        })

    lazy_clf = None  # keep reference for ONNX roundtrip

    if mode in ("agnostic", "both"):
        # LazyClassifier (agnostic, Morgan fingerprints)
        try:
            from lazyqsar.agnostic import LazyClassifier
            clf = LazyClassifier()
            proba, t_fit, t_infer = _fit_predict(
                clf,
                lambda m: m.fit(X=X_train, y=y_train),
                lambda m: m.predict_proba(X=X_test)[:, 1],
            )
            _record("LazyClassifier", proba, t_fit, t_infer)
            lazy_clf = clf
        except Exception as e:
            _record("LazyClassifier", None, None, None, error=e)

        # LazyClassifier without calibration — speed comparison
        try:
            from lazyqsar.agnostic import LazyClassifier
            clf_nocal = LazyClassifier(calibrated=False)
            proba, t_fit, t_infer = _fit_predict(
                clf_nocal,
                lambda m: m.fit(X=X_train, y=y_train),
                lambda m: m.predict_proba(X=X_test)[:, 1],
            )
            _record("LazyClassifier (no cal)", proba, t_fit, t_infer)
        except Exception as e:
            _record("LazyClassifier (no cal)", None, None, None, error=e)

        # LazyClassifier (ONNX) — save, load, infer
        if lazy_clf is not None:
            try:
                from lazyqsar.agnostic import LazyClassifier
                tmp_dir = tempfile.mkdtemp(prefix="lazy_onnx_")
                model_path = os.path.join(tmp_dir, "model")
                try:
                    t0 = time.time()
                    lazy_clf.save(model_path)
                    save_time = round(time.time() - t0, 3)

                    if is_first_dataset:
                        tree_file = os.path.join(output_dir, "model_structure.txt")
                        print_file_tree(model_path, output_file=tree_file)
                        print(f"  (structure saved → {tree_file})")

                    t1 = time.time()
                    artifact = LazyClassifier.load(model_path)
                    load_time = round(time.time() - t1, 3)

                    t2 = time.time()
                    proba_onnx = artifact.predict_proba(X_test)[:, 1]
                    t_infer_onnx = round(time.time() - t2, 4)

                    _record(
                        "LazyClassifier (ONNX)", proba_onnx,
                        t_fit=None, t_infer=t_infer_onnx,
                        save_time=save_time, load_time=load_time,
                    )
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as e:
                _record("LazyClassifier (ONNX)", None, None, None, error=e)

        # LR baseline
        try:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
            proba, t_fit, t_infer = _fit_predict(
                clf,
                lambda m: m.fit(X_train, y_train),
                lambda m: m.predict_proba(X_test)[:, 1],
            )
            _record("LR (default)", proba, t_fit, t_infer)
        except Exception as e:
            _record("LR (default)", None, None, None, error=e)

        # XGB baseline
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(eval_metric="logloss", verbosity=0, n_jobs=-1)
            proba, t_fit, t_infer = _fit_predict(
                clf,
                lambda m: m.fit(X_train, y_train),
                lambda m: m.predict_proba(X_test)[:, 1],
            )
            _record("XGB (default)", proba, t_fit, t_infer)
        except Exception as e:
            _record("XGB (default)", None, None, None, error=e)

        # RF baseline
        try:
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            proba, t_fit, t_infer = _fit_predict(
                clf,
                lambda m: m.fit(X_train, y_train),
                lambda m: m.predict_proba(X_test)[:, 1],
            )
            _record("RF (default)", proba, t_fit, t_infer)
        except Exception as e:
            _record("RF (default)", None, None, None, error=e)

    if mode in ("qsar", "both"):
        # LazyClassifierQSAR (SMILES → rdkit + morgan descriptors)
        try:
            from lazyqsar.qsar import LazyClassifierQSAR
            clf = LazyClassifierQSAR(mode="fast")
            proba, t_fit, t_infer = _fit_predict(
                clf,
                lambda m: m.fit(smiles_train, y_train),
                lambda m: m.predict_proba(smiles_test)[:, 1],
            )
            _record("LazyClassifierQSAR", proba, t_fit, t_infer)
        except Exception as e:
            _record("LazyClassifierQSAR", None, None, None, error=e)

    return results, curves


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def build_pivot(all_results, metric="roc_auc"):
    df = pd.DataFrame(all_results)
    present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    pivot = df.pivot_table(index="dataset", columns="model", values=metric)[present]
    return pivot


def mean_rank_table(pivot):
    ranks = pivot.rank(axis=1, ascending=False, na_option="bottom")
    return ranks.mean(axis=0).sort_values()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _bar_annotations(ax, rects, fmt="{:.3f}", fontsize=5, rotation=90):
    """Add value labels on top of each bar."""
    for rect in rects:
        h = rect.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            h + 0.005,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=fontsize, rotation=rotation,
        )


def plot_bars(pivot, output_path, title, annotate=True):
    import matplotlib.pyplot as plt

    models = list(pivot.columns)
    colors = [MODEL_COLORS.get(m, "#888888") for m in models]
    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(pivot))
    width = 0.8 / len(models)
    for i, (model, color) in enumerate(zip(models, colors)):
        offset = (i - len(models) / 2 + 0.5) * width
        rects = ax.bar(x + offset, pivot[model], width=width * 0.9,
                       label=model, color=color, alpha=0.85)
        if annotate:
            _bar_annotations(ax, rects)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(title.split("(")[0].strip(), fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_timing_bars(all_results, output_path):
    """Two-panel plot: fit_time (top) and infer_time (bottom), bars grouped by model."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame(all_results)
    datasets = sorted(df["dataset"].unique())
    models_fit = [m for m in MODEL_ORDER
                  if m != "LazyClassifier (ONNX)" and m in df["model"].unique()]
    models_infer = [m for m in MODEL_ORDER if m in df["model"].unique()]

    fig, (ax_fit, ax_infer) = plt.subplots(2, 1, figsize=(16, 8), sharex=False)

    def _draw_panel(ax, df_sub, models, metric, title, log_scale=True):
        x = np.arange(len(datasets))
        width = 0.8 / len(models)
        for i, model in enumerate(models):
            vals = []
            for ds in datasets:
                row = df_sub[(df_sub["dataset"] == ds) & (df_sub["model"] == model)]
                vals.append(float(row[metric].iloc[0]) if len(row) > 0 and not row[metric].isna().all() else np.nan)
            offset = (i - len(models) / 2 + 0.5) * width
            color = MODEL_COLORS.get(model, "#888888")
            ax.bar(x + offset, vals, width=width * 0.9, label=model, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(metric.replace("_", " ") + " (s)", fontsize=9)
        ax.set_title(title, fontsize=10)
        if log_scale:
            ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(axis="y", alpha=0.3)

    _draw_panel(ax_fit, df, models_fit, "fit_time", "Fit time by dataset (log scale)", log_scale=True)
    _draw_panel(ax_infer, df, models_infer, "infer_time", "Inference time by dataset (log scale)", log_scale=True)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_radar(all_results, output_path):
    """Radar chart: mean ROC-AUC, PR-AUC, 1-ECE, speed (1/fit_time normalized)."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame(all_results)
    # Only models with fit_time (exclude ONNX for speed axis)
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    categories = ["ROC-AUC", "PR-AUC", "1 - ECE", "Speed"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    # Compute per-model mean metrics
    model_vals = {}
    for model in models:
        sub = df[df["model"] == model]
        roc = sub["roc_auc"].mean()
        pr = sub["pr_auc"].mean()
        ece = sub["ece"].mean()
        ft = sub["fit_time"].mean()
        model_vals[model] = (roc, pr, 1 - ece, ft)

    # Normalize speed: 1/fit_time, scaled so max = 1
    speeds_raw = {m: 1.0 / v[3] if (v[3] and not np.isnan(v[3])) else 0.0
                  for m, v in model_vals.items()}
    max_speed = max(speeds_raw.values()) if speeds_raw else 1.0
    if max_speed == 0:
        max_speed = 1.0

    for model in models:
        roc, pr, one_minus_ece, _ = model_vals[model]
        speed = speeds_raw[model] / max_speed
        values = [roc, pr, one_minus_ece, speed]
        values += values[:1]
        color = MODEL_COLORS.get(model, "#888888")
        ax.plot(angles, values, color=color, linewidth=1.8, label=model)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.set_title("Model comparison (mean across datasets)", fontsize=11, pad=18)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rank_scatter(pivot_roc, output_path):
    """Horizontal boxplot of per-dataset ROC-AUC rank, with individual points."""
    import matplotlib.pyplot as plt

    models = list(pivot_roc.columns)
    ranks = pivot_roc.rank(axis=1, ascending=False, na_option="bottom")

    fig, ax = plt.subplots(figsize=(8, max(4, len(models) * 0.8)))

    for i, model in enumerate(reversed(models)):
        vals = ranks[model].dropna().values
        ax.boxplot(
            vals, positions=[i], vert=False, widths=0.5,
            patch_artist=True,
            boxprops={"facecolor": MODEL_COLORS.get(model, "#888"), "alpha": 0.4},
            medianprops={"color": MODEL_COLORS.get(model, "#888"), "linewidth": 2},
            whiskerprops={"color": "#555"},
            capprops={"color": "#555"},
            flierprops={"marker": ""},
        )
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(vals, np.full(len(vals), i) + jitter,
                   color=MODEL_COLORS.get(model, "#888"), s=18, alpha=0.7, zorder=3)

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(list(reversed(models)), fontsize=9)
    ax.set_xlabel("Rank by ROC-AUC (lower = better)", fontsize=10)
    ax.set_title("Per-dataset rank distribution", fontsize=11)
    ax.axvline(1, color="#aaa", lw=0.8, ls="--")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(all_curves, output_path):
    import matplotlib.pyplot as plt

    datasets = sorted({c["dataset"] for c in all_curves})
    models = [m for m in MODEL_ORDER if any(c["model"] == m for c in all_curves)]
    n = len(datasets)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()

    for ax_idx, ds in enumerate(datasets):
        ax = axes[ax_idx]
        ds_curves = [c for c in all_curves if c["dataset"] == ds]
        for c in ds_curves:
            ax.plot(c["fpr"], c["tpr"], label=c["model"],
                    color=MODEL_COLORS.get(c["model"], "#888"), lw=1.2)
        ax.plot([0, 1], [0, 1], "k--", lw=0.5)
        ax.set_title(ds, fontsize=8)
        ax.set_xlabel("FPR", fontsize=7)
        ax.set_ylabel("TPR", fontsize=7)
        ax.tick_params(labelsize=6)

    for ax_idx in range(len(datasets), len(axes)):
        axes[ax_idx].set_visible(False)

    handles = [plt.Line2D([0], [0], color=MODEL_COLORS.get(m, "#888"), label=m, lw=1.5)
               for m in models]
    fig.legend(handles=handles, loc="lower right", fontsize=8, ncol=len(models))
    fig.suptitle("ROC Curves by Dataset", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(all_curves, output_path):
    import matplotlib.pyplot as plt

    datasets = sorted({c["dataset"] for c in all_curves})
    models = [m for m in MODEL_ORDER if any(c["model"] == m for c in all_curves)]
    n = len(datasets)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()

    for ax_idx, ds in enumerate(datasets):
        ax = axes[ax_idx]
        ds_curves = [c for c in all_curves if c["dataset"] == ds]
        for c in ds_curves:
            ax.plot(c["recall"], c["precision"], label=c["model"],
                    color=MODEL_COLORS.get(c["model"], "#888"), lw=1.2)
        ax.set_title(ds, fontsize=8)
        ax.set_xlabel("Recall", fontsize=7)
        ax.set_ylabel("Precision", fontsize=7)
        ax.tick_params(labelsize=6)

    for ax_idx in range(len(datasets), len(axes)):
        axes[ax_idx].set_visible(False)

    handles = [plt.Line2D([0], [0], color=MODEL_COLORS.get(m, "#888"), label=m, lw=1.5)
               for m in models]
    fig.legend(handles=handles, loc="lower right", fontsize=8, ncol=len(models))
    fig.suptitle("PR Curves by Dataset", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_curves(all_curves, output_path):
    import matplotlib.pyplot as plt

    datasets = sorted({c["dataset"] for c in all_curves})
    models = [m for m in MODEL_ORDER if any(c["model"] == m for c in all_curves)]
    n = len(datasets)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()

    for ax_idx, ds in enumerate(datasets):
        ax = axes[ax_idx]
        ds_curves = [c for c in all_curves if c["dataset"] == ds]
        for c in ds_curves:
            ax.plot(
                c["cal_mean_pred"], c["cal_frac_pos"],
                "o-", label=c["model"],
                color=MODEL_COLORS.get(c["model"], "#888"), lw=1.2, ms=3,
            )
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(ds, fontsize=8)
        ax.set_xlabel("Mean predicted prob.", fontsize=7)
        ax.set_ylabel("Fraction positives", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    for ax_idx in range(len(datasets), len(axes)):
        axes[ax_idx].set_visible(False)

    handles = [plt.Line2D([0], [0], color=MODEL_COLORS.get(m, "#888"), label=m, lw=1.5)
               for m in models]
    handles.append(plt.Line2D([0], [0], color="k", ls="--", label="Perfect"))
    fig.legend(handles=handles, loc="lower right", fontsize=8, ncol=len(models) + 1)
    fig.suptitle("Calibration Curves by Dataset", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LazyClassifier on TDC ADMET binary classification datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=str(_DEFAULT_DATA_DIR),
        help="Path to directory containing .tab files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Output directory for results",
    )
    parser.add_argument(
        "--mode",
        choices=["agnostic", "qsar", "both"],
        default="both",
        help="Which API to benchmark",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        help="Subset of dataset names to run (default: all 18)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"ERROR: data directory not found:\n  {data_dir}")
        print("\nSet --data-dir to the directory containing the .tab files.")
        print(f"Expected location: {_DEFAULT_DATA_DIR}")
        sys.exit(1)

    dataset_names = args.datasets or DATASETS
    all_results = []
    all_curves = []

    for ds_idx, name in enumerate(dataset_names):
        tab_path = data_dir / f"{name}.tab"
        if not tab_path.exists():
            print(f"[SKIP] {name}: not found at {tab_path}")
            continue

        print(f"\n{'─' * 70}")
        print(f"  {name}")
        smiles, y = load_dataset(tab_path)
        pos_rate = y.mean()
        print(f"  n={len(y):,}  pos={pos_rate:.1%}  neg={1-pos_rate:.1%}")
        print(f"{'─' * 70}")

        results, curves = run_dataset(
            name, smiles, y,
            mode=args.mode,
            output_dir=str(output_dir),
            is_first_dataset=(ds_idx == 0),
        )
        all_results.extend(results)
        all_curves.extend(curves)

        for r in results:
            if "error" in r and not isinstance(r.get("error"), float):
                print(f"  {'ERROR':35s}  {r['model']}  →  {r['error'][:80]}")
            elif r["model"] == "LazyClassifier (ONNX)":
                print(
                    f"  {r['model']:35s}"
                    f"  roc={r['roc_auc']:.4f}"
                    f"  pr={r['pr_auc']:.4f}"
                    f"  ece={r['ece']:.4f}"
                    f"  save={r['save_time']:.3f}s"
                    f"  load={r['load_time']:.3f}s"
                    f"  infer={r['infer_time']:.4f}s"
                )
            else:
                print(
                    f"  {r['model']:35s}"
                    f"  roc={r['roc_auc']:.4f}"
                    f"  pr={r['pr_auc']:.4f}"
                    f"  ece={r['ece']:.4f}"
                    f"  fit={r['fit_time']:.3f}s"
                    f"  infer={r['infer_time']:.4f}s"
                )

    if not all_results:
        print("\nNo results collected — check --data-dir.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'═' * 70}")
    print("  ROC-AUC SUMMARY")
    print(f"{'═' * 70}")
    pivot_roc = build_pivot(all_results, "roc_auc")
    print(pivot_roc.round(4).to_string())

    print(f"\n{'═' * 70}")
    print("  PR-AUC SUMMARY")
    print(f"{'═' * 70}")
    pivot_pr = build_pivot(all_results, "pr_auc")
    print(pivot_pr.round(4).to_string())

    print(f"\n{'═' * 70}")
    print("  ECE SUMMARY  (lower = better calibration)")
    print(f"{'═' * 70}")
    pivot_ece = build_pivot(all_results, "ece")
    print(pivot_ece.round(4).to_string())

    print(f"\n{'═' * 70}")
    print("  FIT TIME (seconds)  — LazyClassifier (ONNX) shows save+load times")
    print(f"{'═' * 70}")
    df = pd.DataFrame(all_results)
    timing_rows = []
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        if len(sub) == 0:
            continue
        row = {"model": model,
               "mean_fit_time": sub["fit_time"].mean(),
               "mean_infer_time": sub["infer_time"].mean()}
        if model == "LazyClassifier (ONNX)":
            row["mean_save_time"] = sub["save_time"].mean()
            row["mean_load_time"] = sub["load_time"].mean()
        timing_rows.append(row)
    timing_df = pd.DataFrame(timing_rows).set_index("model")
    print(timing_df.round(4).to_string())

    print(f"\n{'─' * 50}")
    print("  Mean Rank by ROC-AUC (lower = better)")
    print(f"{'─' * 50}")
    print(mean_rank_table(pivot_roc).round(2).to_string())

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults → {csv_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")

        bar_roc_path = output_dir / "benchmark_bars_auroc.png"
        plot_bars(pivot_roc, bar_roc_path, "ROC-AUC by dataset")
        print(f"Bar chart → {bar_roc_path}")

        bar_pr_path = output_dir / "benchmark_bars_aupr.png"
        plot_bars(pivot_pr, bar_pr_path, "PR-AUC by dataset")
        print(f"Bar chart → {bar_pr_path}")

        timing_path = output_dir / "benchmark_timing.png"
        plot_timing_bars(all_results, timing_path)
        print(f"Timing    → {timing_path}")

        radar_path = output_dir / "benchmark_radar.png"
        plot_radar(all_results, radar_path)
        print(f"Radar     → {radar_path}")

        rank_path = output_dir / "benchmark_rank_scatter.png"
        plot_rank_scatter(pivot_roc, rank_path)
        print(f"Rank plot → {rank_path}")

        if all_curves:
            roc_path = output_dir / "roc_curves.png"
            plot_roc_curves(all_curves, roc_path)
            print(f"ROC curves → {roc_path}")

            pr_path = output_dir / "pr_curves.png"
            plot_pr_curves(all_curves, pr_path)
            print(f"PR curves  → {pr_path}")

            cal_path = output_dir / "calibration_curves.png"
            plot_calibration_curves(all_curves, cal_path)
            print(f"Cal curves → {cal_path}")

    except ImportError:
        print("matplotlib not available — skipping plots.")
    except Exception as e:
        print(f"Warning: plot generation failed: {e}")

    print(f"\nDone. All outputs in {output_dir}")


if __name__ == "__main__":
    main()

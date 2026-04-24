"""
Step 03 — Train LazyQSAR models.

Reads data/processed/<pathogen>/01_chembl_datasets.csv to determine which
datasets to train. For each dataset, loads SMILES from the appropriate zip
file in data/raw/<pathogen>/ and trains a LazyClassifierQSAR model saved
under output/models/<pathogen>/<name>/.

  individual/merged datasets → 19_final_datasets.zip  (<name>.csv)
  general datasets           → 20_general_datasets.zip (ORG_<activity>_<unit>_<cutoff>.csv.gz)

A stratified 80/20 split is used to benchmark vanilla RF, LR, and XGB baselines
(Morgan fingerprints) against LazyClassifierQSAR on the same held-out set.
The saved model is the LazyClassifierQSAR trained on the 80% train split.

Usage:
    python scripts/03_train_models.py --pathogen abaumannii --dataset <name>
    python scripts/03_train_models.py --pathogen abaumannii --dataset <name> --mode fast
"""

import argparse
import io
import os
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

import lazyqsar
from lazyqsar.descriptors.morgan import MorganFingerprint
from lazyqsar.qsar import LazyClassifierQSAR
from lazyqsar.utils.metrics import bedroc_score, composite_score

lazyqsar.set_verbosity(True)

PATHOGENS = [
    "abaumannii",
    "calbicans",
    "campylobacter",
    "ecoli",
    "efaecium",
    "enterobacter",
    "hpylori",
    "kpneumoniae",
    "mtuberculosis",
    "ngonorrhoeae",
    "paeruginosa",
    "pfalciparum",
    "saureus",
    "smansoni",
    "spneumoniae",
]


def load_dataset(row: pd.Series, raw_dir: str) -> pd.DataFrame:
    source = row["source"]
    if source in ("individual", "merged"):
        zip_path = os.path.join(raw_dir, "19_final_datasets.zip")
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(f"{row['name']}.csv") as f:
                return pd.read_csv(f)[["smiles", "bin"]]
    else:
        filename = f"ORG_{row['activity_type']}_{row['unit']}_{row['cutoff']}.csv.gz"
        zip_path = os.path.join(raw_dir, "20_general_datasets.zip")
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(filename) as f:
                return pd.read_csv(io.BytesIO(f.read()), compression="gzip")[
                    ["smiles", "bin"]
                ]


DESCRIPTORS_ALL = ["cddd", "chemeleon", "clamp", "morgan", "rdkit"]
HEADS_ALL = ["lr", "xgb", "rf", "svc"]
ONNX_STEM_MAP = {
    "randomforest": "rf",
    "linear": "lr",
    "xgboost": "xgb",
    "svc": "svc",
    "preprocessor": "preprocessor",
}

_PERCENTILES = [5, 25, 50, 75, 95]
_PNAMES = ["p5", "p25", "p50", "p75", "p95"]
_EF_PCTS = [1, 2, 5, 10, 20]


def _onnx_sizes_kb(model_dir: str) -> dict[tuple[str, str], float]:
    """Return {(descriptor, model_type): size_kb} for every ONNX file under model_dir, plus a 'total' key per descriptor."""
    sizes: dict[tuple[str, str], float] = {}
    for descriptor in DESCRIPTORS_ALL:
        desc_dir = os.path.join(model_dir, descriptor)
        if not os.path.isdir(desc_dir):
            continue
        for dirpath, _, filenames in os.walk(desc_dir):
            for fname in filenames:
                stem, ext = os.path.splitext(fname)
                if ext == ".onnx" and stem in ONNX_STEM_MAP:
                    key = (descriptor, ONNX_STEM_MAP[stem])
                    sizes[key] = (
                        sizes.get(key, 0.0)
                        + os.path.getsize(os.path.join(dirpath, fname)) / 1024
                    )
    for descriptor in DESCRIPTORS_ALL:
        parts = [
            sizes[(descriptor, m)]
            for m in ("rf", "lr", "xgb", "svc", "preprocessor")
            if (descriptor, m) in sizes
        ]
        if parts:
            sizes[(descriptor, "total")] = sum(parts)
    return sizes


def _extract_lq_diagnostics(model: LazyClassifierQSAR) -> dict:
    """Extract per-head composite scores, pooler info, and internal timing from a fitted LazyClassifierQSAR."""
    result = {}
    for i, lazy_clf in enumerate(getattr(model, "models", [])):
        desc = model.descriptor_types[i]
        assembler = getattr(lazy_clf, "_model", None)
        if assembler is None:
            continue

        head_scores: dict[str, list] = {}
        fit_times: dict[str, float] = {}
        cal_times: dict[str, float] = {}
        pooler_modes, pooler_scores = [], []

        for batch_clf in getattr(assembler, "models", []):
            portfolio = getattr(batch_clf, "portfolio", [])
            for head, head_name in zip(batch_clf.heads, portfolio):
                hm = head.model
                head_scores.setdefault(head_name, [])
                if hasattr(hm, "oof_probas_") and hasattr(hm, "oof_y_"):
                    try:
                        head_scores[head_name].append(
                            composite_score(hm.oof_y_, hm.oof_probas_)
                        )
                    except Exception:
                        pass
                t = getattr(hm, "timing_", {})
                if head_name in ("xgb", "svc"):
                    fit_times[head_name] = (
                        fit_times.get(head_name, 0.0)
                        + t.get("portfolio_select", 0.0)
                        + t.get("phase2_refit", 0.0)
                    )
                elif head_name == "lr":
                    fit_times[head_name] = fit_times.get(head_name, 0.0) + t.get(
                        "hparam_search", 0.0
                    )
                elif head_name == "rf":
                    fit_times[head_name] = fit_times.get(head_name, 0.0) + t.get(
                        "fit", 0.0
                    )
                cal_times[head_name] = cal_times.get(head_name, 0.0) + t.get(
                    "calibration_total", 0.0
                )

            pooler = getattr(batch_clf, "pooler", None)
            if pooler is not None:
                pooler_modes.append(getattr(pooler, "_mode", "equal"))
                if hasattr(pooler, "_score"):
                    pooler_scores.append(pooler._score)

        for h in head_scores:
            result[f"head_score_{desc}_{h}"] = (
                round(float(np.mean(head_scores[h])), 4) if head_scores[h] else None
            )
        all_heads = set(fit_times) | set(cal_times)
        for h in all_heads:
            result[f"time_{desc}_{h}_fit_s"] = round(fit_times.get(h, 0.0), 2)
            result[f"time_{desc}_{h}_calibration_s"] = round(cal_times.get(h, 0.0), 2)
        result[f"time_{desc}_total_s"] = round(
            sum(fit_times.get(h, 0.0) + cal_times.get(h, 0.0) for h in all_heads), 2
        )
        result[f"pooler_{desc}_mode"] = (
            max(set(pooler_modes), key=pooler_modes.count) if pooler_modes else None
        )
        result[f"pooler_{desc}_score"] = (
            round(float(np.mean(pooler_scores)), 4) if pooler_scores else None
        )

    return result


def _ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(proba[mask].mean() - y_true[mask].mean())
    return ece / len(y_true)


def _calibration_y(y_true: np.ndarray, proba: np.ndarray) -> tuple[str, str]:
    """Return (fraction-of-positives, fraction-of-samples) for 10 fixed-width bins.
    Both are semicolon-separated; empty bins are 'nan'. Last bin is [0.9, 1.0] inclusive."""
    edges = list(np.arange(0, 1, 0.1)) + [1.0]
    n_total = len(y_true)
    pos_parts, frac_parts = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (proba >= lo) & (proba <= hi if i == len(edges) - 2 else proba < hi)
        n = mask.sum()
        pos_parts.append(f"{float(y_true[mask].mean()):.4f}" if n > 0 else "nan")
        frac_parts.append(f"{n / n_total:.4f}" if n > 0 else "nan")
    return ";".join(pos_parts), ";".join(frac_parts)



def _class_dist_stats(y_true: np.ndarray, scores1: np.ndarray, prefix: str) -> dict:
    """Return {prefix}_{active|inactive}_{min/p5/.../max} for a (n,) score array."""
    result = {}
    for label, mask_val in [("active", 1), ("inactive", 0)]:
        s = scores1[y_true == mask_val]
        if len(s) == 0:
            for k in ["min"] + _PNAMES + ["max"]:
                result[f"{prefix}_{label}_{k}"] = np.nan
        else:
            result[f"{prefix}_{label}_min"] = round(float(s.min()), 4)
            for pn, pv in zip(_PNAMES, _PERCENTILES):
                result[f"{prefix}_{label}_{pn}"] = round(float(np.percentile(s, pv)), 4)
            result[f"{prefix}_{label}_max"] = round(float(s.max()), 4)
    return result


def _ef_stats(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Enrichment factor and recall at each screening fraction in _EF_PCTS."""
    n, n_pos = len(y_true), int(y_true.sum())
    order = np.argsort(proba)[::-1]
    cumsum = np.cumsum(y_true[order])
    result = {}
    for pct in _EF_PCTS:
        k = max(1, int(round(n * pct / 100)))
        n_hit = int(cumsum[k - 1])
        expected = k * n_pos / n
        result[f"ef_{pct}pct"] = round(n_hit / max(expected, 1e-9), 4)
        result[f"recall_at_{pct}pct"] = round(n_hit / max(n_pos, 1), 4)
    return result


def _print_summary(
    pathogen: str,
    dataset: str,
    mode: str,
    lazy_auc: float,
    lazy_aupr: float,
    lazy_bedroc: float,
    aupr_baseline: float,
    sensitivity: float,
    specificity: float,
    predict_1k_s: float,
    lq_fit_s: float,
    baselines: list[tuple[str, float]],
    onnx_sizes: dict[tuple[str, str], float],
    diagnostics: dict,
    brier: float,
    brier_baseline: float,
    ece: float,
    n_pos_train: int,
    n_neg_train: int,
    n_pos_test: int,
    n_neg_test: int,
    prob_stats: dict,
    extra_stats: dict,
    ef_stats: dict,
) -> None:
    W = 76
    CW = 10  # column width for numeric grids

    print("\n" + "═" * W)
    print(f" RESULTS — {pathogen} / {dataset}  (mode={mode})")
    print(
        f" train: {n_pos_train + n_neg_train} ({n_pos_train} pos / {n_neg_train} neg)"
        f"   test: {n_pos_test + n_neg_test} ({n_pos_test} pos / {n_neg_test} neg)"
    )
    print("─" * W)

    # Composite
    print(
        f"\n  Composite (LazyClassifierQSAR)   AUC = {lazy_auc:.4f}   AUPR = {lazy_aupr:.4f} (baseline {aupr_baseline:.4f})"
        f"   BEDROC = {lazy_bedroc:.4f}"
        f"   fit = {lq_fit_s:.1f}s   predict/10k = {predict_1k_s:.1f}s"
    )

    # Baselines
    print(f"\n  {'Baseline (Morgan)':<30}  {'AUC':>8}  {'AUPR':>8}  {'BEDROC':>8}")
    print(f"  {'─' * 30}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for name, auc, aupr, bedroc_bl in baselines:
        print(f"  {name:<30}  {auc:>8.4f}  {aupr:>8.4f}  {bedroc_bl:>8.4f}")

    # ONNX sizes
    active = sorted({d for d, _ in onnx_sizes})
    if active:
        onnx_cols = ["rf", "lr", "xgb", "svc", "pre", "total"]
        print(
            f"\n  {'ONNX size (KB)':<16}" + "".join(f"  {m:>{CW}}" for m in onnx_cols)
        )
        print(f"  {'─' * 16}" + ("  " + "─" * CW) * len(onnx_cols))
        for desc in active:
            line = f"  {desc:<16}"
            for m in onnx_cols:
                key = (desc, "preprocessor") if m == "pre" else (desc, m)
                line += f"  {onnx_sizes.get(key, 0.0):>{CW}.1f}"
            print(line)

    # Head composite scores + pooler
    if diagnostics:
        print(
            f"\n  {'Head composite':<16}"
            + "".join(f"  {m:>{CW}}" for m in HEADS_ALL)
            + f"  {'pooler':>{CW}}  {'score':>8}"
        )
        print(f"  {'─' * 16}" + ("  " + "─" * CW) * len(HEADS_ALL) + f"  {'─' * CW}  {'─' * 8}")
        for desc in active:
            line = f"  {desc:<16}"
            for m in HEADS_ALL:
                v = diagnostics.get(f"head_score_{desc}_{m}")
                line += f"  {v:>{CW}.4f}" if v is not None else f"  {'—':>{CW}}"
            line += f"  {diagnostics.get(f'pooler_{desc}_mode', '—'):>{CW}}"
            sc = diagnostics.get(f"pooler_{desc}_score")
            line += f"  {sc:>8.4f}" if sc is not None else f"  {'—':>8}"
            print(line)

    # Internal timing
    timing_rows = [
        (desc, h)
        for desc in active
        for h in HEADS_ALL
        if diagnostics.get(f"time_{desc}_{h}_fit_s") is not None
    ]
    if timing_rows:
        print(
            f"\n  {'Internal timing (s)':<26}  {'fit':>8}  {'calibration':>12}  {'total':>8}"
        )
        print(f"  {'─' * 26}  {'─' * 8}  {'─' * 12}  {'─' * 8}")
        for i, (desc, h) in enumerate(timing_rows):
            fit_t = diagnostics.get(f"time_{desc}_{h}_fit_s", 0.0)
            cal_t = diagnostics.get(f"time_{desc}_{h}_calibration_s", 0.0)
            print(f"  {desc + ' / ' + h:<26}  {fit_t:>8.1f}  {cal_t:>12.1f}")
            is_last_for_desc = (i == len(timing_rows) - 1) or (
                timing_rows[i + 1][0] != desc
            )
            if is_last_for_desc:
                tot = diagnostics.get(f"time_{desc}_total_s")
                if tot is not None:
                    print(f"  {desc + ' (total)':<26}  {'':>8}  {'':>12}  {tot:>8.1f}")

    # Calibration + threshold metrics
    print(
        f"\n  Calibration   Brier = {brier:.4f}  (baseline {brier_baseline:.4f})   ECE = {ece:.4f}"
    )
    print(
        f"  At optimal cutoff   sensitivity = {sensitivity:.4f}   specificity = {specificity:.4f}"
    )

    # Probability distribution by class
    print("\n  Predicted probability distribution (test set)")
    for label, pfx in [("actives  ", "prob_active"), ("inactives", "prob_inactive")]:
        s = prob_stats
        print(
            f"  {label}  min={s[f'{pfx}_min']:.3f}  p5={s[f'{pfx}_p5']:.3f}  p25={s[f'{pfx}_p25']:.3f}"
            f"  p50={s[f'{pfx}_p50']:.3f}  p75={s[f'{pfx}_p75']:.3f}"
            f"  p95={s[f'{pfx}_p95']:.3f}  max={s[f'{pfx}_max']:.3f}"
        )
    print(f"  optimal cutoff = {prob_stats['optimal_cutoff']:.4f}")

    # Score / rank / lift distributions
    extra_rows = [
        ("score", "raw pre-calibration score [0,1]"),
        ("rank", "OOF percentile rank [0,1]"),
        ("lift", "proba / prior [0,∞)"),
    ]
    print(
        f"\n  {'Predicted distribution (test set)':<34}  {'min':>6}  {'p50':>6}  {'max':>6}"
    )
    print(f"  {'─' * 34}  {'─' * 6}  {'─' * 6}  {'─' * 6}")
    for pfx, _ in extra_rows:
        for label in ("active", "inactive"):
            mn = extra_stats.get(f"{pfx}_{label}_min")
            p50 = extra_stats.get(f"{pfx}_{label}_p50")
            mx = extra_stats.get(f"{pfx}_{label}_max")
            row_label = f"{pfx} / {label}"
            mn_s = (
                f"{mn:.4f}"
                if mn is not None and not (isinstance(mn, float) and np.isnan(mn))
                else "—"
            )
            p50_s = (
                f"{p50:.4f}"
                if p50 is not None and not (isinstance(p50, float) and np.isnan(p50))
                else "—"
            )
            mx_s = (
                f"{mx:.4f}"
                if mx is not None and not (isinstance(mx, float) and np.isnan(mx))
                else "—"
            )
            print(f"  {row_label:<34}  {mn_s:>6}  {p50_s:>6}  {mx_s:>6}")

    # Enrichment factors
    print(f"\n  {'Enrichment (test set, prob-ranked)':<34}  {'EF':>6}  {'recall':>8}")
    print(f"  {'─' * 34}  {'─' * 6}  {'─' * 8}")
    for pct in _EF_PCTS:
        ef = ef_stats.get(f"ef_{pct}pct")
        rec = ef_stats.get(f"recall_at_{pct}pct")
        ef_s = f"{ef:.4f}" if ef is not None else "—"
        rec_s = f"{rec:.4f}" if rec is not None else "—"
        print(f"  top {pct:>2}%{'':<28}  {ef_s:>6}  {rec_s:>8}")

    print("═" * W + "\n")


def _save_run_csv(
    output_dir: str,
    pathogen: str,
    dataset: str,
    mode: str,
    n_train: int,
    n_test: int,
    n_pos_train: int,
    n_neg_train: int,
    n_pos_test: int,
    n_neg_test: int,
    lazy_auc: float,
    lazy_aupr: float,
    lazy_bedroc: float,
    aupr_baseline: float,
    sensitivity: float,
    specificity: float,
    predict_1k_s: float,
    lq_fit_s: float,
    baselines: list[tuple[str, float]],
    onnx_sizes: dict[tuple[str, str], float],
    diagnostics: dict,
    brier: float,
    brier_baseline: float,
    ece: float,
    prob_stats: dict,
    calibration_y: str,
    calibration_frac: str,
    extra_stats: dict,
    ef_stats: dict,
    decision_cutoffs: dict = None,
) -> None:
    """
    Write one row to output/results/03_train_models/{pathogen}_{dataset}_{mode}.csv.

    Column reference
    ----------------
    Run identity
      pathogen              Organism the model is trained for.
      dataset               Dataset name (assay identifier).
      mode                  LazyQSAR descriptor mode: fast (morgan+rdkit),
                            default (chemeleon+rdkit+cddd), slow (all four).
      n_train / n_test      Number of compounds in each split.
      n_pos_train / n_neg_train  Class breakdown of the training split.
      n_pos_test  / n_neg_test   Class breakdown of the test split.

    LazyQSAR composite
      auroc_lazy            AUROC of the full LazyQSAR ensemble on the test set.
                            0.5 = random, 1.0 = perfect. Headline metric.
      aupr_lazy             Average precision (area under PR curve) on the test
                            set. More informative than AUROC for imbalanced data.
      bedroc_lazy           BEDROC (alpha=20) on the test set. Exponentially
                            weights early enrichment — actives ranked in the top
                            ~5% of the list contribute ~80% of the score.
                            Range [0, 1]; ~0.5 = random; 1 = perfect.
      aupr_baseline         Random AUPR baseline = positive rate (n_pos/n_total).
                            If aupr_lazy ≈ aupr_baseline the model has no skill.
      sensitivity           True positive rate at the optimal decision cutoff
                            (maximises balanced accuracy on OOF predictions).
      specificity           True negative rate at the same cutoff.

    Baselines (Morgan fingerprints only)
      auroc_rf/lr/xgb       AUROC of standalone RF / LR / XGB trained on Morgan
                            fingerprints. Compare against auroc_lazy to see how
                            much extra value the full pipeline adds.

    On-disk ONNX sizes
      onnx_{desc}_{model}_kb  File size of the saved ONNX model for descriptor
                            `desc` and head `model`. NaN if that descriptor was
                            not used in this mode. Relevant for deployment
                            footprint — larger = heavier inference.

    Per-head scores (LazyQSAR internals)
      head_score_{desc}_{head}  Composite score (same AUROC+AUPR+BEDROC blend as
                            pooler_{desc}_score) of a single head evaluated on
                            OOF predictions. Directly comparable with
                            pooler_{desc}_score — if the pooler score is higher,
                            combining heads helped; if equal, the pooler just
                            picked the best head. NaN if descriptor not used.

    Pooler (how LazyQSAR combines RF/LR/XGB within each descriptor)
      pooler_{desc}_mode    How the three heads were combined: gating = Ridge
                            regression assigns per-compound weights (smart);
                            equal = uniform average; passthrough = single head.
                            NaN/empty if descriptor not used.
      pooler_{desc}_score   Composite score (AUROC+AUPR+BEDROC blend) of the
                            pooled ensemble on OOF predictions. Higher = the
                            pooled model was better than any single head on the
                            training data. NaN if descriptor not used.

    Internal timing (LazyQSAR only)
      time_{desc}_{head}_fit_s        Time to fit the core model for that
                            descriptor+head. For XGB includes portfolio
                            selection and phase-2 refit. NaN if not used.
      time_{desc}_{head}_calibration_s  Time to run k-fold calibration for that
                            descriptor+head. 0 if calibration was skipped (too
                            few samples). NaN if descriptor not used.
      time_lazyqsar_fit_s   Total wall time for the entire LazyQSAR .fit() call.
      predict_time_1k_s     Wall time to score 1,000 compounds. Relevant for
                            deployment planning.

    Calibration quality (test set)
      brier_score           Mean squared error between predicted probability and
                            true label. Lower = better. Confident wrong
                            predictions are heavily penalised.
      brier_baseline        No-skill baseline: pos_rate × (1 − pos_rate).
                            If brier_score ≥ brier_baseline the model is no
                            better than always predicting the dataset's positive
                            rate.
      ece                   Expected Calibration Error. Measures how far
                            predicted probabilities are from actual frequencies
                            (10 equal-width bins). 0 = perfectly calibrated;
                            0.1 means predictions are ~10% off on average.

    Predicted probability distribution (test set, split by class)
      prob_active_{min/p5/p25/p50/p75/p95/max}   Percentiles of predicted
                            probabilities for actives (y=1). A well-separated
                            model has prob_active_p50 >> 0.5.
      prob_inactive_{min/p5/p25/p50/p75/p95/max} Same for inactives (y=0).
                            A well-separated model has prob_inactive_p50 << 0.5.
      optimal_cutoff        Mean decision cutoff across descriptor heads, learned
                            from OOF balanced-accuracy maximisation during
                            LazyQSAR calibration.

    LazyQSAR native prediction outputs (test set, split by class)
      score_{active|inactive}_{min/p5/p25/p50/p75/p95/max}
                            Distribution of predict_score()[:,1] — raw
                            pre-calibration scores before prior correction.
                            Useful for comparing calibrated vs. uncalibrated
                            signal separation.
      rank_{active|inactive}_{min/p5/p25/p50/p75/p95/max}
                            Distribution of predict_rank()[:,1] — percentile
                            rank of each compound within the OOF training score
                            distribution, in [0,1]. 1 = top of training
                            distribution, 0 = bottom. Directly interpretable
                            as a virtual-screening priority score.
      lift_{active|inactive}_{min/p5/p25/p50/p75/p95/max}
                            Distribution of predict_lift()[:,1] —
                            predict_proba()[:,1] / population_prior. Values
                            > 1 indicate enrichment above random baseline;
                            < 1 indicates depletion. NaN if population_prior
                            unavailable on the fitted model.

    Enrichment factors (test set ranked by predicted probability)
      ef_{k}pct             Enrichment factor at top k% of the ranked list
                            (k ∈ {1, 2, 5, 10, 20}).
                            EF = (# actives in top k%) / (k% × N × pos_rate).
                            1 = random; > 1 = enriched; max = 1 / (k/100).
      recall_at_{k}pct      Fraction of all actives captured in the top k% of
                            the ranked list. 0 = none recovered; 1 = all
                            recovered.
    """
    baseline_map = {name: (auc, aupr, bedroc_bl) for name, auc, aupr, bedroc_bl in baselines}

    size_cols = {
        f"onnx_{desc}_{m}_kb": round(onnx_sizes[(desc, m)], 1)
        if (desc, m) in onnx_sizes
        else np.nan
        for desc in DESCRIPTORS_ALL
        for m in ["rf", "lr", "xgb", "svc", "preprocessor", "total"]
    }

    # Fixed head_score + pooler + timing columns for all descriptors × all heads (NaN if absent)
    diag_cols = {}
    for desc in DESCRIPTORS_ALL:
        for h in HEADS_ALL:
            diag_cols[f"head_score_{desc}_{h}"] = diagnostics.get(
                f"head_score_{desc}_{h}", np.nan
            )
            diag_cols[f"time_{desc}_{h}_fit_s"] = diagnostics.get(
                f"time_{desc}_{h}_fit_s", np.nan
            )
            diag_cols[f"time_{desc}_{h}_calibration_s"] = diagnostics.get(
                f"time_{desc}_{h}_calibration_s", np.nan
            )
        diag_cols[f"pooler_{desc}_mode"] = diagnostics.get(f"pooler_{desc}_mode")
        diag_cols[f"pooler_{desc}_score"] = diagnostics.get(
            f"pooler_{desc}_score", np.nan
        )
        diag_cols[f"time_{desc}_total_s"] = diagnostics.get(
            f"time_{desc}_total_s", np.nan
        )

    row = {
        "pathogen": pathogen,
        "dataset": dataset,
        "mode": mode,
        "n_train": n_train,
        "n_test": n_test,
        "n_pos_train": n_pos_train,
        "n_neg_train": n_neg_train,
        "n_pos_test": n_pos_test,
        "n_neg_test": n_neg_test,
        "auroc_lazy": round(lazy_auc, 4),
        "aupr_lazy": round(lazy_aupr, 4),
        "bedroc_lazy": round(lazy_bedroc, 4),
        "aupr_baseline": round(aupr_baseline, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "auroc_rf": round(baseline_map["RF (n=100)"][0], 4) if "RF (n=100)" in baseline_map else np.nan,
        "aupr_rf": round(baseline_map["RF (n=100)"][1], 4) if "RF (n=100)" in baseline_map else np.nan,
        "bedroc_rf": round(baseline_map["RF (n=100)"][2], 4) if "RF (n=100)" in baseline_map else np.nan,
        "auroc_lr": round(baseline_map["LR (L1)"][0], 4) if "LR (L1)" in baseline_map else np.nan,
        "aupr_lr": round(baseline_map["LR (L1)"][1], 4) if "LR (L1)" in baseline_map else np.nan,
        "bedroc_lr": round(baseline_map["LR (L1)"][2], 4) if "LR (L1)" in baseline_map else np.nan,
        "auroc_xgb": round(baseline_map["XGB (n=100)"][0], 4) if "XGB (n=100)" in baseline_map else np.nan,
        "aupr_xgb": round(baseline_map["XGB (n=100)"][1], 4) if "XGB (n=100)" in baseline_map else np.nan,
        "bedroc_xgb": round(baseline_map["XGB (n=100)"][2], 4) if "XGB (n=100)" in baseline_map else np.nan,
        "auroc_svc": round(baseline_map["SVC (linear)"][0], 4) if "SVC (linear)" in baseline_map else np.nan,
        "aupr_svc": round(baseline_map["SVC (linear)"][1], 4) if "SVC (linear)" in baseline_map else np.nan,
        "bedroc_svc": round(baseline_map["SVC (linear)"][2], 4) if "SVC (linear)" in baseline_map else np.nan,
        **size_cols,
        **diag_cols,
        "time_lazyqsar_fit_s": round(lq_fit_s, 2),
        "predict_time_1k_s": round(predict_1k_s, 2),
        "brier_score": round(brier, 4),
        "brier_baseline": round(brier_baseline, 4),
        "ece": round(ece, 4),
        **{k: round(v, 4) for k, v in prob_stats.items()},
        "calibration_y": calibration_y,
        "calibration_frac": calibration_frac,
        **{
            k: (round(v, 4) if isinstance(v, float) and not np.isnan(v) else v)
            for k, v in extra_stats.items()
        },
        **ef_stats,
        **(decision_cutoffs or {}),
    }

    out_path = os.path.join(output_dir, f"{pathogen}_{dataset}_{mode}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"Run logged to {out_path}")


def train_dataset(
    pathogen: str,
    dataset: str,
    mode: str,
    repo_dir: str,
    output_dir: str,
    models_output_dir: str,
) -> None:
    metadata_path = os.path.join(
        repo_dir, "data", "processed", pathogen, "01_chembl_datasets.csv"
    )
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata found for {pathogen}")

    metadata = pd.read_csv(metadata_path)
    duplicates = metadata[metadata["name"].duplicated()]
    if not duplicates.empty:
        raise ValueError(
            f"Duplicate dataset names found: {duplicates['name'].tolist()}"
        )

    matches = metadata[metadata["name"] == dataset]
    if matches.empty:
        raise ValueError(f"Dataset '{dataset}' not found in {pathogen} metadata.")

    row = matches.iloc[0]
    raw_dir = os.path.join(repo_dir, "data", "raw", pathogen)

    df = load_dataset(row, raw_dir)
    smiles_list = df["smiles"].tolist()
    y = np.array(df["bin"].tolist(), dtype=int)

    print(f"Training {pathogen}/{dataset} ({len(smiles_list)} compounds, mode={mode})")

    smiles_train, smiles_test, y_train, y_test = train_test_split(
        smiles_list, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- Morgan fingerprints for baselines ---
    morgan = MorganFingerprint()
    X_train = morgan.transform(smiles_train)
    X_test = morgan.transform(smiles_test)

    baselines = []

    # --- RF baseline ---
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    _rf_p = rf.predict_proba(X_test)[:, 1]
    baselines.append((
        "RF (n=100)",
        roc_auc_score(y_test, _rf_p),
        average_precision_score(y_test, _rf_p),
        bedroc_score(y_test, _rf_p),
    ))

    # --- LR baseline (skipped for large datasets — too slow) ---
    if len(smiles_train) <= 1_000:
        scaler = MaxAbsScaler()
        lr = LogisticRegression(
            C=0.1,
            solver="saga",
            penalty="l1",
            class_weight="balanced",
            max_iter=1_000,
            random_state=42,
        )
        lr.fit(scaler.fit_transform(X_train), y_train)
        _lr_p = lr.predict_proba(scaler.transform(X_test))[:, 1]
        baselines.append((
            "LR (L1)",
            roc_auc_score(y_test, _lr_p),
            average_precision_score(y_test, _lr_p),
            bedroc_score(y_test, _lr_p),
        ))

    # --- XGB baseline (fast: histogram-based splits) ---
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=100,
        tree_method="hist",
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    _xgb_p = xgb.predict_proba(X_test)[:, 1]
    baselines.append((
        "XGB (n=100)",
        roc_auc_score(y_test, _xgb_p),
        average_precision_score(y_test, _xgb_p),
        bedroc_score(y_test, _xgb_p),
    ))

    # --- SVC baseline (LinearSVC on Morgan; decision_function used for AUROC) ---
    if len(smiles_train) <= 5_000:
        scaler_svc = MaxAbsScaler()
        svc = LinearSVC(C=1.0, class_weight="balanced", max_iter=2_000, random_state=42)
        svc.fit(scaler_svc.fit_transform(X_train), y_train)
        _svc_s = svc.decision_function(scaler_svc.transform(X_test))
        baselines.append((
            "SVC (linear)",
            roc_auc_score(y_test, _svc_s),
            average_precision_score(y_test, _svc_s),
            bedroc_score(y_test, _svc_s),
        ))

    # --- LazyClassifierQSAR ---
    t0 = time.perf_counter()
    model = LazyClassifierQSAR(mode=mode)
    model.fit(smiles_train, y_train)
    lq_fit_s = time.perf_counter() - t0

    lazy_proba = model.predict_proba(smiles_test)
    lazy_auc = roc_auc_score(y_test, lazy_proba[:, 1])

    # Benchmark predict on a fixed 1k-compound batch
    rng = np.random.default_rng(42)
    smiles_1k = [
        smiles_train[i] for i in rng.integers(0, len(smiles_train), size=1_000)
    ]
    t0 = time.perf_counter()
    model.predict_proba(smiles_1k)
    predict_1k_s = time.perf_counter() - t0

    # Save model and measure ONNX sizes
    model_dir = os.path.join(models_output_dir, pathogen, dataset, mode)
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)
    print(f"Saved to {model_dir}")
    onnx_sizes = _onnx_sizes_kb(model_dir)

    # Internal diagnostics
    diagnostics = _extract_lq_diagnostics(model)

    # Calibration
    p1 = lazy_proba[:, 1]
    brier = brier_score_loss(y_test, p1)
    pos_rate = y_test.mean()
    brier_baseline = pos_rate * (1 - pos_rate)
    ece = _ece(y_test, p1)
    calibration_y, calibration_frac = _calibration_y(y_test, p1)

    # AUPR + BEDROC
    lazy_aupr = average_precision_score(y_test, p1)
    lazy_bedroc = bedroc_score(y_test, p1)
    aupr_baseline = float(pos_rate)

    # Probability distribution by class + model decision cutoff
    p1_active = p1[y_test == 1]
    p1_inactive = p1[y_test == 0]
    def _mean_cutoff(attr):
        vals = [
            getattr(m._model, attr)
            for m in model.models
            if hasattr(getattr(m, "_model", None), attr)
        ]
        return float(np.mean(vals)) if vals else np.nan

    decision_cutoffs = {
        "decision_cutoff_raw": _mean_cutoff("decision_cutoff_raw_"),
        "decision_cutoff_proba": _mean_cutoff("decision_cutoff_proba_"),
        "decision_cutoff_rank": _mean_cutoff("decision_cutoff_rank_"),
        "decision_cutoff_logit": _mean_cutoff("decision_cutoff_logit_"),
        "decision_cutoff_lift": _mean_cutoff("decision_cutoff_lift_"),
    }
    optimal_cutoff = decision_cutoffs["decision_cutoff_raw"]

    # Sensitivity / specificity at optimal cutoff
    n_pos_train, n_neg_train = int((y_train == 1).sum()), int((y_train == 0).sum())
    n_pos_test, n_neg_test = int((y_test == 1).sum()), int((y_test == 0).sum())
    if not np.isnan(optimal_cutoff):
        y_pred = (p1 >= optimal_cutoff).astype(int)
        sensitivity = float(((y_pred == 1) & (y_test == 1)).sum() / max(n_pos_test, 1))
        specificity = float(((y_pred == 0) & (y_test == 0)).sum() / max(n_neg_test, 1))
    else:
        sensitivity = specificity = np.nan

    prob_stats = {
        "prob_active_min": float(p1_active.min()),
        "prob_active_p5": float(np.percentile(p1_active, 5)),
        "prob_active_p25": float(np.percentile(p1_active, 25)),
        "prob_active_p50": float(np.percentile(p1_active, 50)),
        "prob_active_p75": float(np.percentile(p1_active, 75)),
        "prob_active_p95": float(np.percentile(p1_active, 95)),
        "prob_active_max": float(p1_active.max()),
        "prob_inactive_min": float(p1_inactive.min()),
        "prob_inactive_p5": float(np.percentile(p1_inactive, 5)),
        "prob_inactive_p25": float(np.percentile(p1_inactive, 25)),
        "prob_inactive_p50": float(np.percentile(p1_inactive, 50)),
        "prob_inactive_p75": float(np.percentile(p1_inactive, 75)),
        "prob_inactive_p95": float(np.percentile(p1_inactive, 95)),
        "prob_inactive_max": float(p1_inactive.max()),
        "optimal_cutoff": optimal_cutoff,
    }

    # Native LazyQSAR prediction outputs — class-split distributions
    extra_stats: dict = {}
    for method, prefix in [
        ("predict_score", "score"),
        ("predict_rank", "rank"),
        ("predict_lift", "lift"),
        ("predict_logit", "logit"),
    ]:
        out = getattr(model, method)(smiles_test)
        extra_stats.update(_class_dist_stats(y_test, out[:, 1], prefix))

    # Enrichment factors
    ef_stats = _ef_stats(y_test, p1)

    _print_summary(
        pathogen,
        dataset,
        mode,
        lazy_auc,
        lazy_aupr,
        lazy_bedroc,
        aupr_baseline,
        sensitivity,
        specificity,
        predict_1k_s,
        lq_fit_s,
        baselines,
        onnx_sizes,
        diagnostics,
        brier,
        brier_baseline,
        ece,
        n_pos_train,
        n_neg_train,
        n_pos_test,
        n_neg_test,
        prob_stats,
        extra_stats,
        ef_stats,
    )
    _save_run_csv(
        output_dir,
        pathogen,
        dataset,
        mode,
        len(smiles_train),
        len(smiles_test),
        n_pos_train,
        n_neg_train,
        n_pos_test,
        n_neg_test,
        lazy_auc,
        lazy_aupr,
        lazy_bedroc,
        aupr_baseline,
        sensitivity,
        specificity,
        predict_1k_s,
        lq_fit_s,
        baselines,
        onnx_sizes,
        diagnostics,
        brier,
        brier_baseline,
        ece,
        prob_stats,
        calibration_y,
        calibration_frac,
        extra_stats,
        ef_stats,
        decision_cutoffs,
    )


def _run_csv_path(output_dir: str, pathogen: str, dataset: str, mode: str) -> str:
    return os.path.join(output_dir, f"{pathogen}_{dataset}_{mode}.csv")


def main(args: argparse.Namespace) -> None:
    if args.csv:
        _run_batch(
            args.csv,
            args.modes,
            args.repo_dir,
            args.output_dir,
            args.models_output_dir,
            skip_existing=args.skip_existing,
        )
    else:
        train_dataset(
            args.pathogen,
            args.dataset,
            args.mode,
            args.repo_dir,
            args.output_dir,
            args.models_output_dir,
        )


def _run_batch(
    csv_path: str,
    modes: list[str],
    repo_dir: str,
    output_dir: str,
    models_output_dir: str,
    skip_existing: bool = False,
) -> None:
    df = pd.read_csv(csv_path)
    jobs = [
        (row["pathogen"], row["name"], mode)
        for _, row in df.iterrows()
        for mode in modes
    ]
    total = len(jobs)

    done, skipped, failed = 0, 0, []

    for i, (pathogen, dataset, mode) in enumerate(jobs, 1):
        out_path = _run_csv_path(output_dir, pathogen, dataset, mode)
        tag = f"[{i}/{total}] {pathogen}/{dataset} ({mode})"

        if skip_existing and os.path.exists(out_path):
            print(f"  SKIP  {tag}")
            skipped += 1
            continue

        print(f"\n  RUN   {tag} " + "─" * max(1, 76 - len(tag) - 8))
        try:
            train_dataset(pathogen, dataset, mode, repo_dir, output_dir, models_output_dir)
            done += 1
        except Exception as exc:
            print(f"  FAIL  {tag} — {exc}")
            failed.append((pathogen, dataset, mode, str(exc)))

    print("\n" + "═" * 60)
    print(
        f" Batch complete: {done} done, {skipped} skipped, {len(failed)} failed  (total {total})"
    )
    if failed:
        print("\n Failed runs:")
        for pathogen, dataset, mode, err in failed:
            print(f"   {pathogen}/{dataset} ({mode}): {err}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LazyQSAR binary classifiers for antimicrobial datasets."
    )

    # --- paths ---
    parser.add_argument(
        "--repo_dir",
        required=True,
        metavar="PATH",
        help=(
            "Root of the data repository. Expected layout: "
            "<repo_dir>/data/processed/<pathogen>/01_chembl_datasets.csv (dataset registry) and "
            "<repo_dir>/data/raw/<pathogen>/19_final_datasets.zip or 20_general_datasets.zip (compound SMILES)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        metavar="PATH",
        help=(
            "Root output directory. Results CSVs are written to <output_dir>/<pathogen>_<dataset>_<mode>.csv."
        ),
    )
    parser.add_argument(
        "--models_output_dir",
        required=True,
        metavar="PATH",
        help="Directory where trained models are saved (<models_output_dir>/<pathogen>/<dataset>/<mode>/).",
    )

    # --- single-run args ---
    single = parser.add_argument_group("single run (--pathogen + --dataset + --mode)")
    single.add_argument(
        "--pathogen",
        type=str,
        choices=PATHOGENS,
        help="Pathogen code to train models for.",
    )
    single.add_argument(
        "--dataset",
        type=str,
        help="Dataset name to train (e.g. CHEMBL4296188_INHIBITION_%%_qt_25.0).",
    )
    single.add_argument(
        "--mode",
        type=str,
        default="fast",
        choices=["fast", "slow"],
        help="LazyQSAR descriptor mode (default: fast).",
    )

    # --- batch-run args ---
    batch = parser.add_argument_group("batch run (--csv)")
    batch.add_argument(
        "--csv",
        type=str,
        metavar="PATH",
        help=(
            "Path to a datasets CSV (same format as data/processed/01_chembl_datasets_all.csv). "
            "Trains every row across all --modes."
        ),
    )
    batch.add_argument(
        "--modes",
        nargs="+",
        choices=["fast", "slow"],
        default=["fast", "slow"],
        metavar="MODE",
        help="Modes to run in batch (default: fast slow).",
    )
    batch.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose output CSV already exists.",
    )

    args = parser.parse_args()

    if args.csv is None and (args.pathogen is None or args.dataset is None):
        parser.error(
            "Provide either --csv (batch) or both --pathogen and --dataset (single run)."
        )

    main(args)

"""LazyQSAR on the official TDC ADMET regression benchmarks, scored as the leaderboard is.

    python dev/benchmarks/regression/tdc_official.py --mode fast --label tdc-fast
    python dev/benchmarks/regression/tdc_official.py --mode slow --label tdc-slow

Uses TDC's own ``admet_group`` files (``train_val.csv`` / ``test.csv``, a fixed scaffold
split), downloaded raw from Dataverse into the gitignored cache; nothing from the ``tdc``
package is imported. The leaderboard's metric is MAE for Caco2, lipophilicity,
solubility, PPBR and LD50, and Spearman for VDss, half-life and both clearances. The four
Spearman tasks are fitted on log10(y) (standard practice for these skewed quantities, and
what the top entries do); Spearman is unchanged by the transform. The five MAE tasks are
fitted and scored in the units TDC publishes.

The leaderboard averages five seeds that only reshuffle train/validation; LazyQSAR does its
own internal validation on ``train_val``, so every seed sees identical input. ``--repeats``
refits to measure LazyQSAR's own run-to-run variation instead.

One JSON line per (task, repeat) goes to ``.cache/results/<label>.jsonl``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
RAW = os.path.join(CACHE, "raw", "admet_group")
FEATURES = os.path.join(CACHE, "features", "tdc_official")
RESULTS = os.path.join(CACHE, "results")
ADMET_GROUP_ID = 4426004  # TDC's benchmark2id["admet_group"]
USER_AGENT = "lazyqsar-regression-benchmark/1.0 (+https://github.com/ersilia-os/lazy-qsar)"

# task: (metric, fit on log10?)
TASKS = {
    "caco2_wang": ("mae", False),
    "lipophilicity_astrazeneca": ("mae", False),
    "solubility_aqsoldb": ("mae", False),
    "ppbr_az": ("mae", False),
    "ld50_zhu": ("mae", False),
    "vdss_lombardo": ("spearman", True),
    "half_life_obach": ("spearman", True),
    "clearance_hepatocyte_az": ("spearman", True),
    "clearance_microsome_az": ("spearman", True),
}


def fetch():
    if os.path.isdir(RAW):
        return
    import urllib.request

    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    url = f"https://dataverse.harvard.edu/api/access/datafile/{ADMET_GROUP_ID}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    data = urllib.request.urlopen(req, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = [m for m in zf.namelist() if m.endswith(".csv")]
        zf.extractall(os.path.dirname(RAW), members=members)


def _read(task, part):
    with open(os.path.join(RAW, task, f"{part}.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["Drug"] for r in rows], np.asarray([float(r["Y"]) for r in rows])


def _features(task, part, smiles, descriptor):
    os.makedirs(FEATURES, exist_ok=True)
    path = os.path.join(FEATURES, f"{task}__{part}__{descriptor}.npy")
    if os.path.isfile(path):
        return np.load(path)
    from lazyqsar.registry import get_descriptor_type

    X = np.asarray(get_descriptor_type(descriptor)().transform(smiles), dtype=np.float32)
    np.save(path, X)
    return X


def _qsar_md5(smiles):
    from lazyqsar.qsar import _smiles_md5

    return _smiles_md5(smiles)


def run(label, mode, task_names, repeats):
    from scipy.stats import spearmanr

    from lazyqsar.qsar_regressor import LazyRegressorQSAR
    from lazyqsar.registry import DESCRIPTORS_MODE

    fetch()
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, f"{label}.jsonl")
    done = set()
    if os.path.isfile(out_path):
        with open(out_path) as f:
            done = {(r["task"], r["repeat"]) for r in map(json.loads, f)}
    for task in task_names:
        metric, log10 = TASKS[task]
        smi_tr, y_tr = _read(task, "train_val")
        smi_te, y_te = _read(task, "test")
        pre_tr = {d: _features(task, "train_val", smi_tr, d) for d in DESCRIPTORS_MODE[mode]}
        pre_te = {d: _features(task, "test", smi_te, d) for d in DESCRIPTORS_MODE[mode]}
        target = np.log10(y_tr) if log10 else y_tr
        for rep in range(repeats):
            if (task, rep) in done:
                continue
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                t0 = time.perf_counter()
                model = LazyRegressorQSAR(mode=mode)
                model.fit(smi_tr, target, precomputed=pre_tr)
                fit_seconds = time.perf_counter() - t0
                # predict() takes no precomputed features; seed its per-descriptor cache.
                key = _qsar_md5(smi_te)
                for i, d in enumerate(model.descriptor_types):
                    model._feature_cache[(i, key)] = pre_te[d]
                pred = np.asarray(model.predict(smi_te), dtype=float)
            if log10:
                pred = 10**pred
            # TDC scores every test row; a SMILES LazyQSAR cannot featurize gets the
            # training mean rather than being dropped from the score.
            n_nan = int(np.isnan(pred).sum())
            pred = np.where(np.isnan(pred), float(np.mean(y_tr)), pred)
            score = (
                float(np.mean(np.abs(pred - y_te)))
                if metric == "mae"
                else float(spearmanr(pred, y_te).statistic)
            )
            record = {
                "label": label,
                "mode": mode,
                "task": task,
                "repeat": rep,
                "metric": metric,
                "score": score,
                "fit_seconds": fit_seconds,
                "n_train": len(y_tr),
                "n_test": len(y_te),
                "n_nan_predictions": n_nan,
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"{task:28s} rep={rep} {metric}={score:.3f} t={fit_seconds:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", choices=["fast", "slow"], default="fast")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()
    run(args.label, args.mode, args.tasks, args.repeats)


if __name__ == "__main__":
    main()

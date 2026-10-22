"""LazyQSAR against MoleculeACE's best published method, on MoleculeACE's own splits.

    python dev/benchmarks/regression/ace_reference.py --label ace-ref --what reference
    python dev/benchmarks/regression/ace_reference.py --label ace-fast --what lazyqsar --mode fast

van Tilborg et al. (2022) found SVR on ECFP4 the best of 24 methods on the 30 ChEMBL
datasets, ahead of every graph and sequence network. The repository publishes no results
table, but it does publish each method's tuned hyperparameters per dataset
(``Data/configures/benchmark/<dataset>/<METHOD>.yml``), so the reference is reproduced
here: sklearn's ``SVR``/``RandomForestRegressor``/``GradientBoostingRegressor`` with those
hyperparameters on 1024-bit radius-2 Morgan bits, exactly as ``MoleculeACE.models.ml`` and
``featurization.ecfp`` build them.

Scores are MoleculeACE's: RMSE on the test split, and RMSE restricted to the test
compounds flagged as activity cliffs (``cliff_mol == 1``). The configs are downloaded once
into the gitignored cache.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".cache")
RAW = os.path.join(CACHE, "raw")
CONFIGS = os.path.join(RAW, "moleculeace_configs")
FEATURES = os.path.join(CACHE, "features", "ace_reference")
RESULTS = os.path.join(CACHE, "results")
CONFIG_URL = (
    "https://raw.githubusercontent.com/molML/MoleculeACE/main/MoleculeACE/Data/"
    "configures/benchmark/{dataset}/{method}.yml"
)
USER_AGENT = (
    "lazyqsar-regression-benchmark/1.0 (+https://github.com/ersilia-os/lazy-qsar)"
)
REFERENCE_METHODS = ("SVM_ECFP", "RF_ECFP", "GBM_ECFP")


def datasets():
    return sorted(
        f[len("moleculeace_") : -len(".csv")]
        for f in os.listdir(RAW)
        if f.startswith("moleculeace_") and f.endswith(".csv")
    )


def _read(dataset):
    rows = list(csv.DictReader(open(os.path.join(RAW, f"moleculeace_{dataset}.csv"))))
    out = {}
    for part in ("train", "test"):
        sel = [r for r in rows if r["split"] == part]
        out[part] = (
            [r["smiles"] for r in sel],
            np.asarray([float(r["y [pEC50/pKi]"]) for r in sel]),
            np.asarray([int(r["cliff_mol"]) for r in sel]),
        )
    return out


def _config(dataset, method):
    path = os.path.join(CONFIGS, dataset, f"{method}.yml")
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = CONFIG_URL.format(dataset=dataset, method=method)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with open(path, "wb") as f:
            f.write(urllib.request.urlopen(req, timeout=60).read())
    import yaml

    with open(path) as f:
        params = yaml.safe_load(f) or {}
    # Configs predate scikit-learn 1.3, where max_features="auto" meant all features.
    if params.get("max_features") == "auto":
        params["max_features"] = 1.0
    return params


def _ecfp(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import DataStructs
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

    RDLogger.DisableLog("rdApp.*")  # the legacy call MoleculeACE uses is deprecated
    X = np.zeros((len(smiles), 1024), dtype=np.float32)
    for i, s in enumerate(smiles):
        fp = GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024)
        DataStructs.ConvertToNumpyArray(fp, X[i])
    return X


def _features(dataset, part, smiles, descriptor):
    os.makedirs(FEATURES, exist_ok=True)
    path = os.path.join(FEATURES, f"{dataset}__{part}__{descriptor}.npy")
    if os.path.isfile(path):
        return np.load(path)
    if descriptor == "ecfp1024":
        X = _ecfp(smiles)
    else:
        from lazyqsar.registry import get_descriptor_type

        X = np.asarray(
            get_descriptor_type(descriptor)().transform(smiles), dtype=np.float32
        )
    np.save(path, X)
    return X


def _reference_model(method, params):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.svm import SVR

    cls = {"SVM": SVR, "RF": RandomForestRegressor, "GBM": GradientBoostingRegressor}
    return cls[method.split("_")[0]](**params)


def _scores(y, pred, cliff):
    err = (pred - y) ** 2
    return {
        "rmse": float(np.sqrt(err.mean())),
        "rmse_cliff": float(np.sqrt(err[cliff == 1].mean())) if cliff.any() else None,
    }


def run(label, what, mode, names):
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, f"{label}.jsonl")
    done = set()
    if os.path.isfile(out_path):
        with open(out_path) as f:
            done = {(r["dataset"], r["method"]) for r in map(json.loads, f)}
    for dataset in names:
        d = _read(dataset)
        (s_tr, y_tr, _), (s_te, y_te, c_te) = d["train"], d["test"]
        if what == "reference":
            jobs = [(m, None) for m in REFERENCE_METHODS]
        else:
            jobs = [(f"lazyqsar_{mode}", mode)]
        for method, lq_mode in jobs:
            if (dataset, method) in done:
                continue
            t0 = time.perf_counter()
            if lq_mode is None:
                X_tr = _features(dataset, "train", s_tr, "ecfp1024")
                X_te = _features(dataset, "test", s_te, "ecfp1024")
                model = _reference_model(method, _config(dataset, method))
                pred = model.fit(X_tr, y_tr).predict(X_te)
            else:
                from lazyqsar.qsar import _smiles_md5
                from lazyqsar.qsar_regressor import LazyRegressorQSAR

                sink = io.StringIO()
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    model = LazyRegressorQSAR(mode=lq_mode)
                    pre_tr = {
                        n: _features(dataset, "train", s_tr, n)
                        for n in model.descriptor_types
                    }
                    pre_te = {
                        n: _features(dataset, "test", s_te, n)
                        for n in model.descriptor_types
                    }
                    t0 = time.perf_counter()
                    model.fit(s_tr, y_tr, precomputed=pre_tr)
                    key = _smiles_md5(s_te)
                    for i, n in enumerate(model.descriptor_types):
                        model._feature_cache[(i, key)] = pre_te[n]
                    pred = np.asarray(model.predict(s_te), dtype=float)
            record = {
                "label": label,
                "dataset": dataset,
                "method": method,
                "fit_seconds": time.perf_counter() - t0,
                "n_train": len(y_tr),
                "n_test": len(y_te),
                "n_test_cliff": int(c_te.sum()),
                **_scores(y_te, pred, c_te),
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(
                f"{dataset:18s} {method:16s} rmse={record['rmse']:.3f} "
                f"cliff={record['rmse_cliff']:.3f} t={record['fit_seconds']:.0f}s",
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--what", choices=["reference", "lazyqsar"], required=True)
    ap.add_argument("--mode", choices=["fast", "slow"], default="fast")
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    run(args.label, args.what, args.mode, args.datasets or datasets())


if __name__ == "__main__":
    main()

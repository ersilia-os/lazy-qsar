"""Pooling study: which way of combining the heads predicts best, scored offline.

    python dev/benchmarks/regression/poolers.py collect --label pool-v1
    python dev/benchmarks/regression/poolers.py score --label pool-v1

``collect`` fits one ``LazyRegressor`` per (task, split) -- ``random-0`` and the
structural split -- and keeps, per head, the out-of-fold predictions on the training
rows and the predictions on the test rows (standardised units), plus the gate's own
test prediction. ``score`` then compares pooling rules on the same numbers, so every
rule sees identical heads and no rule pays for a refit:

- ``gate``    -- the current Gaussian-likelihood Ridge gate (what LazyQSAR ships);
- ``equal``   -- the plain mean of the heads;
- ``nnls``    -- non-negative least squares on the OOF predictions (Breiman 1996,
  "Stacked regressions"), weights normalised to sum to one;
- ``best``    -- the single head with the lowest OOF RMSE;
- ``top2``    -- the mean of the two heads with the lowest OOF RMSE.

Every rule is fitted on out-of-fold predictions only and scored on the test split.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tasks as task_registry  # noqa: E402

CACHE = os.path.join(HERE, ".cache")
RESULTS = os.path.join(CACHE, "results")


def collect(label, task_names):
    from lazyqsar.agnostic import LazyRegressor

    out_dir = os.path.join(CACHE, "pooling", label)
    os.makedirs(out_dir, exist_ok=True)
    for task in task_names:
        smiles, y, X, provided = task_registry.load(task)
        for split, tr, te in task_registry.splits(task, smiles, provided, n_random=1):
            path = os.path.join(out_dir, f"{task}__{split}.npz")
            if os.path.isfile(path):
                continue
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                model = LazyRegressor()
                model.fit(X=X[tr], y=y[tr])
                inner = model._model
                batch = inner.models[0]
                X_te = batch.prep.transform(X[te])
                oof = np.column_stack([h.model.oof_preds_ for h in batch.heads])
                test = np.column_stack([h.predict(X_te) for h in batch.heads])
                gate = batch.pooler.predict(test, X_te)
                rf = [h.model for h in batch.heads if h.name == "rf"]
                oob = rf[0].oob_preds_ if rf else np.empty(0)
            np.savez_compressed(
                path,
                heads=np.asarray(batch.portfolio),
                oof=oof,
                test=test,
                gate=gate,
                y_train=(y[tr] - inner.y_mean_) / inner.y_std_,
                y_test=(y[te] - inner.y_mean_) / inner.y_std_,
                single_batch=np.asarray(len(inner.models) == 1),
                rf_oob=oob,
            )
            print(f"{task:28s} {split:9s} heads={list(batch.portfolio)}", flush=True)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _nnls_weights(P, y):
    from scipy.optimize import nnls

    w, _ = nnls(P, y)
    if w.sum() <= 0:
        return np.full(P.shape[1], 1.0 / P.shape[1])
    return w / w.sum()


def pool(rule, oof, test, y_train, gate):
    if rule == "gate":
        return gate
    if rule == "equal":
        return test.mean(axis=1)
    errors = np.array([_rmse(oof[:, j], y_train) for j in range(oof.shape[1])])
    if rule == "best":
        return test[:, int(np.argmin(errors))]
    if rule == "top2":
        keep = np.argsort(errors)[:2]
        return test[:, keep].mean(axis=1)
    if rule == "nnls":
        return test @ _nnls_weights(oof, y_train)
    raise KeyError(rule)


RULES = ("gate", "equal", "nnls", "best", "top2")


def score(label):
    from scipy.stats import spearmanr

    in_dir = os.path.join(CACHE, "pooling", label)
    rows = []
    for fn in sorted(os.listdir(in_dir)):
        d = np.load(os.path.join(in_dir, fn))
        task, split = fn[: -len(".npz")].split("__")
        row = {"task": task, "split": split, "heads": d["heads"].tolist()}
        for rule in RULES:
            pred = pool(rule, d["oof"], d["test"], d["y_train"], d["gate"])
            row[rule] = {
                "rmse_z": _rmse(pred, d["y_test"]),
                "rho": float(spearmanr(pred, d["y_test"]).statistic),
            }
        rows.append(row)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"pooling-{label}.json"), "w") as f:
        json.dump(rows, f, indent=1)

    print(f"{len(rows)} (task, split) pairs")
    rel = {r: [] for r in RULES}
    for row in rows:
        ref = row["gate"]["rmse_z"]
        for rule in RULES:
            rel[rule].append(row[rule]["rmse_z"] / ref - 1.0)
    for rule in RULES:
        v = np.array(rel[rule])
        wins = int((v < -1e-9).sum())
        print(
            f"{rule:6s} mean RMSE vs gate {100 * v.mean():+.2f}%  "
            f"median {100 * np.median(v):+.2f}%  wins {wins}/{len(v)}  "
            f"mean rho {np.mean([row[rule]['rho'] for row in rows]):.4f}"
        )


def _with_oob(d):
    """The OOF matrix with the RF column replaced by the forest's out-of-bag output."""
    heads = d["heads"].tolist()
    oof = d["oof"].copy()
    oob = d["rf_oob"]
    if "rf" not in heads or oob.size != oof.shape[0] or not np.isfinite(oob).all():
        return None
    oof[:, heads.index("rf")] = oob
    return oof


def ablations(label, rule="nnls"):
    """RF OOF from folds vs OOB, and which head subsets earn their place."""
    from itertools import combinations

    in_dir = os.path.join(CACHE, "pooling", label)
    files = sorted(os.listdir(in_dir))
    oob_rel, subset_rel = [], {}
    for fn in files:
        d = np.load(os.path.join(in_dir, fn))
        heads = d["heads"].tolist()
        ref = _rmse(
            pool(rule, d["oof"], d["test"], d["y_train"], d["gate"]), d["y_test"]
        )
        oof_oob = _with_oob(d)
        if oof_oob is not None:
            alt = _rmse(
                pool(rule, oof_oob, d["test"], d["y_train"], d["gate"]), d["y_test"]
            )
            oob_rel.append(alt / ref - 1)
        for k in range(1, len(heads) + 1):
            for sub in combinations(range(len(heads)), k):
                name = "+".join(heads[i] for i in sub)
                cols = list(sub)
                pred = pool(
                    rule, d["oof"][:, cols], d["test"][:, cols], d["y_train"], d["gate"]
                )
                subset_rel.setdefault(name, []).append(
                    _rmse(pred, d["y_test"]) / ref - 1
                )
    if oob_rel:
        v = np.array(oob_rel)
        print(
            f"RF OOF from OOB instead of folds ({rule}): mean {100 * v.mean():+.2f}%  "
            f"median {100 * np.median(v):+.2f}%  worse on {(v > 1e-9).sum()}/{len(v)}"
        )
    print(f"head subsets ({rule}), RMSE vs all heads:")
    for name, v in sorted(subset_rel.items(), key=lambda kv: np.mean(kv[1])):
        v = np.array(v)
        print(
            f"  {name:18s} n={len(v):3d}  mean {100 * v.mean():+.2f}%  median {100 * np.median(v):+.2f}%"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["collect", "score", "ablations"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    args = ap.parse_args()
    if args.action == "collect":
        collect(args.label, args.tasks or task_registry.task_names())
    elif args.action == "score":
        score(args.label)
    else:
        ablations(args.label)


if __name__ == "__main__":
    main()

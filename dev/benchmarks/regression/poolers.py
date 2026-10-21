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
            np.savez_compressed(
                path,
                heads=np.asarray(batch.portfolio),
                oof=oof,
                test=test,
                gate=gate,
                y_train=(y[tr] - inner.y_mean_) / inner.y_std_,
                y_test=(y[te] - inner.y_mean_) / inner.y_std_,
                single_batch=np.asarray(len(inner.models) == 1),
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["collect", "score"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    args = ap.parse_args()
    if args.action == "collect":
        collect(args.label, args.tasks or task_registry.task_names())
    else:
        score(args.label)


if __name__ == "__main__":
    main()

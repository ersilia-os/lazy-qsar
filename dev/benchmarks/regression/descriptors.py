"""Descriptor-combination study: how to blend per-descriptor regressors, scored offline.

    python dev/benchmarks/regression/descriptors.py collect --label desc-v1
    python dev/benchmarks/regression/descriptors.py score --label desc-v1

``collect`` fits one ``LazyRegressor`` per descriptor (Morgan and RDKit: the two that
need no model download) on each (task, split) -- ``random-0`` and the structural split
-- plus an applicability domain per descriptor, and keeps each descriptor's out-of-fold
predictions and ranks, test predictions and ranks, and AD scores. ``score`` compares
combination rules on exactly those numbers:

- ``current``   -- :func:`lazyqsar.ensemble.combine` as ``LazyRegressorQSAR`` uses it:
  skill ``2*rho_oof - rho_train`` blended with the per-sample rank -> error reliability
  curve, AD veto;
- ``no_curve``  -- the same without the reliability curve (skill and AD veto only);
- ``nnls``      -- non-negative least squares on the descriptors' OOF predictions, then
  the AD veto (weights renormalised over the surviving descriptors);
- ``mean``      -- the plain mean;
- ``best``      -- the descriptor with the best OOF RMSE.
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
DESCRIPTORS = ("morgan", "rdkit")


def collect(label, task_names):
    from lazyqsar.agnostic import LazyRegressor
    from lazyqsar.applicability import ApplicabilityDomain
    from lazyqsar.qsar_regressor import _quality_rho, _rank_error_curve

    out_dir = os.path.join(CACHE, "descriptors", label)
    os.makedirs(out_dir, exist_ok=True)
    for task in task_names:
        feats = {}
        for d in DESCRIPTORS:
            smiles, y, X, provided = task_registry.load(task, d)
            feats[d] = X
        for split, tr, te in task_registry.splits(task, smiles, provided, n_random=1):
            path = os.path.join(out_dir, f"{task}__{split}.npz")
            if os.path.isfile(path):
                continue
            arrays = {"y_train": y[tr], "y_test": y[te]}
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                for d in DESCRIPTORS:
                    X = feats[d]
                    model = LazyRegressor()
                    model.fit(X=X[tr], y=y[tr])
                    value, rank = model.oof_channels(X=X[tr])
                    ad = ApplicabilityDomain()
                    ad.fit(X[tr])
                    ad_train = ad.score(X[tr])
                    test_value = model.predict(X=X[te])
                    arrays[f"{d}_oof"] = value
                    arrays[f"{d}_oof_rank"] = rank
                    arrays[f"{d}_test"] = test_value
                    arrays[f"{d}_test_rank"] = model._oof_percentile_from_value(
                        test_value
                    )
                    arrays[f"{d}_ad_train"] = ad_train
                    arrays[f"{d}_ad_test"] = ad.score(X[te])
                    arrays[f"{d}_ad_cutoff"] = np.float64(np.percentile(ad_train, 5))
                    arrays[f"{d}_quality"] = np.float64(
                        _quality_rho(model.oof_rho_, model.train_rho_) or 0.0
                    )
                    curve = _rank_error_curve(rank, value, y[tr])
                    arrays[f"{d}_curve"] = (
                        np.asarray(curve, dtype=float) if curve else np.empty(0)
                    )
            np.savez_compressed(path, **arrays)
            print(f"{task:28s} {split:9s} done", flush=True)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _veto(W, A, cutoffs, base):
    W = W.copy()
    for j, c in enumerate(cutoffs):
        W[A[:, j] < c, j] = 0.0
    dead = W.sum(axis=1) == 0
    W[dead] = base
    return W / W.sum(axis=1, keepdims=True)


def combined(rule, d):
    from lazyqsar.ensemble import EnsembleSpec, combine
    from lazyqsar.poolers.regression.inner_pooler import stacking_weights

    Y = np.column_stack([d[f"{k}_test"] for k in DESCRIPTORS])
    R = np.column_stack([d[f"{k}_test_rank"] for k in DESCRIPTORS])
    A = np.column_stack([d[f"{k}_ad_test"] for k in DESCRIPTORS])
    cut = tuple(float(d[f"{k}_ad_cutoff"]) for k in DESCRIPTORS)
    quality = tuple(float(d[f"{k}_quality"]) for k in DESCRIPTORS)
    curves = tuple(
        (d[f"{k}_curve"][0], d[f"{k}_curve"][1]) if d[f"{k}_curve"].size else None
        for k in DESCRIPTORS
    )
    if rule in ("current", "no_curve"):
        spec = EnsembleSpec(
            descriptor_names=DESCRIPTORS,
            oof_aucs=quality,
            rank_error_curves=curves if rule == "current" else None,
            ad_hard_cutoffs=cut,
            task="regression",
        )
        return combine(Y, R, None, A, spec=spec, outputs=("value",)).values["value"]
    P = np.column_stack([d[f"{k}_oof"] for k in DESCRIPTORS])
    if rule == "mean":
        return Y.mean(axis=1)
    if rule == "best":
        errors = [_rmse(P[:, j], d["y_train"]) for j in range(P.shape[1])]
        return Y[:, int(np.argmin(errors))]
    if rule == "nnls":
        w = stacking_weights(P, d["y_train"])
        W = _veto(np.tile(w, (len(Y), 1)), A, cut, w if w.sum() else np.ones_like(w))
        return (W * Y).sum(axis=1)
    raise KeyError(rule)


RULES = ("current", "no_curve", "nnls", "mean", "best")


def score(label):
    in_dir = os.path.join(CACHE, "descriptors", label)
    rows = []
    for fn in sorted(os.listdir(in_dir)):
        d = np.load(os.path.join(in_dir, fn))
        task, split = fn[: -len(".npz")].split("__")
        row = {"task": task, "split": split}
        for rule in RULES:
            row[rule] = _rmse(combined(rule, d), d["y_test"])
        rows.append(row)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"descriptors-{label}.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"{len(rows)} (task, split) pairs; RMSE relative to `current`:")
    for rule in RULES:
        rel = np.array([r[rule] / r["current"] - 1 for r in rows])
        print(
            f"  {rule:9s} mean {100 * rel.mean():+.2f}%  median "
            f"{100 * np.median(rel):+.2f}%  better on {(rel < -1e-9).sum()}/{len(rel)}"
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

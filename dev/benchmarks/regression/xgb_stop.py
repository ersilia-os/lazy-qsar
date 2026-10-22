"""XGBoost stopping-rule study: how many boosting rounds the regression head needs.

    python dev/benchmarks/regression/xgb_stop.py collect --label stop-v1
    python dev/benchmarks/regression/xgb_stop.py score --label stop-v1

The regression head early-stops on a held-out 10% with a patience of 50 rounds and no
minimum improvement; at eta 0.08 the validation RMSE keeps creeping down and a fit can run
past 1,000 rounds (TDC lipophilicity: 1,100 rounds, 4x the classifier's fit time).
``collect`` reproduces the head's own phase 1 -- same preset, same quantile-stratified
split -- on each (task, split), and keeps the full validation curve plus the test RMSE
at every round of that booster. ``score`` replays stopping rules (patience, minimum
improvement ``min_delta`` in standardised-RMSE units) on the stored curves, so every rule
is judged on identical boosters: test RMSE relative to the shipped rule, and rounds (the
fit time of phase 2 and of every OOF fold is proportional to them).

The booster scored here is the phase-1 one (90% of the training rows); the shipped head
refits on all rows for the chosen round count, so the relative numbers are the measure.
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
RULES = [(50, 0.0), (50, 1e-4), (50, 3e-4), (50, 1e-3), (25, 0.0), (25, 3e-4), (25, 1e-3)]


def collect(label, task_names, min_n, max_n=10**9):
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    from lazyqsar.base.xgboost import model as xgb_model
    from lazyqsar.base.xgboost.inspector import inspect
    from lazyqsar.base.xgboost.presets import probst_regression_params
    from lazyqsar.preprocessors import Preprocessor
    from lazyqsar.utils.splits import quantile_strata

    out_path = os.path.join(RESULTS, f"xgb_stop-{label}.jsonl")
    os.makedirs(RESULTS, exist_ok=True)
    done = set()
    if os.path.isfile(out_path):
        done = {(r["task"], r["split"]) for r in map(json.loads, open(out_path))}
    for task in task_names:
        smiles, y, X, provided = task_registry.load(task)
        for split, tr, te in task_registry.splits(task, smiles, provided, n_random=1):
            if not min_n <= len(tr) <= max_n or (task, split) in done:
                continue
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                prep = Preprocessor("regression")
                prep.fit(X[tr], y[tr])
                Xtr, Xte = prep.transform(X[tr]), prep.transform(X[te])
                mu, sd = float(y[tr].mean()), float(y[tr].std()) or 1.0
                z = (y[tr] - mu) / sd
                profile = inspect(Xtr, z, task="regression")
            params = probst_regression_params(profile, device="cpu")
            params["nthread"] = os.cpu_count() or 1
            a, b, za, zb = train_test_split(
                Xtr,
                z,
                test_size=xgb_model._VAL_FRACTION,
                random_state=xgb_model._RANDOM_STATE,
                stratify=quantile_strata(z, 5, min_count=2),
            )
            booster_params = {
                k: v for k, v in params.items() if k not in xgb_model._META_KEYS
            }
            dtrain = xgb.QuantileDMatrix(a, label=za, max_bin=params.get("max_bin", 256))
            dval = xgb.QuantileDMatrix(b, label=zb, ref=dtrain)
            history = {}
            booster = xgb.train(
                booster_params,
                dtrain,
                num_boost_round=params["n_estimators"],
                evals=[(dval, "val")],
                early_stopping_rounds=params["early_stopping_rounds"],
                evals_result=history,
                verbose_eval=False,
            )
            val_curve = history["val"]["rmse"]
            dtest = xgb.DMatrix(Xte)
            zte = (y[te] - mu) / sd
            # Test RMSE every 5 rounds (and at the last), in standardised units.
            rounds = sorted(set(range(5, len(val_curve) + 1, 5)) | {len(val_curve)})
            test_curve = {
                r: float(
                    np.sqrt(
                        np.mean((booster.predict(dtest, iteration_range=(0, r)) - zte) ** 2)
                    )
                )
                for r in rounds
            }
            record = {
                "task": task,
                "split": split,
                "n_train": int(len(tr)),
                "eta": params["learning_rate"],
                "val_curve": [float(v) for v in val_curve],
                "test_curve": test_curve,
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(
                f"{task:28s} {split:9s} n={len(tr):5d} rounds={booster.best_iteration + 1}",
                flush=True,
            )


def stop_round(curve, patience, min_delta):
    """Rounds kept (best iteration + 1) by early stopping with a minimum improvement."""
    best, best_i = float("inf"), 0
    for i, v in enumerate(curve):
        if v < best - min_delta:
            best, best_i = v, i
        elif i - best_i >= patience:
            break
    return best_i + 1


def _test_at(test_curve, r):
    keys = sorted(int(k) for k in test_curve)
    nearest = min(keys, key=lambda k: abs(k - r))
    return test_curve[str(nearest)]


def score(label):
    rows = [json.loads(l) for l in open(os.path.join(RESULTS, f"xgb_stop-{label}.jsonl"))]
    print(f"{len(rows)} (task, split) pairs; relative to patience 50, no min_delta:")
    for lo, hi in ((0, 2000), (2000, 5000), (5000, 10**9), (0, 10**9)):
        sub = [r for r in rows if lo < r["n_train"] <= hi]
        if not sub:
            continue
        print(f"  n in ({lo}, {hi}]: {len(sub)} splits")
        for patience, delta in RULES:
            rel, ratio = [], []
            for r in sub:
                ref = stop_round(r["val_curve"], 50, 0.0)
                got = stop_round(r["val_curve"], patience, delta)
                rel.append(_test_at(r["test_curve"], got) / _test_at(r["test_curve"], ref) - 1)
                ratio.append(got / ref)
            rel = np.array(rel)
            print(
                f"    patience {patience:3d} min_delta {delta:<6g} test RMSE "
                f"{100 * rel.mean():+.2f}% (worst {100 * rel.max():+.2f}%)  "
                f"rounds x{np.mean(ratio):.2f} (median x{np.median(ratio):.2f})"
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["collect", "score"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--min-n", type=int, default=500)
    ap.add_argument("--max-n", type=int, default=10**9)
    args = ap.parse_args()
    if args.action == "collect":
        collect(
            args.label, args.tasks or task_registry.task_names(), args.min_n, args.max_n
        )
    else:
        score(args.label)


if __name__ == "__main__":
    main()

"""Run the regression benchmark for the code currently checked out.

    python dev/benchmarks/regression/run.py --label baseline
    python dev/benchmarks/regression/run.py --label candidate --tasks tdc_caco2 ace_CHEMBL204_Ki
    python dev/benchmarks/regression/run.py --label baseline --clf-time   # + classification time

Each (task, split) fit is one JSON line in ``.cache/results/<label>.jsonl``: metrics on
the held-out rows in the target's units (RMSE, MAE, Spearman, R²), fit seconds, and what
the model chose (heads, per-head preset and out-of-fold ρ) so a change in the portfolio
can be traced to the decision that moved. Re-running skips (task, split) pairs already
recorded under the label, so an interrupted run resumes.

``--clf-time`` additionally fits a ``LazyClassifier`` on the same training rows with the
target binarised at its median -- the time reference for the "regression fit at most 2x
classification" acceptance criterion.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tasks as task_registry  # noqa: E402

RESULTS = os.path.join(HERE, ".cache", "results")


def _metrics(y, p):
    from lazyqsar.utils.metrics import regression_metrics

    return regression_metrics(y, p)


def _describe(model):
    """Heads and per-head choices of the (first batch of the) fitted assembler."""
    inner = model._model
    batch = inner.models[0]
    heads = {}
    for name, head in zip(batch.portfolio, batch.heads):
        m = head.model
        heads[name] = {
            "preset": getattr(m, "preset_name_", None)
            or getattr(m, "selected_preset_", None)
            or getattr(m, "regime_", None),
            "oof_rho": getattr(m, "oof_rho_", None),
        }
    return {
        "portfolio": list(batch.portfolio),
        "heads": heads,
        "n_batches": len(inner.models),
        "oof_rho": inner.oof_rho_,
    }


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True
        ).strip()
    except Exception:  # noqa: BLE001 - informational only
        return None


def run(label, task_names, n_random, clf_time, descriptor):
    from lazyqsar.agnostic import LazyClassifier, LazyRegressor

    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, f"{label}.jsonl")
    done = set()
    if os.path.isfile(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["task"], r["split"]))
    sha = _git_sha()
    for task in task_names:
        smiles, y, X, provided = task_registry.load(task, descriptor)
        for split, tr, te in task_registry.splits(task, smiles, provided, n_random):
            if (task, split) in done:
                continue
            record = {
                "label": label,
                "git": sha,
                "task": task,
                "split": split,
                "descriptor": descriptor,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
            }
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                t0 = time.perf_counter()
                model = LazyRegressor()
                model.fit(X=X[tr], y=y[tr])
                record["fit_seconds"] = time.perf_counter() - t0
                record["test"] = _metrics(y[te], model.predict(X=X[te]))
                record["model"] = _describe(model)
                if clf_time:
                    yb = (y[tr] > np.median(y[tr])).astype(int)
                    t0 = time.perf_counter()
                    LazyClassifier().fit(X=X[tr], y=yb)
                    record["clf_fit_seconds"] = time.perf_counter() - t0
            with open(out_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            m = record["test"]
            print(
                f"{task:30s} {split:10s} n={len(tr):5d} "
                f"rmse={m['rmse']:.3f} rho={m['spearman']:.3f} "
                f"t={record['fit_seconds']:.1f}s"
                + (f" clf={record['clf_fit_seconds']:.1f}s" if clf_time else ""),
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--clf-time", action="store_true")
    ap.add_argument("--descriptor", default="morgan")
    args = ap.parse_args()
    names = args.tasks or task_registry.task_names()
    run(args.label, names, args.n_random, args.clf_time, args.descriptor)


if __name__ == "__main__":
    main()

"""SVR size-guard study: what the SVR slot is worth once its ONNX graph is capped.

    python dev/benchmarks/regression/svr_guard.py collect --label guard-v1
    python dev/benchmarks/regression/svr_guard.py score --label guard-v1

The kernel SVR's ONNX graph stores every support vector densely (float32 x features), so
the 5 MB cap (before 3.8) allowed about 600 support vectors on 2,048 Morgan bits; past it the head
halves C three times and then falls back to a LinearSVR. ``collect`` fits, through the
real regression heads, on each (task, split) between ``--min-n`` and ``--max-n`` rows:

- ``lr``, ``xgb``, ``rf``          -- the other heads as shipped;
- ``svr``                          -- the SVR as shipped (guarded);
- ``svr_kernel``                   -- the same SVR with no size cap;
- ``svr_eps<e>``                   -- the kernel SVR with a larger epsilon (fewer
  support vectors), no cap.

and keeps each one's out-of-fold and test predictions, fit seconds and ONNX bytes.
``score`` stacks subsets with NNLS on the OOF predictions (as the pooler does) and reports
test RMSE relative to the shipped ``lr + xgb + svr``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tasks as task_registry  # noqa: E402

CACHE = os.path.join(HERE, ".cache")
RESULTS = os.path.join(CACHE, "results")
EPSILONS = (0.2, 0.4)


def _fit_head(name, Xtr, ytr, Xte, cap=True, epsilon=None):
    from lazyqsar.base.svc import regressor as svr_module
    from lazyqsar.heads import Head

    old_cap, old_params = svr_module.BaseSVR._onnx_max_bytes, svr_module.svr_rbf_params
    if not cap:
        svr_module.BaseSVR._onnx_max_bytes = float("inf")
    if epsilon is not None:
        svr_module.svr_rbf_params = lambda profile: dict(
            old_params(profile), epsilon=epsilon
        )
    try:
        t0 = time.perf_counter()
        head = Head("regression", name)
        head.model.fit(Xtr, ytr)
        seconds = time.perf_counter() - t0
        model = head.model
        size = len(model._onnx_bytes_) if getattr(model, "_onnx_bytes_", None) else None
        return (
            model.oof_preds_,
            model.predict(Xte),
            seconds,
            size,
            model.preset_name_ if hasattr(model, "preset_name_") else None,
        )
    finally:
        svr_module.BaseSVR._onnx_max_bytes = old_cap
        svr_module.svr_rbf_params = old_params


def collect(label, task_names, min_n, max_n):
    from lazyqsar.preprocessors import Preprocessor

    out_dir = os.path.join(CACHE, "svr_guard", label)
    os.makedirs(out_dir, exist_ok=True)
    for task in task_names:
        smiles, y, X, provided = task_registry.load(task)
        for split, tr, te in task_registry.splits(task, smiles, provided, n_random=1):
            if not (min_n <= len(tr) <= max_n):
                continue
            path = os.path.join(out_dir, f"{task}__{split}.npz")
            if os.path.isfile(path):
                continue
            sink = io.StringIO()
            arrays = {"y_train": y[tr], "y_test": y[te]}
            meta = {}
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                prep = Preprocessor("regression")
                prep.fit(X[tr], y[tr])
                Xtr, Xte = prep.transform(X[tr]), prep.transform(X[te])
                mu, sd = float(y[tr].mean()), float(y[tr].std()) or 1.0
                ztr = (y[tr] - mu) / sd
                runs = [("lr", {}), ("xgb", {}), ("rf", {}), ("svr", {})]
                runs += [("svr_kernel", {"cap": False})]
                runs += [
                    (f"svr_eps{e}", {"cap": False, "epsilon": e}) for e in EPSILONS
                ]
                for key, kw in runs:
                    name = "svr" if key.startswith("svr") else key
                    oof, test, seconds, size, preset = _fit_head(
                        name, Xtr, ztr, Xte, **kw
                    )
                    arrays[f"{key}_oof"] = np.asarray(oof) * sd + mu
                    arrays[f"{key}_test"] = np.asarray(test) * sd + mu
                    meta[key] = {
                        "seconds": seconds,
                        "onnx_bytes": size,
                        "preset": preset,
                    }
            arrays["meta"] = np.asarray(json.dumps(meta))
            np.savez_compressed(path, **arrays)
            print(
                f"{task:28s} {split:9s} n={len(tr):5d} "
                + " ".join(
                    f"{k}={v['seconds']:.1f}s/{(v['onnx_bytes'] or 0) / 1e6:.1f}MB"
                    for k, v in meta.items()
                ),
                flush=True,
            )


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _stack(d, heads):
    from lazyqsar.poolers.regression.inner_pooler import stacking_weights

    P = np.column_stack([d[f"{h}_oof"] for h in heads])
    T = np.column_stack([d[f"{h}_test"] for h in heads])
    ok = np.all(np.isfinite(P), axis=1)
    w = stacking_weights(P[ok], d["y_train"][ok])
    return T @ w, w


def score(label):
    in_dir = os.path.join(CACHE, "svr_guard", label)
    svrs = ["svr", "svr_kernel"] + [f"svr_eps{e}" for e in EPSILONS]
    subsets = [("lr", "xgb")] + [("lr", "xgb", s) for s in svrs] + [("lr", "xgb", "rf")]
    rows = []
    for fn in sorted(os.listdir(in_dir)):
        d = np.load(os.path.join(in_dir, fn))
        meta = json.loads(str(d["meta"]))
        row = {"file": fn, "n": len(d["y_train"]), "meta": meta}
        for s in subsets:
            pred, w = _stack(d, s)
            row["+".join(s)] = _rmse(pred, d["y_test"])
            row["w:" + "+".join(s)] = [round(float(x), 3) for x in w]
        rows.append(row)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"svr_guard-{label}.json"), "w") as f:
        json.dump(rows, f, indent=1)
    ref = "lr+xgb+svr"
    print(
        f"{len(rows)} (task, split) pairs; test RMSE of the NNLS stack relative to {ref}:"
    )
    for s in subsets:
        key = "+".join(s)
        rel = np.array([r[key] / r[ref] - 1 for r in rows])
        w_last = np.mean([r["w:" + key][-1] for r in rows])
        print(
            f"  {key:22s} mean {100 * rel.mean():+.2f}%  better on "
            f"{(rel < -1e-9).sum()}/{len(rel)}  mean weight of last head {w_last:.2f}"
        )
    for key in ["svr"] + svrs[1:] + ["rf"]:
        secs = [r["meta"][key]["seconds"] for r in rows]
        mb = [(r["meta"][key]["onnx_bytes"] or 0) / 1e6 for r in rows]
        print(
            f"  {key:12s} fit {np.median(secs):.1f}s median  ONNX {np.median(mb):.1f} MB "
            f"median / {np.max(mb):.1f} MB max"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["collect", "score"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--min-n", type=int, default=600)
    ap.add_argument("--max-n", type=int, default=2000)
    args = ap.parse_args()
    if args.action == "collect":
        collect(
            args.label, args.tasks or task_registry.task_names(), args.min_n, args.max_n
        )
    else:
        score(args.label)


if __name__ == "__main__":
    main()

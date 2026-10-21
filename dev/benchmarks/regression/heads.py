"""Head-level preset study: each candidate preset, fitted alone, on the benchmark tasks.

    python dev/benchmarks/regression/heads.py --label presets-v1
    python dev/benchmarks/regression/heads.py --label presets-v1 --heads rf svr

Answers "which presets should each regression head race between" without paying for a
full ensemble fit per variant. For every (task, split) the features go through LazyQSAR's
own regression preprocessor (fitted on the training rows), the target is standardised as
the assembler does, and each preset is fitted *by itself* on the training rows. Held-out
RMSE (original units), Spearman and fit seconds go to
``.cache/results/heads-<label>.jsonl``.

Splits: ``random-0`` plus the structural split (scaffold, or MoleculeACE's own), which is
where presets that overfit training chemotypes are found out.

Presets and their sources are listed in ``PRESETS`` below; each builder takes the
dataset shape and returns an unfitted estimator, so this file is the single record of
what was compared.
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

RESULTS = os.path.join(HERE, ".cache", "results")


# ------------------------------------------------------------------ preset builders


def _rf(**kw):
    from sklearn.ensemble import RandomForestRegressor

    def build(n, p, sparse):
        params = dict(kw)
        n_est = 100 if n < 500 else (200 if n < 5000 else 300)
        params.setdefault("n_estimators", n_est)
        mf = params.get("max_features")
        if callable(mf):
            params["max_features"] = mf(n, p, sparse)
        return RandomForestRegressor(n_jobs=-1, random_state=42, **params)

    return build


def _svr(kind, **kw):
    def build(n, p, sparse, z_train=None, X_train=None):
        from sklearn.svm import SVR, LinearSVR

        if kind == "linear":
            C = kw.get("C", 0.1 if n < 500 else (1.0 if n < 2000 else 10.0))
            return LinearSVR(
                C=C,
                epsilon=0.0,
                loss="squared_epsilon_insensitive",
                dual="auto",
                max_iter=10_000,
                random_state=42,
            )
        C, eps = kw.get("C", 1.0), kw.get("epsilon", 0.1)
        if kw.get("cherkassky_ma"):
            C, eps = _cherkassky_ma(z_train, X_train)
        return SVR(kernel="rbf", C=C, epsilon=eps, gamma="scale")

    build.needs_data = kind != "linear"
    return build


def _cherkassky_ma(z, X, k=5):
    """Cherkassky & Ma (2004): C from the target's spread, epsilon from noise level.

    C = max(|mean + 3 sd|, |mean - 3 sd|) and epsilon = 3 * sigma * sqrt(ln n / n), with
    the noise sigma estimated from k-nearest-neighbour regression residuals (their
    recommendation for when noise is unknown).
    """
    from sklearn.neighbors import KNeighborsRegressor

    n = len(z)
    C = float(max(abs(z.mean() + 3 * z.std()), abs(z.mean() - 3 * z.std())))
    knn = KNeighborsRegressor(n_neighbors=k + 1).fit(X, z)
    # Exclude the point itself: leave-one-out kNN prediction.
    dist, idx = knn.kneighbors(X)
    pred = z[idx[:, 1:]].mean(axis=1)
    sigma = float(np.sqrt(np.mean((z - pred) ** 2) * k / (k - 1)))
    eps = 3.0 * sigma * np.sqrt(np.log(n) / n)
    return C, eps


def _xgb(**kw):
    def build(n, p, sparse):
        return dict(kw)

    build.is_xgb = True
    return build


def _ridge(kind):
    def build(n, p, sparse):
        from sklearn.linear_model import ElasticNetCV, RidgeCV

        if kind == "ridge_gcv":
            return RidgeCV(alphas=float(n) * np.logspace(-4, 2, 13))
        return ElasticNetCV(l1_ratio=[0.5], cv=5, max_iter=5000, n_jobs=-1)

    return build


PRESETS = {
    "rf": {
        # sklearn RandomForestRegressor default: bagged trees, all features per split.
        "sklearn_default": _rf(max_features=1.0, min_samples_leaf=1),
        # Breiman (2001) / R randomForest regression defaults: mtry = p/3, nodesize 5.
        "breiman_p3_leaf5": _rf(max_features=1 / 3, min_samples_leaf=5),
        # Probst et al. 2019 (JMLR) tuned optimum for ranger: mtry ~ 0.26 p,
        # sample fraction ~ 0.7 (classification-derived; tested here for regression).
        "probst": _rf(max_features=0.26, max_samples=0.7, min_samples_leaf=1),
        # The current 3.8.0 heuristic: p/3 dense, log2 on wide sparse fingerprints.
        "lazyqsar_3_8_heuristic": _rf(
            max_features=lambda n, p, s: "log2" if (s and p > 500) else 0.33,
            min_samples_leaf=1,
        ),
        # Classification's sqrt rule, for reference.
        "sqrt": _rf(max_features="sqrt", min_samples_leaf=1),
    },
    "xgb": {
        # xgboost defaults (eta 0.3, depth 6): the baseline every preset must beat.
        "xgb_default": _xgb(eta=0.3, max_depth=6),
        # Probst, Boulesteix & Bischl (2019) "Tunability" new defaults, validated on
        # OpenML regression: eta 0.052, depth 11, min_child_weight 2, subsample 0.87,
        # colsample_bytree 0.71, colsample_bylevel 0.64.
        "probst": _xgb(
            eta=0.052,
            max_depth=11,
            min_child_weight=2,
            subsample=0.87,
            colsample_bytree=0.71,
            colsample_bylevel=0.64,
        ),
        # Same, shallower: QSAR boosting work (Sheridan 2016) uses depth ~6-8.
        "probst_depth7": _xgb(
            eta=0.052,
            max_depth=7,
            min_child_weight=2,
            subsample=0.87,
            colsample_bytree=0.71,
            colsample_bylevel=0.64,
        ),
        # AutoGluon tabular XGBoost default (lr 0.1, depth 6, colsample 1.0).
        "autogluon": _xgb(eta=0.1, max_depth=6),
        # Slow-and-wide: eta 0.05, depth 6, subsample 0.8, colsample 0.5 (fingerprint-
        # style column subsampling, common in QSAR GBM practice).
        "qsar_colsample": _xgb(
            eta=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.5
        ),
    },
    "svr": {
        # sklearn SVR default on standardised y: C=1, epsilon=0.1, gamma='scale'.
        "sklearn_default": _svr("rbf", C=1.0, epsilon=0.1),
        # Cherkassky & Ma (2004) analytic C and epsilon.
        "cherkassky_ma": _svr("rbf", cherkassky_ma=True),
        # Wider margin penalty, for low-noise targets.
        "c10": _svr("rbf", C=10.0, epsilon=0.1),
        "linear": _svr("linear"),
    },
    "lr": {
        "ridge_gcv": _ridge("ridge_gcv"),
        "elasticnet_cv": _ridge("elasticnet_cv"),
    },
}


# ------------------------------------------------------------------ fitting


def _fit_predict(builder, Xtr, ztr, Xte, n, p, sparse):
    if getattr(builder, "is_xgb", False):
        import xgboost as xgb
        from sklearn.model_selection import train_test_split

        from lazyqsar.utils.splits import quantile_bins

        params = builder(n, p, sparse)
        params.update({"objective": "reg:squarederror", "verbosity": 0})
        a, b, za, zb = train_test_split(
            Xtr, ztr, test_size=0.1, random_state=42, stratify=quantile_bins(ztr, 5)
        )
        booster = xgb.train(
            params,
            xgb.DMatrix(a, label=za),
            num_boost_round=3000,
            evals=[(xgb.DMatrix(b, label=zb), "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        rounds = booster.best_iteration + 1
        booster = xgb.train(params, xgb.DMatrix(Xtr, label=ztr), num_boost_round=rounds)
        return booster.predict(xgb.DMatrix(Xte))
    if getattr(builder, "needs_data", False):
        est = builder(n, p, sparse, z_train=ztr, X_train=Xtr)
    else:
        est = builder(n, p, sparse)
    return est.fit(Xtr, ztr).predict(Xte)


def run(label, heads, task_names, max_n):
    from lazyqsar.base.preprocessing import BaseRegressorPreprocessor
    from lazyqsar.utils.metrics import regression_metrics

    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, f"heads-{label}.jsonl")
    done = set()
    if os.path.isfile(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["task"], r["split"], r["head"], r["preset"]))
    for task in task_names:
        smiles, y, X, provided = task_registry.load(task)
        if len(y) > max_n:
            continue
        for split, tr, te in task_registry.splits(task, smiles, provided, n_random=1):
            with contextlib.redirect_stdout(io.StringIO()):
                prep = BaseRegressorPreprocessor().fit(X[tr], y[tr])
                Xtr, Xte = prep.transform(X[tr]), prep.transform(X[te])
            mu, sd = float(y[tr].mean()), float(y[tr].std()) or 1.0
            ztr = (y[tr] - mu) / sd
            n, p = Xtr.shape
            sparse = float((X == 0).mean()) > 0.85
            for head in heads:
                for preset, builder in PRESETS[head].items():
                    if (task, split, head, preset) in done:
                        continue
                    t0 = time.perf_counter()
                    try:
                        pred = _fit_predict(builder, Xtr, ztr, Xte, n, p, sparse)
                        record = {"test": regression_metrics(y[te], pred * sd + mu)}
                    except Exception as exc:  # noqa: BLE001 - record and move on
                        record = {"error": f"{type(exc).__name__}: {exc}"}
                    record.update(
                        {
                            "task": task,
                            "split": split,
                            "head": head,
                            "preset": preset,
                            "n_train": int(n),
                            "p": int(p),
                            "fit_seconds": time.perf_counter() - t0,
                        }
                    )
                    with open(out_path, "a") as f:
                        f.write(json.dumps(record) + "\n")
                    m = record.get("test")
                    print(
                        f"{task:28s} {split:9s} {head:4s} {preset:24s} "
                        + (
                            f"rmse={m['rmse']:.3f} rho={m['spearman']:.3f}"
                            if m
                            else record["error"][:60]
                        )
                        + f" t={record['fit_seconds']:.1f}s",
                        flush=True,
                    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--heads", nargs="*", default=list(PRESETS))
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--max-n", type=int, default=100_000)
    args = ap.parse_args()
    run(args.label, args.heads, args.tasks or task_registry.task_names(), args.max_n)


if __name__ == "__main__":
    main()

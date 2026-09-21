"""Per-head SVC ONNX round trip, with the inversion invariant and both flavours.

Run it directly; it needs the ``fit`` extra and nothing else::

    python dev/tools/check_svc_head_roundtrip.py

Why this exists
---------------
v3.4.3 shipped an inverted SVC score column. ``skl2onnx`` routes ``LinearSVC`` and kernel
``SVC`` through different converters, and those converters order the two mutually-negated
score columns oppositely. 3.4.3 fixed one and broke the other, and it shipped because the
tests of the day used two datasets that both happened to produce kernel SVC. Half the
flavours is half a test.

The invariant
-------------
Within one head the calibrated probability is a *monotone* function of the raw score, so
``AUROC(onnx) == AUROC(fit)`` no matter how far float32 export moves individual values. An
inverted column does not degrade the AUROC, it **reflects** it::

    AUROC(onnx) == 1 - AUROC(fit)

A tolerance check reports that as a large delta among other large deltas. This reports it
as what it is. The detector is itself checked, by injecting an inversion and confirming it
is caught -- a guard that cannot fail is not a guard.

Coverage is asserted, not hoped for
-----------------------------------
Which flavour a model gets is decided by the data, not requested. The heuristic in
``base/svc/params.py`` proposes one, but the decision that actually sticks is the **5 MB
ONNX budget** in ``base/svc/model.py``: a kernel SVC is fitted first, its export size is
estimated as ``n_support_vectors x n_features x 4`` bytes, and only if that exceeds
``_ONNX_MAX_BYTES`` is it refitted as ``LinearSVC``. Sparse count features alone do *not*
force the linear branch -- verified: a 400x256 Morgan-like matrix profiles as
``is_sparse_counts=True`` and still selects kernel SVC.

There is a second selector in front of that: ``_portfolio_select_svc`` fits four presets
(``heuristic``, ``default``, ``linear``, ``balanced_rbf``) and keeps whichever scores best,
so on linearly separable data the ``linear`` preset wins on merit whatever the budget says.
The two datasets here therefore differ in the shape of the boundary, not just in size: one
wide, sparse and noisy enough to blow the export budget, one with a radial boundary a
linear model cannot find. The script fails if it did not in fact exercise both -- which it
did, twice, during its own development.

This replaces the head-level coverage that ``tests/fit/test_head_onnx_roundtrip.py`` and
``tests/fit/test_svc_score_column.py`` used to provide. The pytest suite now only compares
whole models, where a single inverted head is damped by its neighbours: 3.4.3's inversion
was visible end-to-end only because it was total.
"""

import shutil
import sys
import tempfile

import numpy as np

AUROC_TOL = 5e-3
SCORE_TOL = 1e-4
SEED = 42


def auroc(y, s):
    """Rank-based AUROC, so nothing here depends on scikit-learn's metrics."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # Mid-ranks for ties, or a plateau would bias the estimate.
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("AUROC needs both classes present")
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def budget_busting_data(n=1200, p=2048, rng=None):
    """Wide, sparse and deliberately hard, so nearly every point is a support vector.

    ``n_sv x p x 4`` must exceed the 5 MB budget for the linear branch to be taken; at
    p=2048 that needs more than ~610 support vectors. Heavy label noise is what supplies
    them -- a cleanly separable problem would keep only a handful and stay on kernel SVC.
    """
    rng = rng or np.random.default_rng(SEED)
    X = rng.poisson(0.08, size=(n, p)).astype(np.float64)
    w = np.zeros(p)
    w[:12] = rng.normal(1.5, 0.3, size=12)
    y = (X @ w - 0.6 + rng.normal(0, 4.0, size=n) > 0).astype(int)
    return X, y


def radial_data(n=400, p=8, rng=None):
    """A boundary no linear model can find, so the RBF preset wins on merit.

    The portfolio picks by validation AUC, not by the size heuristic, and on linearly
    separable data the ``linear`` preset simply wins -- which is how two earlier versions
    of this script accidentally exercised one flavour twice. A radial boundary inverts
    that: a linear model scores near chance, so ``heuristic``/``balanced_rbf`` win and the
    kernel branch is taken.
    """
    rng = rng or np.random.default_rng(SEED + 1)
    X = rng.normal(size=(n, p))
    r2 = (X**2).sum(axis=1)
    y = (r2 > np.median(r2)).astype(int)
    return X, y


def fit_and_roundtrip(X, y, label):
    """Fit one SVC head, export it, and compare the export against the model it came from."""
    from lazyqsar.base.svc import BaseSVCArtifact, BaseSVCClassifier

    workdir = tempfile.mkdtemp(prefix=f"svc-{label}-")
    try:
        clf = BaseSVCClassifier(random_state=SEED)
        clf.fit(X, y)
        clf.save(workdir)
        art = BaseSVCArtifact.load(workdir)

        s_fit = np.asarray(clf.predict_score(X))[:, 1]
        s_onnx = np.asarray(art.predict_score(X))[:, 1]

        a_fit, a_onnx = auroc(y, s_fit), auroc(y, s_onnx)
        use_linear = bool(clf._use_linear_)
        return {
            "label": label,
            "flavour": "LinearSVC" if use_linear else "kernel SVC",
            "auroc_fit": a_fit,
            "auroc_onnx": a_onnx,
            "auroc_delta": abs(a_fit - a_onnx),
            "inverted": abs(a_onnx - (1.0 - a_fit)) < abs(a_onnx - a_fit),
            "max_score_delta": float(np.max(np.abs(s_fit - s_onnx))),
            "scores_fit": s_fit,
            "y": y,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def detector_catches_an_injected_inversion(row):
    """The guard must fail on a known-bad input, or it is decoration."""
    a_fit = row["auroc_fit"]
    a_onnx_inverted = auroc(row["y"], -row["scores_fit"])
    looks_inverted = abs(a_onnx_inverted - (1.0 - a_fit)) < abs(a_onnx_inverted - a_fit)
    return looks_inverted, a_onnx_inverted


def main():
    try:
        import lazyqsar  # noqa: F401
    except ImportError:
        print("lazyqsar is not importable here", file=sys.stderr)
        return 2

    rows = [
        fit_and_roundtrip(*budget_busting_data(), label="wide-noisy"),
        fit_and_roundtrip(*radial_data(), label="radial"),
    ]

    print(
        f"{'dataset':15s} {'flavour':12s} {'AUROC fit':>10s} {'AUROC onnx':>11s} "
        f"{'dAUROC':>9s} {'max|ds|':>10s}  inverted"
    )
    for r in rows:
        print(
            f"{r['label']:15s} {r['flavour']:12s} {r['auroc_fit']:10.5f} "
            f"{r['auroc_onnx']:11.5f} {r['auroc_delta']:9.2e} "
            f"{r['max_score_delta']:10.2e}  {r['inverted']}"
        )

    failures = []

    flavours = {r["flavour"] for r in rows}
    if flavours != {"LinearSVC", "kernel SVC"}:
        failures.append(
            f"both SVC flavours must be exercised; got {sorted(flavours)}. "
            "This is the exact condition under which v3.4.3 shipped an inversion."
        )

    for r in rows:
        if r["inverted"]:
            failures.append(
                f"[{r['label']}] score column is INVERTED across the export"
            )
        if r["auroc_delta"] > AUROC_TOL:
            failures.append(
                f"[{r['label']}] AUROC moved {r['auroc_delta']:.2e} > {AUROC_TOL:g}"
            )

    caught, a_inv = detector_catches_an_injected_inversion(rows[0])
    print(
        f"\ninversion detector, checked against an injected inversion: "
        f"AUROC {rows[0]['auroc_fit']:.5f} -> {a_inv:.5f}, caught={caught}"
    )
    if not caught:
        failures.append("the inversion detector did not catch an injected inversion")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: both flavours exercised, no inversion, AUROC stable across the export")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A saved model must score identically to the model it was saved from.

Regression test for the SVC sign flip fixed in 3.4.3: skl2onnx emits the binary-SVC score
output as two mutually-negated columns, and the artifact read the wrong one, so every
saved model containing an `svc` head scored that head backwards. Whole-model `predict()`
agreement against its own export was 21%.

The check deliberately covers every head in the portfolio, not just `svc` — positional
extraction from an ONNX output is the class of bug, and the other heads were only ever
verified by hand.
"""

import contextlib
import io
import tempfile

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("xgboost")
pytest.importorskip("skl2onnx")

import lazyqsar
from lazyqsar.agnostic import LazyClassifier

TOL = 1e-5


def _fit(n, d, pct, seed):
    lazyqsar.set_verbosity(0)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X @ rng.normal(size=d) > np.percentile(X @ rng.normal(size=d), pct)).astype(
        int
    )
    rng2 = np.random.default_rng(seed)
    Xd = rng2.normal(size=(n, d))
    w = rng2.normal(size=d)
    lg = Xd @ w
    y = (lg > np.percentile(lg, pct)).astype(int)
    model = LazyClassifier()
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X=Xd, y=y)
    return model, rng2.normal(loc=0.3, size=(600, d))


@pytest.mark.parametrize(
    "n,d,pct,seed",
    [
        (900, 35, 88, 3),  # portfolio: xgb, rf, svc
        (300, 150, 93, 0),  # small + high-dim: also pulls in lr
    ],
)
def test_each_head_matches_its_onnx_export(n, d, pct, seed):
    model, Xq = _fit(n, d, pct, seed)
    with tempfile.TemporaryDirectory() as td:
        with contextlib.redirect_stdout(io.StringIO()):
            model.save(td)
            loaded = LazyClassifier.load(td)

        batch_mem = model._model.models[0]
        batch_art = loaded._batches[0]
        X_prep = batch_mem.prep.transform(Xq)

        for name, h_mem, h_art in zip(
            batch_mem.portfolio, batch_mem.heads, batch_art.heads
        ):
            diff = np.abs(
                h_mem.predict_score(X_prep)[:, 1] - h_art.predict_score(X_prep)[:, 1]
            ).max()
            assert diff < TOL, (
                f"head {name!r} diverges from its ONNX export by {diff:g}"
            )

            diff = np.abs(
                h_mem.predict_proba(X_prep)[:, 1] - h_art.run(X_prep)[:, 1]
            ).max()
            assert diff < TOL, f"head {name!r} proba diverges by {diff:g}"

            # Guards two separate defects: estimators whose scores live on an exact grid
            # (RF vote fractions) used to flip labels on the cutoff between runtimes, and
            # the artifacts used to threshold the CALIBRATED probability against the RAW
            # cutoff while the fit-time model thresholded raw against raw.
            # XGBoostArtifact/LinearArtifact do not implement predict(); the pipeline
            # never calls it at head level, so only check the heads that do.
            if hasattr(h_art, "predict"):
                n_diff = int((h_mem.predict(X_prep) != h_art.predict(X_prep)).sum())
                assert n_diff == 0, (
                    f"head {name!r}: {n_diff} labels differ from ONNX export"
                )

        assert np.array_equal(model.predict(X=Xq), loaded.predict(X=Xq))
        assert (
            np.abs(
                model.predict_score(X=Xq)[:, 1] - loaded.predict_score(X=Xq)[:, 1]
            ).max()
            < TOL
        )


def test_legacy_model_without_score_canary_still_resolves():
    """Models saved before 3.4.3 carry no canary; the reader must still repair them."""
    import glob
    import json
    import os

    model, Xq = _fit(900, 35, 88, 3)
    with tempfile.TemporaryDirectory() as td:
        with contextlib.redirect_stdout(io.StringIO()):
            model.save(td)
        for path in glob.glob(os.path.join(td, "*", "svc.json")):
            with open(path) as fh:
                meta = json.load(fh)
            meta.pop("score_canary", None)
            with open(path, "w") as fh:
                json.dump(meta, fh)
        with contextlib.redirect_stdout(io.StringIO()):
            loaded = LazyClassifier.load(td)
        assert np.array_equal(model.predict(X=Xq), loaded.predict(X=Xq))

"""The four regression heads honour the head contract the ensemble relies on.

Each head (linear, XGBoost, random forest, SVR) must, after ``fit``:
  - carry out-of-fold predictions for every row (the gate's training signal),
  - report Spearman ρ on OOF and training predictions,
  - rank new predictions into [0, 1] monotonically via the OOF ECDF,
  - export to ONNX and come back through its artifact with the same predictions and ranks.

Synthetic data, small enough to keep the tier fast.
"""

import contextlib
import io

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")

from lazyqsar.artifacts.linear import LinearArtifact  # noqa: E402
from lazyqsar.artifacts.rf import RandomForestArtifact  # noqa: E402
from lazyqsar.artifacts.svc import SVRArtifact  # noqa: E402
from lazyqsar.artifacts.xgboost import XGBoostArtifact  # noqa: E402
from lazyqsar.base.linear.model import BaseLinearRegressor  # noqa: E402
from lazyqsar.base.randomforest import BaseRFRegressor  # noqa: E402
from lazyqsar.base.svc import BaseSVR  # noqa: E402
from lazyqsar.base.xgboost.model import BaseXGBRegressor  # noqa: E402
from lazyqsar.utils.splits import (  # noqa: E402
    check_trainable_regression,
    make_quantile_oof_splits,
)

HEADS = {
    "linear": (BaseLinearRegressor, LinearArtifact),
    "xgb": (BaseXGBRegressor, XGBoostArtifact),
    "rf": (BaseRFRegressor, RandomForestArtifact),
    "svr": (BaseSVR, SVRArtifact),
}


def _dataset(n=250, p=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    # Large offset and scale: heads must not assume y is standardised.
    y = (
        50.0 * (2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2])
        + 5.0 * rng.normal(size=n)
        + 300
    )
    return X, y


@pytest.fixture(scope="module", params=sorted(HEADS))
def fitted(request, tmp_path_factory):
    est_cls, art_cls = HEADS[request.param]
    X, y = _dataset()
    with contextlib.redirect_stdout(io.StringIO()):
        model = est_cls().fit(X, y)
        directory = str(tmp_path_factory.mktemp(request.param))
        model.save(directory)
        artifact = art_cls.load(directory)
    return request.param, model, artifact, X, y


def test_oof_predictions_cover_every_row(fitted):
    _, model, _, X, y = fitted
    assert model.oof_preds_.shape == (len(y),)
    assert np.isfinite(model.oof_preds_).all()


def test_oof_skill_is_high_on_a_learnable_signal(fitted):
    _, model, _, _, _ = fitted
    assert model.oof_rho_ > 0.8
    assert model.train_rho_ >= model.oof_rho_ - 0.05


def test_predictions_are_in_target_units(fitted):
    _, model, _, X, y = fitted
    pred = model.predict(X)
    assert abs(pred.mean() - y.mean()) < 0.2 * y.std()


def test_rank_is_bounded_and_monotone_in_prediction(fitted):
    _, model, _, X, _ = fitted
    pred, rank = model.predict(X), model.predict_rank(X)
    assert rank.shape == pred.shape
    assert (rank >= 0).all() and (rank <= 1).all()
    order = np.argsort(pred, kind="stable")
    assert (np.diff(rank[order]) >= -1e-12).all()


def test_onnx_artifact_matches_the_fitted_head(fitted):
    _, model, artifact, X, y = fitted
    np.testing.assert_allclose(
        artifact.run(X), model.predict(X), rtol=0, atol=1e-5 * np.abs(y).max()
    )
    np.testing.assert_allclose(
        artifact.predict_rank(X), model.predict_rank(X), atol=1e-4
    )
    assert artifact.metadata["task"] == "regression"


def test_quantile_splits_partition_and_span_the_range():
    rng = np.random.default_rng(1)
    y = np.sort(rng.normal(size=97))  # sorted input: plain unshuffled KFold would fail
    k, splits = make_quantile_oof_splits(y)
    seen = np.concatenate([val for _, val in splits])
    assert sorted(seen.tolist()) == list(range(len(y)))
    lo, hi = np.quantile(y, [0.2, 0.8])
    for _, val in splits:
        assert (y[val] < lo).any() and (y[val] > hi).any()


@pytest.mark.parametrize(
    "y, message",
    [
        ([1.0] * 20, "variation"),
        ([1.0, np.nan] * 10, "NaN or infinite"),
        ([1.0, 2.0, 3.0], "at least"),
    ],
)
def test_check_trainable_regression_rejects(y, message):
    with pytest.raises(ValueError, match=message):
        check_trainable_regression(y)


def test_oversized_svr_goes_straight_to_the_linear_model(monkeypatch):
    """C does not shrink an SVR's support set, so the guard refits no kernel models."""
    import lazyqsar.base.svc._common as svm_common

    monkeypatch.setattr(svm_common, "_ONNX_MAX_BYTES", 1)
    made = []
    original = BaseSVR._make
    monkeypatch.setattr(
        BaseSVR, "_make", staticmethod(lambda p: made.append(dict(p)) or original(p))
    )
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    y = X[:, 0] + 0.1 * rng.normal(size=200)
    with contextlib.redirect_stdout(io.StringIO()):
        model = BaseSVR(oof=False).fit(X, y)
    assert model.preset_name_.endswith("+linear_guard")
    assert [p.get("use_linear", False) for p in made] == [False, True]
    assert model.params_["C"] == 1.25

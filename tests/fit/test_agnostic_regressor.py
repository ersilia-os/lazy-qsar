"""``LazyRegressor`` (descriptor-agnostic): fit, export, reload, and the guarantees that hold.

The regression twin of ``test_agnostic_classifier``. The contract:
  - predictions are in the target's own units (y is standardised internally only),
  - the ONNX artifact reproduces the fitted model,
  - out-of-fold predictions exist for every row and score a sensible Spearman ρ,
  - the OOF-percentile reliability signal agrees across the export boundary,
  - large datasets batch by size, each batch spanning the full target range.
"""

import contextlib
import io

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")

from lazyqsar.agnostic import LazyRegressor  # noqa: E402
from lazyqsar.assemblers.regressor import _plan_regression_batches  # noqa: E402

N, P = 260, 24


def _data(n=N, p=P, seed=0, offset=500.0, scale=40.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    signal = 2.0 * X[:, 0] - X[:, 1] + np.sin(2 * X[:, 2])
    y = offset + scale * (signal + 0.3 * rng.normal(size=n))
    return X, y


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    X, y = _data()
    model = LazyRegressor()
    _quiet(model.fit, X=X, y=y)
    directory = str(tmp_path_factory.mktemp("reg"))
    _quiet(model.save, directory)
    loaded = _quiet(LazyRegressor.load, directory)
    return model, loaded, X, y, directory


def test_predictions_are_in_target_units(fitted):
    model, _, X, y, _ = fitted
    pred = model.predict(X=X)
    assert pred.shape == (len(y),)
    assert abs(pred.mean() - y.mean()) < 0.1 * y.std()


def test_out_of_fold_skill(fitted):
    model, _, _, _, _ = fitted
    assert model.oof_rho_ > 0.8
    assert model.train_rho_ >= model.oof_rho_


def test_exported_model_matches_the_fitted_model(fitted):
    model, loaded, X, y, _ = fitted
    np.testing.assert_allclose(
        loaded.predict(X), model.predict(X=X), rtol=0, atol=1e-4 * y.std()
    )
    assert loaded.oof_rho_ == pytest.approx(model.oof_rho_)


def test_oof_percentile_agrees_across_export(fitted):
    model, loaded, X, _, _ = fitted
    r_fit = model._oof_percentile(X=X)
    r_onnx = loaded._oof_percentile(X)
    assert r_fit.shape == (X.shape[0],)
    assert ((r_fit >= 0) & (r_fit <= 1)).all()
    np.testing.assert_allclose(r_onnx, r_fit, atol=1e-3)


def test_oof_channels_cover_the_training_rows(fitted):
    model, _, X, y, _ = fitted
    value, rank = model.oof_channels(X=X)
    assert value.shape == rank.shape == y.shape
    assert np.isfinite(value).all()
    # The OOF rank is this value's own ECDF, so it is uniform by construction.
    assert abs(rank.mean() - 0.5) < 0.05


def test_metadata_records_the_task(fitted):
    _, loaded, _, _, _ = fitted
    assert loaded.task == "regression"
    assert loaded.metadata["task"] == "regression"


def test_zip_round_trip(fitted, tmp_path):
    model, loaded, X, _, _ = fitted
    path = _quiet(model.save, str(tmp_path / "m.zip"))
    again = _quiet(LazyRegressor.load, path)
    np.testing.assert_array_equal(again.predict(X), loaded.predict(X))


def test_constant_offset_shifts_predictions():
    """Standardisation makes the fit translation-equivariant: y + c predicts f(x) + c.

    Not bit-exactly. ``(y + c - mean(y + c)) / std`` differs from the unshifted z-scores
    in the last float digits (~1e-15), and tree heads amplify that: a split threshold
    or a quantile-fold boundary lands a hair differently. Refitting on the *same* y is
    deterministic to ~1e-13, so anything beyond a small fraction of the target SD here
    would mean the offset genuinely leaked into the model.
    """
    X, y = _data(n=150, seed=3)
    a, b = LazyRegressor(), LazyRegressor()
    _quiet(a.fit, X=X, y=y)
    _quiet(b.fit, X=X, y=y + 1000.0)
    shift = b.predict(X=X) - a.predict(X=X)
    assert abs(np.median(shift) - 1000.0) < 0.01 * y.std()
    assert np.abs(shift - 1000.0).max() < 0.05 * y.std()


def test_constant_target_is_rejected():
    X, _ = _data(n=50)
    with pytest.raises(ValueError, match="variation"):
        _quiet(LazyRegressor().fit, X=X, y=np.full(50, 3.0))


def test_size_batches_partition_and_span_the_target():
    rng = np.random.default_rng(0)
    y = rng.normal(size=1000)
    batches = _plan_regression_batches(y, max_batch_size=300)
    assert len(batches) == 4
    seen = np.sort(np.concatenate(batches))
    np.testing.assert_array_equal(seen, np.arange(1000))
    q = np.quantile(y, [0.05, 0.95])
    for idx in batches:
        assert y[idx].min() < q[0] and y[idx].max() > q[1]


def test_multi_batch_fit_round_trips(tmp_path):
    X, y = _data(n=240, seed=5)
    model = LazyRegressor(max_batch_size=120)
    _quiet(model.fit, X=X, y=y)
    assert len(model._model.models) == 2
    _quiet(model.save, str(tmp_path))
    loaded = _quiet(LazyRegressor.load, str(tmp_path))
    np.testing.assert_allclose(
        loaded.predict(X), model.predict(X=X), rtol=0, atol=1e-4 * y.std()
    )


def test_feature_level_intervals_cover_and_survive_the_export(tmp_path):
    """Conformal intervals on the pooled OOF residuals: about `level` coverage on fresh
    rows, identical after reload, wider at a higher level."""
    X, y = _data(n=900, seed=3)
    tr, te = np.arange(600), np.arange(600, 900)
    model = LazyRegressor()
    _quiet(model.fit, X=X[tr], y=y[tr])
    lo, hi = model.predict_interval(X=X[te], level=0.9)
    coverage = float(np.mean((y[te] >= lo) & (y[te] <= hi)))
    assert 0.8 <= coverage <= 0.98, coverage
    loaded = _quiet(LazyRegressor.load, _quiet(model.save, str(tmp_path / "m")))
    la, ha = loaded.predict_interval(X[te], level=0.9)
    scale = 1e-6 * float(np.std(y))
    np.testing.assert_allclose(np.c_[la, ha], np.c_[lo, hi], rtol=0, atol=scale)
    lo5, hi5 = model.predict_interval(X=X[te], level=0.5)
    assert np.all(hi - lo >= hi5 - lo5)

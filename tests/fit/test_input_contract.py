"""Feature values a float32 ONNX graph cannot represent are refused, not silently changed.

NaN is accepted everywhere -- the exported preprocessor imputes it exactly as at fit
time. ``inf`` is not imputed, and a finite float64 beyond the float32 range becomes
``inf`` in the cast; either would produce a prediction for a value the model never saw.
"""

import contextlib
import io

import numpy as np
import pytest

pytest.importorskip("sklearn")

from lazyqsar.agnostic import LazyClassifier, LazyRegressor  # noqa: E402


def _data(n=120, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    return X, (X[:, 0] > 0).astype(int), X[:, 0] * 2.0


@pytest.mark.parametrize("bad", [np.inf, -np.inf, 1e39])
@pytest.mark.parametrize("cls", [LazyClassifier, LazyRegressor])
def test_fit_refuses_unrepresentable_values(cls, bad):
    X, yc, yr = _data()
    X[3, 2] = bad
    y = yc if cls is LazyClassifier else yr
    with pytest.raises(ValueError, match="float32"):
        cls().fit(X=X, y=y)


def test_predict_refuses_unrepresentable_values_and_accepts_nan():
    X, _, yr = _data()
    model = LazyRegressor()
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X=X, y=yr)
    Q = X[:10].copy()
    Q[0, 0] = np.nan
    assert np.isfinite(model.predict(X=Q)).all()
    Q[1, 1] = np.inf
    with pytest.raises(ValueError, match="1 infinite value"):
        model.predict(X=Q)


def test_loaded_models_refuse_infinite_features_too(tmp_path):
    """The fitted entry points refuse inf; a loaded checkpoint must not score it silently."""
    import contextlib
    import io

    from lazyqsar.agnostic import LazyRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6)).astype(np.float32)
    model = LazyRegressor()
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(X=X, y=X[:, 0] + rng.normal(size=120))
        loaded = LazyRegressor.load(model.save(str(tmp_path / "m")))
    bad = X[:3].astype(np.float64)
    bad[0, 2] = np.inf
    bad[1, 1] = 1e300  # beyond float32: becomes inf in the cast
    with pytest.raises(ValueError, match="infinite or beyond-float32"):
        loaded.predict(bad)
    with_nan = X[:3].copy()
    with_nan[0, 0] = np.nan  # NaN is imputed, as at fit time
    assert np.isfinite(loaded.predict(with_nan)).all()

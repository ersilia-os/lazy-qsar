"""Inputs that used to break a fit or a flag: one varying feature, the offline switch."""

import contextlib
import io

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")

from lazyqsar.agnostic import LazyClassifier, LazyRegressor  # noqa: E402


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_one_varying_feature_fits_and_round_trips(task, tmp_path):
    """skl2onnx cannot export a VarianceThreshold over a single column; the fit used to
    die binding the linear head to its graph."""
    rng = np.random.default_rng(0)
    n = 300
    X = np.zeros((n, 10), dtype=np.float32)
    X[:, 3] = rng.normal(size=n)
    y = 2.0 * X[:, 3] + 0.1 * rng.normal(size=n)
    if task == "classification":
        model, target = LazyClassifier(), (y > 0).astype(int)
    else:
        model, target = LazyRegressor(), y
    _quiet(model.fit, X=X, y=target)
    loaded = _quiet(type(model).load, _quiet(model.save, str(tmp_path / "m")))
    if task == "classification":
        got, want = loaded.predict_proba(X)[:, 1], model.predict_proba(X=X)[:, 1]
    else:
        got, want = loaded.predict(X), model.predict(X=X)
    np.testing.assert_allclose(got, want, atol=1e-5)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", False),
        ("0", False),
        ("false", False),
        ("No", False),
        ("1", True),
        ("yes", True),
    ],
)
def test_offline_flag_reads_falsy_strings_as_off(monkeypatch, value, expected):
    from lazyqsar.reference.download import offline

    monkeypatch.setenv("LAZYQSAR_REFERENCE_OFFLINE", value)
    assert offline() is expected

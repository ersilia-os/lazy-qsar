"""Power-scaled columns that a float32 rounding step can move are left untransformed.

Yeo-Johnson with a strongly negative lambda (RDKit's ``BCUT2D_*`` columns reach -5) maps
a whole column onto almost one value, and standardizing then divides by that tiny spread.
One float32 step of the transformed value became 0.01-0.07 standard deviations, and numpy's
``log1p``/``expm1`` round some values differently on AVX-512 CPUs, so the fitted pipeline
disagreed with its own ONNX graph on Intel but not on AMD (ersilia-os/lazy-qsar#66).

``BasePreprocessor._relax_fragile_power_columns`` gives such a column lambda = 1 (the
identity, only standardized). The CPU difference itself cannot be reproduced on a single
machine, so these tests pin the invariant that makes it harmless: after fitting, no column
moves by more than ``_FLOAT32_STEP_LIMIT`` standard deviations per float32 step.
"""

import contextlib
import io

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")
pytest.importorskip("onnxruntime")

from sklearn.preprocessing import PowerTransformer  # noqa: E402

from lazyqsar.base.preprocessing import pipeline as pp  # noqa: E402

N, P_SKEWED = 400, 19
FRAGILE = 0


def _bcut_like(rng, n):
    """70% tightly clustered just above 10, 30% spread far above: lambda ~ -3.7."""
    tight = 10.5 + rng.normal(0, 0.05, n).clip(-0.5, 0.5)
    wide = 20 + rng.gamma(2, 4, n)
    return np.where(rng.random(n) < 0.7, tight, wide)


def _matrix(with_fragile: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    cols = [rng.gamma(1.5, 2.0, N) for _ in range(P_SKEWED)]
    first = _bcut_like(rng, N) if with_fragile else rng.gamma(1.5, 2.0, N)
    X = np.column_stack([first, *cols])
    y = (rng.random(N) < 0.4).astype(int)
    return X, y


def _amplification(power_transformer) -> np.ndarray:
    """Standard deviations moved by one float32 step of each transformed column."""
    sc = power_transformer._scaler
    step = np.spacing(np.abs(sc.mean_).astype(np.float32)).astype(np.float64)
    return step / sc.scale_


def _fit(X, y, monkeypatch, guard=True):
    monkeypatch.setattr(pp, "select_scaler", lambda profile: "power")
    if not guard:
        monkeypatch.setattr(
            pp.BasePreprocessor, "_relax_fragile_power_columns", lambda *a: None
        )
    with contextlib.redirect_stdout(io.StringIO()):
        return pp.BasePreprocessor(task="classification").fit(X, y)


def test_the_synthetic_column_is_fragile_without_the_guard():
    X, _ = _matrix(with_fragile=True)
    plain = PowerTransformer(method="yeo-johnson").fit(X)
    amp = _amplification(plain)
    assert plain.lambdas_[FRAGILE] < -2
    assert amp[FRAGILE] > pp._FLOAT32_STEP_LIMIT
    assert np.all(np.delete(amp, FRAGILE) <= pp._FLOAT32_STEP_LIMIT)


def test_only_the_fragile_column_keeps_lambda_one(monkeypatch):
    X, y = _matrix(with_fragile=True)
    prep = _fit(X, y, monkeypatch)
    pt = prep.pipeline_.named_steps["scaler"]
    plain = PowerTransformer(method="yeo-johnson").fit(X)
    assert prep.scaler_name_ == "power"
    assert pt.lambdas_[FRAGILE] == 1.0
    np.testing.assert_array_equal(
        np.delete(pt.lambdas_, FRAGILE), np.delete(plain.lambdas_, FRAGILE)
    )
    # lambda = 1 is the identity, so the column is only standardized.
    z = prep.pipeline_[:3].transform(X)[:, FRAGILE]
    np.testing.assert_allclose(
        z, (X[:, FRAGILE] - X[:, FRAGILE].mean()) / X[:, FRAGILE].std()
    )


def test_no_column_amplifies_a_rounding_step_past_the_limit(monkeypatch):
    X, y = _matrix(with_fragile=True)
    prep = _fit(X, y, monkeypatch)
    amp = _amplification(prep.pipeline_.named_steps["scaler"])
    assert np.all(amp <= pp._FLOAT32_STEP_LIMIT)
    assert prep._onnx_session_ is not None


def test_a_fit_without_fragile_columns_is_unchanged(monkeypatch):
    X, y = _matrix(with_fragile=False)
    guarded = _fit(X, y, monkeypatch)
    plain = _fit(X, y, monkeypatch, guard=False)
    np.testing.assert_array_equal(
        guarded.pipeline_.named_steps["scaler"].lambdas_,
        plain.pipeline_.named_steps["scaler"].lambdas_,
    )
    assert guarded.kept_feature_indices_ == plain.kept_feature_indices_
    assert _graph_without_name(guarded) == _graph_without_name(plain)


def _graph_without_name(prep) -> bytes:
    """The bound ONNX graph minus its name, which skl2onnx draws at random per export."""
    import onnx

    model = onnx.load_from_string(prep._onnx_bytes_)
    model.graph.name = ""
    return model.SerializeToString()

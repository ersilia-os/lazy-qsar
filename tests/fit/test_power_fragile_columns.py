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


def _mrhi_like(rng, n):
    """The column traced in review of #67: ``BCUT2D_MRHI`` on the ``all`` deck, raw range
    4.7-14.1, lambda ~ -5.2, transformed to ~0.192 with a spread of a few 1e-6.

    Here: lambda -5.22, transformed mean 0.1914, and one float32 step is 2.8e-3 SD.
    """
    bulk = 4.7 + rng.gamma(2, 0.37, n)
    tail = rng.uniform(6, 14.1, n)
    x = np.where(rng.random(n) < 0.85, bulk, tail).clip(4.7, 14.1)
    x[:2] = 4.7, 14.1
    return x


def _matrix(first=None, seed: int = 0):
    """Column 0 drawn by *first*, or one more skewed column like the rest when None."""
    rng = np.random.default_rng(seed)
    cols = [rng.gamma(1.5, 2.0, N) for _ in range(P_SKEWED)]
    col0 = first(rng, N) if first else rng.gamma(1.5, 2.0, N)
    X = np.column_stack([col0, *cols])
    y = (rng.random(N) < 0.4).astype(int)
    return X, y


def _round_like_avx512(monkeypatch):
    """Make scikit-learn's Yeo-Johnson land one float32 step up, as numpy's AVX-512
    ``log1p``/``expm1`` did on the Intel nodes. The ONNX graph is left alone, so the bind
    check then compares the two sides the way it did on the failing machines."""
    from scipy import stats

    real = stats.yeojohnson

    def one_step_up(x, lmbda=None, *args, **kwargs):
        out = real(x, lmbda, *args, **kwargs)
        return np.nextafter(out, np.asarray(np.inf, dtype=out.dtype))

    monkeypatch.setattr(stats, "yeojohnson", one_step_up)


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
    X, _ = _matrix(_bcut_like)
    plain = PowerTransformer(method="yeo-johnson").fit(X)
    amp = _amplification(plain)
    assert plain.lambdas_[FRAGILE] < -2
    assert amp[FRAGILE] > pp._FLOAT32_STEP_LIMIT
    assert np.all(np.delete(amp, FRAGILE) <= pp._FLOAT32_STEP_LIMIT)


def test_only_the_fragile_column_keeps_lambda_one(monkeypatch):
    X, y = _matrix(_bcut_like)
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
    X, y = _matrix(_bcut_like)
    prep = _fit(X, y, monkeypatch)
    amp = _amplification(prep.pipeline_.named_steps["scaler"])
    assert np.all(amp <= pp._FLOAT32_STEP_LIMIT)
    assert prep._onnx_session_ is not None


def test_a_fit_without_fragile_columns_is_unchanged(monkeypatch):
    X, y = _matrix()
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


def test_the_column_from_the_review_is_real_shaped():
    X, _ = _matrix(_mrhi_like)
    assert X[:, FRAGILE].min() == 4.7 and X[:, FRAGILE].max() == 14.1
    plain = PowerTransformer(method="yeo-johnson").fit(X)
    assert -5.5 < plain.lambdas_[FRAGILE] < -5.0
    assert _amplification(plain)[FRAGILE] > 10 * pp._FLOAT32_STEP_LIMIT


def test_the_column_from_the_review_fails_the_bind_check_without_the_guard(monkeypatch):
    """The #66 failure through the real bind check. Whether the fit's own bind already
    fails depends on how this CPU rounds; shifting scikit-learn's side by one float32
    step afterwards makes the failure certain on any CPU."""
    from lazyqsar.utils.onnx_check import OnnxExportMismatch

    X, y = _matrix(_mrhi_like)
    with pytest.raises(OnnxExportMismatch):
        prep = _fit(X, y, monkeypatch, guard=False)
        _round_like_avx512(monkeypatch)
        prep._bind_onnx_runtime(X)


def test_the_column_from_the_review_passes_the_bind_check_with_the_guard(monkeypatch):
    X, y = _matrix(_mrhi_like)
    prep = _fit(X, y, monkeypatch)
    assert prep.pipeline_.named_steps["scaler"].lambdas_[FRAGILE] == 1.0
    _round_like_avx512(monkeypatch)
    prep._bind_onnx_runtime(X)
    assert prep._onnx_session_ is not None

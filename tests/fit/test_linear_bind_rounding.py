"""The linear classifier's bind check allows for float32 rounding that grows with the model.

The graph computes the logit z = w.x + b in float32, and the rounding error of that sum is
proportional to sum_j |w_j x_j| + |b|, not to z. A weakly regularised fit on nearly
separable data -- the large regime's SGDClassifier with a small alpha -- reached |w| ~ 700
and b ~ -7,600 on CLAMP features of a ~4,300-compound dataset; on a probe row near the
boundary the correct float32 graph was 8e-4 (one CPU) and 9e-4 (another) away from the
float64 model, the fit failed with OnnxExportMismatch, and which fold of which dataset
failed depended on the CPU. These tests inflate an ordinary fit to the same scale.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")
pytest.importorskip("onnxruntime")

from lazyqsar.base.linear import BaseLinearClassifier  # noqa: E402
from lazyqsar.utils.onnx_check import OnnxExportMismatch  # noqa: E402

N, P = 400, 300
SCALE = 5_000.0


def _fit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, P)).astype(np.float32)
    y = (X[:, :5].sum(axis=1) + 0.5 * rng.normal(size=N) > 0).astype(int)
    return BaseLinearClassifier(regime="standard", calibrated=False).fit(X, y)


def _inflate(model, z0=0.0):
    """Dense weights of order SCALE, and one probe row put at logit z0 (the boundary).

    That row's logit is then b plus terms in the hundreds of thousands cancelling to z0,
    which is what the over-fitted SGD heads produced (|w| ~ 700, b ~ -7,600, the terms of
    a near-boundary row summing to ~1e5). The graph is unchanged in structure.
    """
    est = model._estimator
    rng = np.random.default_rng(1)
    est.coef_ = rng.normal(scale=SCALE, size=est.coef_.shape)
    z = model._transform(model._onnx_probe_.astype(np.float64)) @ est.coef_.ravel()
    est.intercept_ = np.array([z0 - float(z[0])])
    return model


def _magnitude(model, probe):
    """S = sum_j |w_j x_j| + |b| per probe row, computed here independently."""
    est = model._estimator
    Xt = np.abs(model._transform(probe.astype(np.float64)))
    return Xt @ np.abs(est.coef_.ravel()) + abs(float(est.intercept_[0]))


def test_inflated_model_is_off_by_more_than_the_fixed_tolerance():
    """Guard for the test itself: the inflated model really shows the rounding gap."""
    import onnxruntime as rt

    model = _inflate(_fit())
    probe = model._onnx_probe_
    native = model._raw_native(probe)

    session = rt.InferenceSession(
        model._onnx_bytes(), providers=["CPUExecutionProvider"]
    )
    out = session.run(None, {session.get_inputs()[0].name: probe.astype(np.float32)})
    p_onnx = np.asarray(out[1], dtype=np.float64)[:, 1]
    # The fixed check's own criterion (np.allclose with atol = rtol = the tolerance).
    tol = model._onnx_tolerance
    assert np.any(np.abs(p_onnx - native) > tol + tol * np.abs(native))


def test_inflated_model_binds():
    model = _inflate(_fit())
    model._bind_onnx()  # raised OnnxExportMismatch before the rounding bound


def test_rounding_bound_is_negligible_for_an_ordinary_fit():
    model = _fit()
    probe = model._onnx_probe_
    bound = model._rounding_atol(probe, model._raw_native(probe))
    assert bound.max() < 0.1 * model._onnx_tolerance


def test_wrong_graph_is_still_refused(monkeypatch):
    """A graph that lost the intercept is off by tenths and must still fail to bind."""
    model = _inflate(_fit())
    real_bytes = BaseLinearClassifier._onnx_bytes

    def without_intercept(self):
        saved = self._estimator.intercept_.copy()
        self._estimator.intercept_ = np.zeros_like(saved)
        try:
            return real_bytes(self)
        finally:
            self._estimator.intercept_ = saved

    monkeypatch.setattr(BaseLinearClassifier, "_onnx_bytes", without_intercept)
    with pytest.raises(OnnxExportMismatch, match="does not reproduce"):
        model._bind_onnx()


@pytest.mark.parametrize("z0", [0.0, 2.0])
def test_allowed_logit_error_is_exactly_the_factor_times_u_s(z0):
    """Pins the bound's form: the allowed move, mapped back to the logit, is k u S.

    Mapped towards the boundary, where the sigmoid is steepest. Dropping the
    p(1 - p) factor, or using the slope at p instead of the exact sigmoid (which
    understates the move at z0 = 2), would fail this, whatever the CPU's rounding.
    """
    model = _inflate(_fit(), z0)
    probe = model._onnx_probe_
    native = model._raw_native(probe)
    bound = model._rounding_atol(probe, native)
    p = native[0]
    toward = p - bound[0] if z0 > 0 else p + bound[0]
    allowed_dz = abs(np.log(p / (1 - p)) - np.log(toward / (1 - toward)))
    u = float(np.finfo(np.float32).eps) / 2
    expected = model._onnx_rounding_factor * u * _magnitude(model, probe)[0]
    assert allowed_dz == pytest.approx(expected, rel=1e-6)


def test_rounding_factor_stays_in_the_measured_range():
    """The factor covers the worst error measured, without a wide excess.

    Measured max |z32 - z64| / (u S): 0.63 to 2.1 on the three failing fits (#71), and
    up to 2.5 on ORT LinearClassifier graphs with 300 to 30,000 features (review of
    #72). A factor much above 16 would let wrong graphs through on large-S rows.
    """
    assert 2.5 < BaseLinearClassifier._onnx_rounding_factor <= 32

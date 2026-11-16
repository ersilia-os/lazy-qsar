"""Power-scaled columns whose lambda sits at 0 or 2 are left untransformed.

scikit-learn's Yeo-Johnson switches to ``log1p`` (float64) when lambda is 0, and to
``-log1p(-x)`` for negative inputs when lambda is 2. skl2onnx's float32 graph instead
computes ``((x + 1)**lambda - 1) / lambda`` (and ``(1 - (1 - x)**(2 - lambda)) / (2 - lambda)``),
which cancels catastrophically as lambda nears the special value: on an RDKit column of a
*P. falciparum* ChEMBL dataset, lambda = -9.8e-7 put the graph 1.4% below scikit-learn
and failed the bind check on every CPU. At exactly 0 (or 2) the graph's dedicated branch
computes 0 / 0, which its NaN clean-up turns into 0 for every row.

``BasePreprocessor._relax_fragile_power_columns`` gives such a column lambda = 1 (the
identity, only standardized). These tests force a lambda onto one column of an otherwise
ordinary fit and check the bind with and without the guard.
"""

import contextlib
import io

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")
pytest.importorskip("onnxruntime")

from scipy import stats  # noqa: E402

from lazyqsar.base.preprocessing import pipeline as pp  # noqa: E402
from lazyqsar.utils.onnx_check import OnnxExportMismatch  # noqa: E402

N, P_SKEWED = 400, 19
EDGE = 0


def _matrix(negative: bool = False, seed: int = 0):
    """Skewed gamma columns; column 0 spans 0-25 (or -25-0 when *negative*)."""
    rng = np.random.default_rng(seed)
    cols = [rng.gamma(1.5, 2.0, N) for _ in range(P_SKEWED)]
    col0 = rng.gamma(1.2, 4.0, N).clip(0, 25)
    X = np.column_stack([-col0 if negative else col0, *cols])
    y = (rng.random(N) < 0.4).astype(int)
    return X, y


def _fit(X, y, monkeypatch, lam=None, guard=True):
    """Fit a power-scaled preprocessor, forcing *lam* onto column 0 before the guard runs."""
    monkeypatch.setattr(pp, "select_scaler", lambda profile: "power")
    real_guard = pp.BasePreprocessor._relax_fragile_power_columns

    def force_then_guard(self, X, y):
        if lam is not None:
            steps = self.pipeline_.named_steps
            pt, sc = steps["scaler"], steps["scaler"]._scaler
            pt.lambdas_[EDGE] = lam
            # Re-standardize column 0 on scikit-learn's own transform at the forced lambda.
            Xv = np.asarray(self.pipeline_[:2].transform(X), dtype=np.float64)
            t = stats.yeojohnson(Xv[:, EDGE], lmbda=lam)  # what scikit-learn calls
            sc.mean_[EDGE], sc.var_[EDGE] = t.mean(), t.var()
            sc.scale_[EDGE] = np.sqrt(sc.var_[EDGE])
            steps["reducer"].fit(self.pipeline_[:3].transform(X), y)
        if guard:
            real_guard(self, X, y)

    monkeypatch.setattr(
        pp.BasePreprocessor, "_relax_fragile_power_columns", force_then_guard
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return pp.BasePreprocessor(task="classification").fit(X, y)


FAILING = [
    pytest.param(-9.8e-7, False, id="lambda=-9.8e-7 (the P. falciparum column)"),
    pytest.param(0.0, False, id="lambda=0 exactly"),
    pytest.param(2.0 - 1e-6, True, id="lambda=2-1e-6, negative inputs"),
    pytest.param(2.0, True, id="lambda=2 exactly, negative inputs"),
]
# Inside the edge but still passing the bind check unguarded on this data: the graph's
# error is about 6e-8 / lambda, so the 1e-3 edge keeps a margin below the 1e-4 tolerance.
MARGIN = [pytest.param(4e-4, False, id="lambda=+4e-4 (margin)")]


@pytest.mark.parametrize("lam, negative", FAILING)
def test_an_edge_lambda_fails_the_bind_check_without_the_guard(monkeypatch, lam, negative):
    X, y = _matrix(negative)
    with pytest.raises(OnnxExportMismatch):
        _fit(X, y, monkeypatch, lam=lam, guard=False)


@pytest.mark.parametrize("lam, negative", FAILING + MARGIN)
def test_the_guard_leaves_an_edge_lambda_column_untransformed(monkeypatch, lam, negative):
    X, y = _matrix(negative)
    prep = _fit(X, y, monkeypatch, lam=lam)
    pt = prep.pipeline_.named_steps["scaler"]
    assert pt.lambdas_[EDGE] == 1.0
    assert prep._onnx_session_ is not None
    # lambda = 1 is the identity, so the column is only standardized.
    z = prep.pipeline_[:3].transform(X)[:, EDGE]
    np.testing.assert_allclose(z, (X[:, EDGE] - X[:, EDGE].mean()) / X[:, EDGE].std())


@pytest.mark.parametrize("lam", [-0.05, 0.02, 0.5, 1.9])
def test_a_lambda_away_from_the_edges_is_kept(monkeypatch, lam):
    X, y = _matrix()
    prep = _fit(X, y, monkeypatch, lam=lam)
    assert prep.pipeline_.named_steps["scaler"].lambdas_[EDGE] == pytest.approx(lam)
    assert prep._onnx_session_ is not None

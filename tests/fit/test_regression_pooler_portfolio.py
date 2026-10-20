"""Regression gating oracle and head selection.

The pooler's oracle is a per-sample Gaussian likelihood of the target under each head's
out-of-fold prediction; the gate should learn to favour, *per region of feature space*,
the head that is accurate there. The portfolio reuses the classification rules and must
emit regression head names.
"""

import contextlib
import io

import numpy as np
import pytest

pytest.importorskip("sklearn")

from lazyqsar.poolers.regression.inner_pooler import (  # noqa: E402
    InnerRegressorPooler,
    _robust_sigma,
)
from lazyqsar.portfolios.regression import Portfolio  # noqa: E402


def test_robust_sigma_ignores_a_single_outlier():
    r = np.random.default_rng(0).normal(size=(2001, 1))
    r[0, 0] = 1e6  # would blow a standard deviation up to ~2e4
    assert _robust_sigma(r)[0] == pytest.approx(1.0, rel=0.1)


def test_gate_prefers_the_locally_accurate_head():
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 3))
    y = rng.normal(size=n)
    left = X[:, 0] < 0
    noise = rng.normal(scale=1.0, size=n)
    # Head 0 is exact where x0 < 0 and noisy elsewhere; head 1 the opposite.
    P = np.column_stack([np.where(left, y, y + noise), np.where(left, y + noise, y)])
    pooler = InnerRegressorPooler(["a", "b"])
    with contextlib.redirect_stdout(io.StringIO()):
        pooler.fit(P, y, X_prep=X)
    W = pooler.get_weights(X)
    assert W[left, 0].mean() > 0.6
    assert W[~left, 1].mean() > 0.6
    pooled = pooler.predict(P, X)
    assert np.mean((pooled - y) ** 2) < min(np.mean((P - y[:, None]) ** 2, axis=0))


def test_single_head_passes_through():
    P = np.arange(5.0)[:, None]
    pooler = InnerRegressorPooler(["xgb"])
    with contextlib.redirect_stdout(io.StringIO()):
        pooler.fit(P, P[:, 0], X_prep=np.ones((5, 2)))
    np.testing.assert_array_equal(pooler.predict(P, np.ones((5, 2))), P[:, 0])


@pytest.mark.parametrize("n, p", [(150, 40), (1200, 30), (8000, 20)])
def test_portfolio_uses_regression_head_names(n, p):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, p))
    y = X[:, 0] + rng.normal(size=n)
    pf = Portfolio()
    with contextlib.redirect_stdout(io.StringIO()):
        pf.fit(X, y)
    heads = pf.get()
    assert "svc" not in heads
    assert {"xgb", "rf"} <= set(heads) <= {"lr", "xgb", "rf", "svr"}
    if n > 5000:
        assert heads == ["xgb", "rf"]

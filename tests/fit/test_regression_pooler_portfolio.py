"""Regression stacking pooler and head selection.

The pooler blends the heads with fixed non-negative weights fitted by least squares on
their out-of-fold predictions; the selector picks heads by dataset size. Both are
documented with the benchmark evidence in their modules.
"""

import contextlib
import io
import json

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from lazyqsar.poolers.regression.inner_pooler import (  # noqa: E402
    InnerRegressorPooler,
    InnerRegressorPoolerArtifact,
    stacking_weights,
)
from lazyqsar.portfolios.regression import Portfolio  # noqa: E402


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def test_stacking_weights_favour_the_better_head_and_sum_to_one():
    rng = np.random.default_rng(0)
    y = rng.normal(size=2000)
    P = np.column_stack(
        [y + 0.2 * rng.normal(size=2000), y + 1.0 * rng.normal(size=2000)]
    )
    w = stacking_weights(P, y)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    assert w[0] > 0.8


def test_stack_beats_each_head_and_equal_weights():
    rng = np.random.default_rng(1)
    n = 3000
    y = rng.normal(size=n)
    P = np.column_stack([y + s * rng.normal(size=n) for s in (0.3, 0.6, 1.2)])
    pooler = InnerRegressorPooler(["a", "b", "c"])
    _quiet(pooler.fit, P, y, X_prep=np.zeros((n, 2)))
    mse = np.mean((pooler.predict(P) - y) ** 2)
    assert mse < min(np.mean((P - y[:, None]) ** 2, axis=0))
    assert mse < np.mean((P.mean(axis=1) - y) ** 2)


def test_useless_heads_get_no_weight_and_all_zero_falls_back_to_equal():
    y = np.linspace(-1, 1, 50)
    P = np.column_stack([y, -y])  # the second head is anti-correlated
    assert stacking_weights(P, y)[1] == pytest.approx(0.0)
    assert stacking_weights(-P[:, :1], y).tolist() == [1.0]


def test_single_head_passes_through():
    P = np.arange(5.0)[:, None]
    pooler = InnerRegressorPooler(["xgb"])
    _quiet(pooler.fit, P, P[:, 0], X_prep=np.ones((5, 2)))
    np.testing.assert_array_equal(pooler.predict(P, np.ones((5, 2))), P[:, 0])


def test_weights_round_trip_through_the_artifact(tmp_path):
    rng = np.random.default_rng(2)
    y = rng.normal(size=400)
    P = np.column_stack([y + rng.normal(size=400), y + 0.5 * rng.normal(size=400)])
    pooler = InnerRegressorPooler(["xgb", "svr"])
    _quiet(pooler.fit, P, y, X_prep=np.zeros((400, 3)))
    pooler.save(str(tmp_path))
    data = json.load(open(tmp_path / "pooler.json"))
    assert data["task"] == "regression" and len(data["head_weights"]) == 2
    art = InnerRegressorPoolerArtifact.load(str(tmp_path))
    np.testing.assert_allclose(art.predict(P, np.zeros((400, 3))), pooler.predict(P))
    np.testing.assert_allclose(
        art.get_weights(np.zeros((3, 3))), np.tile(pooler._head_weights, (3, 1))
    )


@pytest.mark.parametrize(
    "n, expected",
    [(150, ["lr", "xgb", "svr"]), (2000, ["lr", "xgb", "svr"]), (3000, ["lr", "xgb"]),
     (8000, ["xgb", "rf"])],
)
def test_portfolio_picks_heads_by_size(n, expected):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, 20))
    y = X[:, 0] + rng.normal(size=n)
    pf = Portfolio()
    _quiet(pf.fit, X, y)
    assert pf.get() == expected

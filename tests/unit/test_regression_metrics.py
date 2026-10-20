"""Spearman ρ and the regression metric bundle -- the skill signal for regression.

Spearman replaces AUC everywhere a regression model is judged: head gating, descriptor
weighting, descriptor pruning. It is implemented in numpy (ties share their average rank)
so it can live in the base tier, and it returns ``None`` -- never NaN -- when the
correlation is undefined, so callers can route "unknown skill" explicitly.
"""

import numpy as np
import pytest

from lazyqsar.utils.metrics import (
    _average_ranks,
    regression_metrics,
    skill_score,
    spearman_score,
)


def test_average_ranks_share_ties():
    np.testing.assert_array_equal(_average_ranks([10, 20, 20, 30]), [1, 2.5, 2.5, 4])


def test_perfect_monotone_relationship_is_one():
    x = np.arange(20, dtype=float)
    assert spearman_score(x, np.exp(x / 5)) == pytest.approx(1.0)
    assert spearman_score(x, -x) == pytest.approx(-1.0)


def test_matches_pearson_on_average_ranks_with_ties():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 5, 60).astype(float)
    b = a + rng.normal(size=60)
    ra, rb = _average_ranks(a), _average_ranks(b)
    expected = np.corrcoef(ra, rb)[0, 1]
    assert spearman_score(a, b) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        ([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]),  # constant predictions
        ([4.0, 4.0, 4.0], [1.0, 2.0, 3.0]),  # constant target
        ([1.0, 2.0], [1.0, 2.0]),  # too few pairs
        ([1.0, np.nan, 3.0, np.inf], [1.0, 2.0, 3.0, 4.0]),  # too few finite pairs
    ],
)
def test_undefined_correlation_is_none(y_true, y_pred):
    assert spearman_score(y_true, y_pred) is None


def test_regression_metrics_values():
    m = regression_metrics([1, 2, 3, 4], [1, 2, 3, 5])
    assert m["spearman"] == pytest.approx(1.0)
    assert m["r2"] == pytest.approx(0.8)
    assert m["rmse"] == pytest.approx(0.5)
    assert m["mae"] == pytest.approx(0.25)


def test_skill_score_regression_floors_at_zero_and_handles_none():
    x = np.arange(10, dtype=float)
    assert skill_score("regression", x, -x) == 0.0
    assert skill_score("regression", x, np.ones(10)) == 0.0
    assert skill_score("regression", x, x) == pytest.approx(1.0)

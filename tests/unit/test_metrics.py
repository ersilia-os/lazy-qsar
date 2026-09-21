"""BEDROC -- early-recognition scoring for imbalanced screens.

This is the metric the portfolio and the descriptor weighting lean on when actives are rare,
and it is the one most likely to be subtly wrong: the normalisation divides by a theoretical
RIE range that depends on prevalence, so an error shows up as a plausible-looking number
rather than an exception.

The tests pin the two ends of that normalisation (a perfect ranking is 1, a reversed one is
0) and the expected value under random ranking. Numpy only -- ``aupr_score`` and
``composite_metrics`` reach for sklearn and are tested in ``tests/fit``.
"""

import numpy as np
import pytest

from lazyqsar.utils.metrics import bedroc_random_baseline, bedroc_score


def _labels(n=200, n_actives=20):
    y = np.zeros(n, dtype=int)
    y[:n_actives] = 1
    return y


def test_perfect_ranking_scores_one():
    y = _labels()
    scores = y.astype(float)  # every active ranked above every inactive
    assert bedroc_score(y, scores) == pytest.approx(1.0, abs=1e-9)


def test_reversed_ranking_scores_zero():
    y = _labels()
    scores = 1.0 - y.astype(float)
    assert bedroc_score(y, scores) == pytest.approx(0.0, abs=1e-9)


def test_random_ranking_lands_near_the_random_baseline():
    y = _labels()
    rng = np.random.default_rng(0)
    observed = np.mean([bedroc_score(y, rng.random(len(y))) for _ in range(200)])
    assert observed == pytest.approx(bedroc_random_baseline(y), abs=0.05)


def test_alpha_controls_how_much_the_top_counts():
    """A higher alpha weights the head of the ranking more, which is the whole point."""
    y = _labels(n=200, n_actives=20)
    scores = np.zeros(200)
    scores[:20] = 1.0  # all actives in the top 10%
    assert bedroc_score(y, scores, alpha=20.0) > bedroc_score(y, scores, alpha=1.0)


@pytest.mark.parametrize(
    "y",
    [np.zeros(50, dtype=int), np.ones(50, dtype=int)],
    ids=["no_actives", "all_actives"],
)
def test_degenerate_label_vectors_return_a_finite_number(y):
    """A single-class fold must not raise -- it happens on small imbalanced datasets."""
    rng = np.random.default_rng(1)
    value = bedroc_score(y, rng.random(len(y)))
    assert np.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_bounded_for_arbitrary_scores():
    y = _labels(n=300, n_actives=15)
    rng = np.random.default_rng(2)
    for _ in range(50):
        value = bedroc_score(y, rng.normal(size=len(y)))
        assert 0.0 <= value <= 1.0

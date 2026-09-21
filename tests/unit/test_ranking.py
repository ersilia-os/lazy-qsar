"""Unit tests for lazyqsar.utils.ranking (numpy-only; runs on the base install)."""

import numpy as np
import pytest

from lazyqsar.utils.ranking import rank_from_reference, prepare_knots, rank_from_knots


def _plateau_knots():
    """9000 copies of 0.001 followed by 1000 rising values."""
    return np.concatenate([np.full(9000, 0.001), np.linspace(0.001, 1.0, 1000)])


def test_tie_plateau_returns_midrank_not_top_edge():
    # Regression test for the np.interp last-duplicate bug: a score sitting inside a tie
    # plateau used to be awarded the rank of the plateau's TOP edge (0.9001).
    assert rank_from_knots(0.001, _plateau_knots()) == pytest.approx(0.45, abs=0.01)


def test_endpoints_match_plain_ecdf():
    knots = _plateau_knots()
    assert rank_from_knots(0.0005, knots) == 0.0  # below every knot
    assert rank_from_knots(5.0, knots) == 1.0  # above every knot


def test_monotone_and_bounded():
    rng = np.random.default_rng(0)
    knots = rng.normal(size=5000)
    scores = np.sort(rng.normal(size=2000))
    ranks = rank_from_knots(scores, knots)
    assert np.all(np.diff(ranks) >= -1e-12)
    assert ranks.min() >= 0.0 and ranks.max() <= 1.0


def test_no_ties_matches_legacy_formula():
    # The three continuous-score heads (xgb, lr, svc) must not shift materially.
    rng = np.random.default_rng(1)
    knots = rng.normal(size=4000)
    scores = rng.normal(size=1000)
    legacy = np.interp(scores, np.sort(knots), np.linspace(0.0, 1.0, len(knots)))
    assert np.abs(rank_from_knots(scores, knots) - legacy).max() < 1e-3


def test_degenerate_single_value_knots():
    ranks = rank_from_knots([0.1, 0.3, 0.9], np.full(10, 0.3))
    assert list(ranks) == [0.0, 0.5, 1.0]


def test_prepared_matches_on_the_fly():
    rng = np.random.default_rng(2)
    knots = rng.normal(size=500)
    scores = rng.normal(size=100)
    prepared = prepare_knots(knots)
    assert np.array_equal(
        rank_from_knots(scores, knots), rank_from_knots(scores, prepared=prepared)
    )


def test_empty_knots_raises():
    with pytest.raises(ValueError):
        prepare_knots([])


def test_self_ranking_is_approximately_uniform():
    rng = np.random.default_rng(3)
    sample = rng.normal(size=20000)
    ranks = rank_from_knots(sample, sample)
    # mean of a uniform is 0.5; deciles should be evenly populated
    assert ranks.mean() == pytest.approx(0.5, abs=0.01)
    counts = np.histogram(ranks, bins=10, range=(0, 1))[0]
    assert counts.min() > 0.08 * len(sample)


def test_reference_ranks_accept_a_scalar():
    """`decision_cutoff_rank` is one number, not an array.

    `np.interp` of a scalar returns a numpy scalar, which has no item assignment, so a tail
    written with boolean indexing raised `TypeError` at save time -- on a real fit only,
    which is why the unit tests missed it.
    """
    knots = np.linspace(0.065, 0.334, 1000)
    assert 0.0 <= float(rank_from_reference(0.9, knots=knots)) <= 1.0
    assert 0.0 <= float(rank_from_reference(0.01, knots=knots)) <= 1.0


def test_reference_ranks_extrapolate_past_the_library_ceiling():
    """A selective model scores generic chemistry low, so the reference stops well short
    of 1 -- measured 0.065 to 0.334. Clamping there would tie every active at exactly 1.0
    and stop `rank` ordering molecules at the only end anyone looks at."""
    knots = np.linspace(0.065, 0.334, 1000)
    above = rank_from_reference(np.array([0.4, 0.6, 0.9, 0.99]), knots=knots)
    assert np.all(np.diff(above) > 0)
    assert above.max() < 1.0


def test_reference_ranks_stay_monotone_across_the_joins():
    knots = np.linspace(0.2, 0.8, 500)
    scores = np.linspace(0.001, 0.999, 400)
    ranks = rank_from_reference(scores, knots=knots)
    assert np.all(np.diff(ranks) >= 0)
    assert ranks.min() >= 0.0 and ranks.max() <= 1.0

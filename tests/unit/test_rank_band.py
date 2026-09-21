"""Summarising where known molecules land on the rank scale.

Pure numpy, so it runs on a base install. The band is what turns a rank into a decision --
"known actives here score 0.85 to 0.95, and your compound scored 0.94" -- and it is
deliberately *reported* rather than used to anchor the scale. Anchoring the scale on the
out-of-fold actives would put every model's median active at 0.95 by construction and make a
model with AUC 0.95 indistinguishable from one with AUC 0.55.
"""

import numpy as np

from lazyqsar.qsar import _rank_band


def test_the_band_is_the_quartiles_with_a_count():
    band = _rank_band(np.linspace(0.0, 1.0, 101))
    assert band["n"] == 101
    assert band["rank_p25"] == 0.25
    assert band["rank_p50"] == 0.50
    assert band["rank_p75"] == 0.75


def test_the_quartiles_come_back_ordered():
    band = _rank_band(np.random.default_rng(0).random(500))
    assert band["rank_p25"] <= band["rank_p50"] <= band["rank_p75"]


def test_an_empty_class_is_none_rather_than_an_error():
    """A task with no actives, or none surviving, must not fail a fit over an advisory
    number."""
    assert _rank_band(np.array([])) is None


def test_non_finite_ranks_are_dropped_not_propagated():
    """A NaN would otherwise poison every quantile and report the model as unscored."""
    band = _rank_band(np.array([0.2, np.nan, 0.4, np.inf, 0.6]))
    assert band["n"] == 3
    assert band["rank_p50"] == 0.4


def test_all_non_finite_is_treated_as_empty():
    assert _rank_band(np.array([np.nan, np.inf])) is None


def test_a_single_molecule_still_gives_a_band():
    band = _rank_band(np.array([0.7]))
    assert band == {"n": 1, "rank_p25": 0.7, "rank_p50": 0.7, "rank_p75": 0.7}

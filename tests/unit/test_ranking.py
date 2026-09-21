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


def _reference(n=50_000, lo=0.065, hi=0.334, seed=0):
    """A reference library shaped like a real one: a selective model scores generic
    chemistry into a narrow band well below 1."""
    rng = np.random.default_rng(seed)
    return np.sort(lo + (hi - lo) * rng.beta(2.0, 3.0, n))


def test_the_quartiles_of_the_reference_land_on_the_quartiles_of_the_scale():
    """The anchoring, and the whole point of the scale.

    0.25, 0.50 and 0.75 mean exactly the quartiles of drug-like chemical space, so between
    them `rank` is the true percentile.
    """
    ref = _reference()
    prep = prepare_knots(ref)
    q1, med, q3 = np.percentile(ref, [25, 50, 75])
    assert float(rank_from_reference(q1, prepared=prep)) == pytest.approx(
        0.25, abs=1e-4
    )
    assert float(rank_from_reference(med, prepared=prep)) == pytest.approx(
        0.50, abs=1e-4
    )
    assert float(rank_from_reference(q3, prepared=prep)) == pytest.approx(
        0.75, abs=1e-4
    )


def test_a_quarter_of_the_reference_sits_outside_each_anchor():
    """Distribution-free, and the single assertion that catches almost any mis-wiring.

    Whatever shape the reference has, exactly a quarter of it must fall above 0.75 and a
    quarter below 0.25 -- that is what anchoring on quartiles means.
    """
    for seed in (0, 1, 2):
        ref = _reference(seed=seed)
        r = rank_from_reference(ref, prepared=prepare_knots(ref))
        assert (r > 0.75).mean() == pytest.approx(0.25, abs=1e-3)
        assert (r < 0.25).mean() == pytest.approx(0.25, abs=1e-3)


def test_molecules_past_the_reference_keep_spreading_instead_of_tying_at_one():
    """Why this is not a plain ECDF.

    A selective model's actives sit two to eight reference-IQRs above the reference median.
    An ECDF pins all of them at exactly 1.0, which is an unreadable hit list; here they
    stay ordered and visibly apart.
    """
    ref = _reference()
    prep = prepare_knots(ref)
    actives = np.array([0.40, 0.55, 0.70, 0.85, 0.95])
    r = rank_from_reference(actives, prepared=prep)
    assert np.all(np.diff(r) > 0)
    assert r.max() - r.min() > 0.1, "actives must be separable, not a wall of 1.000"
    assert r.max() < 1.0
    # ... and every one of them is above the whole reference library.
    assert r.min() > rank_from_reference(ref.max(), prepared=prep)


def test_reference_ranks_accept_a_scalar():
    """`decision_cutoff_rank` is one number, not an array.

    `np.interp` of a scalar returns a numpy scalar, which has no item assignment, so a
    branch written with boolean indexing raised `TypeError` at save time -- on a real fit
    only, which is why the unit tests missed it.
    """
    prep = prepare_knots(_reference())
    for p in (0.01, 0.2, 0.9):
        assert 0.0 <= float(rank_from_reference(p, prepared=prep)) <= 1.0


def test_reference_ranks_are_monotone_and_bounded_across_both_joins():
    ref = _reference()
    grid = np.linspace(1e-6, 1.0, 20_000)
    r = rank_from_reference(grid, prepared=prepare_knots(ref))
    assert np.all(np.diff(r) >= 0)
    assert r.min() >= 0.0 and r.max() <= 1.0


def test_the_scale_is_continuous_where_the_segments_meet():
    """Three pieces joined at Q1 and Q3; a gap there would be a visible discontinuity in
    every screening output."""
    ref = _reference()
    prep = prepare_knots(ref)
    for q in np.percentile(ref, [25, 75]):
        below = float(rank_from_reference(q - 1e-9, prepared=prep))
        above = float(rank_from_reference(q + 1e-9, prepared=prep))
        assert below == pytest.approx(above, abs=1e-6)


def test_only_certainty_reaches_one():
    prep = prepare_knots(_reference())
    assert float(rank_from_reference(1.0, prepared=prep)) == pytest.approx(1.0)
    assert float(rank_from_reference(0.999, prepared=prep)) < 1.0


def test_a_degenerate_reference_does_not_divide_by_zero():
    """One distinct value carries no interior. It must still return something monotone and
    bounded rather than raising."""
    r = rank_from_reference(np.array([0.1, 0.4, 0.9]), knots=np.full(16, 0.4))
    assert np.all(np.diff(r) >= 0)
    assert r.min() >= 0.0 and r.max() <= 1.0


# ------------------------------------------------------------------ tail anchors


def test_the_anchors_land_exactly_on_their_ranks():
    """`p05` of the inactives is 0.05 and `p95` of the actives is 0.95, by construction."""
    ref = _reference()
    prep = prepare_knots(ref)
    low, high = 0.03, 0.40
    assert float(rank_from_reference(low, prepared=prep, anchors=(low, high))) == (
        pytest.approx(0.05)
    )
    assert float(rank_from_reference(high, prepared=prep, anchors=(low, high))) == (
        pytest.approx(0.95)
    )


def test_anchoring_leaves_the_reference_quartiles_alone():
    """The middle of the scale still means the quartiles of drug-like chemical space."""
    ref = _reference()
    prep = prepare_knots(ref)
    q1, med, q3 = np.percentile(ref, [25, 50, 75])
    anchors = (0.03, 0.40)
    for value, expected in ((q1, 0.25), (med, 0.50), (q3, 0.75)):
        assert float(
            rank_from_reference(value, prepared=prep, anchors=anchors)
        ) == pytest.approx(expected, abs=1e-3)


def test_no_anchors_reproduces_the_unanchored_scale_exactly():
    """A checkpoint fitted before anchoring must rank bit-for-bit as it always did."""
    ref = _reference()
    prep = prepare_knots(ref)
    grid = np.linspace(1e-9, 1.0, 5_000)
    assert np.array_equal(
        rank_from_reference(grid, prepared=prep),
        rank_from_reference(grid, prepared=prep, anchors=None),
    )


def test_the_anchors_make_the_top_of_the_scale_reachable():
    """The reason they exist.

    A model whose probabilities stop at 0.45 could otherwise never exceed rank 0.85, so a
    sixth of the scale was unreachable -- and the ceiling moves with prevalence as much as
    with skill, so a perfect model on a rare target read lower than a mediocre one on an
    easy target.
    """
    ref = _reference()
    prep = prepare_knots(ref)
    ceiling = 0.45
    unanchored = float(rank_from_reference(ceiling, prepared=prep))
    anchored = float(rank_from_reference(ceiling, prepared=prep, anchors=(0.03, 0.42)))
    assert unanchored < 0.87
    assert anchored > 0.95


@pytest.mark.parametrize(
    "anchors",
    [
        (None, 0.05),  # high anchor below Q3
        (0.5, None),  # low anchor above Q1
        (0.5, 0.05),  # both incompatible
    ],
)
def test_an_incompatible_anchor_falls_back_to_the_plain_segment(anchors):
    """It must never produce a non-monotone scale, and must match the unanchored result on
    the side that fell back."""
    ref = _reference()
    prep = prepare_knots(ref)
    grid = np.linspace(1e-9, 1.0, 5_000)
    got = rank_from_reference(grid, prepared=prep, anchors=anchors)
    assert np.array_equal(got, rank_from_reference(grid, prepared=prep))


def test_one_side_can_anchor_while_the_other_falls_back():
    ref = _reference()
    prep = prepare_knots(ref)
    q1 = float(np.percentile(ref, 25))
    # Low anchor is above Q1 and therefore unusable; the high one still applies.
    got = rank_from_reference(np.array([0.40]), prepared=prep, anchors=(q1 * 2, 0.40))
    assert float(got[0]) == pytest.approx(0.95)


def test_the_anchored_scale_is_monotone_and_continuous():
    ref = _reference()
    prep = prepare_knots(ref)
    low, high = 0.03, 0.40
    grid = np.linspace(1e-9, 1.0, 50_000)
    r = rank_from_reference(grid, prepared=prep, anchors=(low, high))
    assert np.all(np.diff(r) >= 0)
    assert r.min() >= 0.0 and r.max() <= 1.0
    q1, q3 = np.percentile(ref, [25, 75])
    for join in (low, q1, q3, high):
        below = float(
            rank_from_reference(join - 1e-9, prepared=prep, anchors=(low, high))
        )
        above = float(
            rank_from_reference(join + 1e-9, prepared=prep, anchors=(low, high))
        )
        assert below == pytest.approx(above, abs=1e-6)


def test_anchored_ranks_accept_a_scalar():
    prep = prepare_knots(_reference())
    assert (
        0.0
        <= float(rank_from_reference(0.9, prepared=prep, anchors=(0.03, 0.40)))
        <= 1.0
    )

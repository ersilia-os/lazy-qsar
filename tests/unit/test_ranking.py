"""Unit tests for lazyqsar.utils.ranking (numpy-only; runs on the base install)."""

import numpy as np
import pytest

from lazyqsar.utils.ranking import (
    ACTIVES_RANK,
    DECISION_RANK,
    TAIL_ANCHORS,
    prepare_knots,
    proba_from_reference_rank,
    rank_from_knots,
    rank_from_reference,
    reference_anchor_table,
)


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


def test_the_tail_percentiles_land_on_their_ranks():
    """The anchoring, and the whole point of the scale.

    0.50 means the top 10% of drug-like chemical space, 0.65 the top 1%, 0.75 the top 0.1%
    -- each step a 10x shrink of the tail, which is where a bioactivity model's hits are.
    """
    ref = _reference()
    prep = prepare_knots(ref)
    for q, rank in TAIL_ANCHORS:
        p = float(np.interp(q / 100.0, prep[1], prep[0]))
        assert float(rank_from_reference(p, prepared=prep)) == pytest.approx(
            rank, abs=1e-4
        )


def test_the_reference_tail_fractions_are_distribution_free():
    """The single assertion that catches almost any mis-wiring.

    Whatever shape the reference has, the fraction of it above each anchor rank is fixed by
    the anchor's percentile: half above 0.25, a tenth above 0.50, a hundredth above 0.65.
    That is what anchoring on tail percentiles means -- and the 0.65 row is the property the
    decision cutoff rests on.
    """
    for seed in (0, 1, 2):
        ref = _reference(seed=seed)
        r = rank_from_reference(ref, prepared=prepare_knots(ref))
        for q, rank in TAIL_ANCHORS:
            assert (r > rank).mean() == pytest.approx(1.0 - q / 100.0, abs=2e-3)


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


@pytest.mark.parametrize("anchors", [None, (0.03, 0.40)], ids=["plain", "anchored"])
def test_reference_ranks_are_monotone_and_bounded_across_every_join(anchors):
    """One assertion for both scale variants: they differ in where the joins are, not in
    what must hold across them."""
    ref = _reference()
    grid = np.linspace(1e-9, 1.0, 50_000)
    r = rank_from_reference(grid, prepared=prepare_knots(ref), anchors=anchors)
    assert np.all(np.diff(r) >= 0)
    assert r.min() >= 0.0 and r.max() <= 1.0


def test_the_scale_is_continuous_where_the_segments_meet():
    """A gap at any anchor would be a visible discontinuity in every screening output."""
    ref = _reference()
    prep = prepare_knots(ref)
    joins = [float(np.interp(q / 100.0, prep[1], prep[0])) for q, _ in TAIL_ANCHORS]
    for p in joins:
        below = float(rank_from_reference(p - 1e-9, prepared=prep))
        above = float(rank_from_reference(p + 1e-9, prepared=prep))
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


def test_the_actives_anchor_lands_exactly_on_its_rank():
    """`p95` of the out-of-fold actives is 0.95, by construction."""
    ref = _reference()
    prep = prepare_knots(ref)
    high = 0.40
    assert float(
        rank_from_reference(high, prepared=prep, anchors=(0.03, high))
    ) == pytest.approx(ACTIVES_RANK)


def test_the_low_element_of_the_anchor_pair_does_not_shape_the_scale():
    """The tail table has no low anchor; `p05` of the inactives is recorded, not used.

    Passing any low value must give bit-identical ranks, because the region it used to
    govern now lies inside the ``0 -> reference p50`` segment.
    """
    prep = prepare_knots(_reference())
    grid = np.linspace(1e-9, 1.0, 5_000)
    baseline = rank_from_reference(grid, prepared=prep, anchors=(None, 0.40))
    for low in (0.001, 0.03, 0.5, 0.99):
        got = rank_from_reference(grid, prepared=prep, anchors=(low, 0.40))
        assert np.array_equal(got, baseline)


def test_anchoring_leaves_the_reference_percentiles_alone():
    """The actives anchor must not move the marks below it.

    It enters the table above the reference's p99.9, so every tail anchor keeps its rank --
    which is what makes the decision cutoff at 0.65 independent of the training set.
    """
    ref = _reference()
    prep = prepare_knots(ref)
    for q, rank in TAIL_ANCHORS:
        p = float(np.interp(q / 100.0, prep[1], prep[0]))
        assert float(
            rank_from_reference(p, prepared=prep, anchors=(0.03, 0.40))
        ) == pytest.approx(rank, abs=1e-4)


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


def test_the_anchored_scale_is_continuous_at_all_four_joins():
    """Monotonicity and bounds are covered above; this is the part unique to anchoring --
    four joins instead of two, and a gap at any of them is a visible step in every output."""
    ref = _reference()
    prep = prepare_knots(ref)
    low, high = 0.03, 0.40
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


# --------------------------------------------------------------------------- anchor table
#
# The shared (probability -> rank) table, and its inverse. No caller yet: `rank_from_reference`
# still interpolates the quartile-anchored segments. These tests pin the table and the inverse
# against each other so that switching the forward map over cannot silently break the
# round-trip the decision cutoff depends on.


def test_the_tail_anchors_land_exactly_on_their_ranks():
    """Each anchor percentile receives exactly the rank the table assigns it."""
    prep = prepare_knots(_reference())
    xs, ys, used = reference_anchor_table(prepared=prep)
    assert not used  # no anchor_high supplied
    for q, rank in TAIL_ANCHORS:
        p = float(np.interp(q / 100.0, prep[1], prep[0]))
        assert float(np.interp(p, xs, ys)) == pytest.approx(rank, abs=1e-12)


def test_the_inverse_recovers_the_probability_exactly():
    prep = prepare_knots(_reference())
    for anchor_high in (None, 0.40, 0.95):
        xs, ys, _ = reference_anchor_table(prepared=prep, anchor_high=anchor_high)
        # Sample ranks rather than probabilities: the map is what must round-trip, and
        # sampling its output side exercises every segment at the same density.
        ranks = np.linspace(0.0, 1.0, 2001)
        back = proba_from_reference_rank(ranks, prepared=prep, anchor_high=anchor_high)
        assert np.all(np.diff(back) >= 0)
        assert float(np.abs(np.interp(back, xs, ys) - ranks).max()) < 1e-9


def test_the_decision_rank_never_touches_the_actives_anchor():
    """The property that makes a fixed cutoff defensible on a model with few actives.

    DECISION_RANK sits at or below the last reference anchor, so the probability it inverts
    to must not move when the out-of-fold actives anchor appears or changes. A future anchor
    tweak that broke this would otherwise be invisible.
    """
    prep = prepare_knots(_reference())
    baseline = float(proba_from_reference_rank(DECISION_RANK, prepared=prep))
    for anchor_high in (0.40, 0.60, 0.95, None):
        got = float(
            proba_from_reference_rank(
                DECISION_RANK, prepared=prep, anchor_high=anchor_high
            )
        )
        assert got == pytest.approx(baseline, abs=1e-12)


def test_the_decision_rank_is_the_reference_ninety_ninth_percentile():
    """0.65 must mean a 1% generic hit rate, which is the entire reason for the constant."""
    knots = _reference()
    prep = prepare_knots(knots)
    p_cut = float(proba_from_reference_rank(DECISION_RANK, prepared=prep))
    assert float((knots >= p_cut).mean()) == pytest.approx(0.01, abs=1e-3)


def test_a_narrow_library_collapses_colliding_anchors_instead_of_breaking():
    """p99 and p99.9 can land on one probability, and np.interp needs a rising xp.

    Measured on a real model whose reference library spans only 0.17 to 0.26. The cutoff
    then comes out strictly *below* p99, which is why the anchor probabilities are read off
    the collapsed table rather than from np.percentile.
    """
    knots = np.sort(np.full(10_000, 0.2) + np.linspace(0, 0.06, 10_000))
    knots = np.concatenate([knots[:9_990], np.full(10, knots[-1])])
    prep = prepare_knots(knots)
    xs, ys, _ = reference_anchor_table(prepared=prep)
    assert np.all(np.diff(xs) > 0)
    assert np.all(np.diff(ys) > 0)
    p_cut = float(proba_from_reference_rank(DECISION_RANK, prepared=prep))
    assert p_cut <= float(np.interp(0.99, prep[1], prep[0])) + 1e-12


def test_the_actives_anchor_is_dropped_when_it_would_fold_the_table():
    """A model whose known actives do not clear the reference's p99.9.

    Measured on mtuberculosis_743175: p95 of its out-of-fold actives is 0.429 while the
    reference's p99.9 is 0.491, so pinning 0.95 to the actives would make a lower
    probability carry a higher rank.
    """
    prep = prepare_knots(_reference())
    last_reference = float(np.interp(0.999, prep[1], prep[0]))

    xs, ys, used = reference_anchor_table(
        prepared=prep, anchor_high=last_reference - 1e-3
    )
    assert not used
    assert np.all(np.diff(xs) > 0)
    assert np.all(np.diff(ys) > 0)

    _, _, used_ok = reference_anchor_table(
        prepared=prep, anchor_high=last_reference + 1e-3
    )
    assert used_ok


def test_the_table_spends_its_resolution_on_the_tail():
    """The reason for tail anchors: the library's top 1% gets far more axis than its middle.

    Under a flat percentile the middle 50% would get 0.50 of the axis and the top 1% would
    get 0.01. Here the trade is deliberately the other way round.
    """
    prep = prepare_knots(_reference())
    xs, ys, _ = reference_anchor_table(prepared=prep)

    def rank_at_percentile(q):
        """The rank awarded to the probability at library percentile *q*."""
        return float(np.interp(float(np.interp(q, prep[1], prep[0])), xs, ys))

    middle_50 = rank_at_percentile(0.75) - rank_at_percentile(0.25)
    top_1 = rank_at_percentile(1.0 - 1e-9) - rank_at_percentile(0.99)
    assert top_1 > 0.10  # a flat percentile would give 0.01
    assert middle_50 < 0.30  # a flat percentile would give 0.50

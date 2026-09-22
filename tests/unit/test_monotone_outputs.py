"""One defect, three layers: an output column that orders molecules differently from `proba`.

`proba`, `rank` and `score` are the same opinion re-expressed, so sorting a screening run by
any of them must give the same order. Until v3.5.0 two of them did not, and the cause is not
the obvious one. Calibration never reorders a head's own molecules -- every calibrator is
monotone -- but the heads get *different* curves, so calibration changes how far apart each
head's opinions sit. A weighted average of differently-stretched monotone curves is not a
monotone function of the weighted average of the originals. No way of pooling the raw values
fixes that; both outputs have to be derived from the pooled probability instead.

What a user would have seen: a top-100 list that differs depending on which column you
sorted by, with no basis for choosing between them.

These were three files -- `test_pooled_rank_reference.py`, `test_combine_pooled_rank.py` and
`test_combine_score_ordering.py` -- split by changeset rather than by subject. They are one
subject, in three sections:

1. the reference itself, from first principles: why an average of percentiles is not the
   percentile of an average, and what `build_pooled_rank_knots` has to produce instead;
2. `rank` through `combine`, including the refusal when a checkpoint predates the reference;
3. `score` through `combine`, which is the same argument for a different column.

Pure numpy: no model is fitted anywhere here. That is what makes this the *stronger* proof
of the multi-descriptor case -- a real fast-mode fit has one descriptor, where the weights
collapse to a single column and the defect cannot appear at all.
"""

import numpy as np
import pytest

from lazyqsar.ensemble import EnsembleSpec, build_pooled_rank_knots, combine
from lazyqsar.utils.ranking import rank_from_knots, rank_from_reference

# ---------------------------------------------------------------- 1. the reference itself


def _descriptor_channels(n=200, d=3, seed=7):
    """Per-descriptor probabilities, and the percentile of each within its own column.

    The shape the real channels have: ``R`` is each descriptor's own ECDF evaluated at its
    own prediction, which is what every head actually reports.
    """
    rng = np.random.default_rng(seed)
    Y = rng.uniform(0.02, 0.98, size=(n, d))
    R = np.column_stack([rank_from_knots(Y[:, j], Y[:, j]) for j in range(d)])
    return Y, R


def _reference_spec(d=3, **kw):
    return EnsembleSpec(
        descriptor_names=tuple(f"d{j}" for j in range(d)),
        oof_aucs=tuple(0.6 + 0.1 * j for j in range(d)),
        **kw,
    )


def test_averaging_percentiles_disagrees_with_the_percentile_of_the_average():
    """The defect, reproduced from first principles.

    If these two ever stop disagreeing, this test is no longer measuring anything and the
    ones below are guarding a distinction that does not exist.
    """
    Y, R = _descriptor_channels()
    spec = _reference_spec()
    res = combine(Y, R, R, None, spec=spec, outputs=("proba",))
    # Computed here rather than asked of `combine`, which no longer offers it: this is the
    # pre-v3.5.0 arithmetic, kept only to show what the pooled reference replaced.
    legacy = (res.weights * R).sum(axis=1)
    proba = res.values["proba"][:, 1]

    assert not np.array_equal(
        np.argsort(legacy, kind="stable"), np.argsort(proba, kind="stable")
    ), "the fixture no longer exercises the disagreement this change exists to remove"


def test_the_reference_is_the_sorted_pooled_probability():
    Y, R = _descriptor_channels()
    spec = _reference_spec()
    knots = build_pooled_rank_knots(Y, R, R, None, spec)
    expected = np.sort(
        combine(Y, R, R, None, spec=spec, outputs=("proba",)).values["proba"][:, 1]
    )
    np.testing.assert_array_equal(knots, expected)


def test_the_reference_makes_rank_agree_with_proba_across_descriptors():
    """The fix, at the level the defect lived on."""
    Y, R = _descriptor_channels()
    spec_without = _reference_spec()
    spec_with = _reference_spec(
        pooled_rank_knots=build_pooled_rank_knots(Y, R, R, None, spec_without)
    )

    res = combine(Y, R, R, None, spec=spec_with, outputs=("proba", "rank"))
    proba, rank = res.values["proba"][:, 1], res.values["rank"][:, 1]
    np.testing.assert_array_equal(
        np.argsort(rank, kind="stable"), np.argsort(proba, kind="stable")
    )


def test_the_reference_is_uniform_on_the_data_it_was_built_from():
    """The self-check that catches a reference built from the wrong quantity.

    A reference assembled with the wrong weights still produces a monotone rank in [0, 1],
    so no ordering test can see it. What it stops doing is spreading its own training rows
    evenly, because the knots are no longer that distribution's quantiles.
    """
    Y, R = _descriptor_channels()
    knots = build_pooled_rank_knots(Y, R, R, None, _reference_spec())
    self_ranks = rank_from_knots(knots, knots)
    assert abs(self_ranks.mean() - 0.5) < 0.02
    counts = np.histogram(self_ranks, bins=10, range=(0.0, 1.0))[0]
    assert counts.min() > 0.5 * (len(knots) / 10), counts.tolist()


def test_the_reference_is_capped():
    Y, R = _descriptor_channels(n=500)
    knots = build_pooled_rank_knots(Y, R, R, None, _reference_spec(), max_knots=100)
    assert len(knots) == 100
    assert np.array_equal(knots, np.sort(knots))


def test_no_reference_without_every_channel():
    """A reference pooled with different weights from the ones inference uses is worse
    than none, and nothing downstream could detect it."""
    Y, R = _descriptor_channels()
    assert build_pooled_rank_knots(Y, None, R, None, _reference_spec()) is None
    assert build_pooled_rank_knots(Y, R, None, None, _reference_spec()) is None
    assert build_pooled_rank_knots(None, R, R, None, _reference_spec()) is None
    assert (
        build_pooled_rank_knots(np.empty((0, 3)), R, R, None, _reference_spec()) is None
    )


# ------------------------------------------------------- 2. `rank` through combine

# Spans the whole probability range, so nothing is clamped to the ECDF's flat ends and
# every comparison below is about the interpolated interior.
WIDE_KNOTS = np.linspace(0.001, 0.999, 1000)


def _inputs(n=60, d=3, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(0.01, 0.99, size=(n, d)),  # Y
        rng.uniform(0.0, 1.0, size=(n, d)),  # R
        rng.uniform(0.0, 1.0, size=(n, d)),  # S
        rng.uniform(0.0, 1.0, size=(n, d)),  # A
    )


def _rank_spec(d=3, **kw):
    return EnsembleSpec(
        descriptor_names=tuple(f"d{j}" for j in range(d)),
        oof_aucs=tuple(0.6 + 0.1 * j for j in range(d)),
        ad_hard_cutoffs=(0.2,) * d,
        **kw,
    )


def test_rank_is_the_reference_scale_applied_to_the_pooled_probability():
    """The definition, pinned: no averaging of per-descriptor ranks is involved.

    `rank_from_reference`, not `rank_from_knots` -- the reference's quartiles anchor the
    scale at 0.25/0.50/0.75, and a plain ECDF would tie every active at exactly 1.0.
    """
    Y, R, S, A = _inputs()
    res = combine(Y, R, S, A, spec=_rank_spec(pooled_rank_knots=WIDE_KNOTS))
    expected = rank_from_reference(res.values["proba"][:, 1], WIDE_KNOTS)
    np.testing.assert_array_equal(res.values["rank"][:, 1], expected)
    np.testing.assert_allclose(res.values["rank"].sum(axis=1), 1.0)


def test_rank_orders_molecules_exactly_as_proba_does():
    """The invariant the change exists to create.

    Top-k by rank is then the same set as top-k by proba, and any ordering-only metric --
    AUROC, AUPRC, BEDROC -- gives the same number computed from either.
    """
    Y, R, S, A = _inputs()
    res = combine(Y, R, S, A, spec=_rank_spec(pooled_rank_knots=WIDE_KNOTS))
    proba, rank = res.values["proba"][:, 1], res.values["rank"][:, 1]
    assert len(np.unique(proba)) == len(proba), "fixture should have no tied probas"
    by_proba = np.argsort(proba, kind="stable")
    assert np.all(np.diff(rank[by_proba]) > 0)
    np.testing.assert_array_equal(np.argsort(rank, kind="stable"), by_proba)


def test_equal_probabilities_get_equal_ranks():
    """Monotone both ways: the map cannot invent an order proba does not have."""
    Y = np.tile(np.array([[0.3, 0.6, 0.2]]), (4, 1))
    res = combine(Y, None, None, None, spec=_rank_spec(pooled_rank_knots=WIDE_KNOTS))
    assert len(np.unique(res.values["rank"][:, 1])) == 1


def test_asking_for_rank_alone_still_works():
    """`rank` needs the pooled log-odds sum now, and nothing else computes it."""
    Y, R, S, A = _inputs()
    spec = _rank_spec(pooled_rank_knots=WIDE_KNOTS)
    alone = combine(Y, R, S, A, spec=spec, outputs=("rank",))
    together = combine(Y, R, S, A, spec=spec, outputs=("proba", "rank"))
    np.testing.assert_array_equal(alone.values["rank"], together.values["rank"])
    assert set(alone.values) == {"rank"}


def test_without_a_reference_rank_refuses_rather_than_falling_back():
    """The old weighted mean of per-descriptor training percentiles is gone.

    It answered a different question. Returning it under the same name would mean one
    `rank` column meant "beats 99% of drug-like space" on one checkpoint and "beats 99% of
    its own training set" on another, with nothing in the output distinguishing them. An
    uncalibrated rank is worse than an error, because it gets believed.
    """
    Y, R, S, A = _inputs()
    with pytest.raises(ValueError, match="no reference-library rank"):
        combine(Y, R, S, A, spec=_rank_spec(), outputs=("rank",))


def test_the_refusal_names_the_remedy():
    """An error a user cannot act on is only marginally better than a wrong number."""
    Y, R, S, A = _inputs()
    with pytest.raises(ValueError) as exc:
        combine(Y, R, S, A, spec=_rank_spec(), outputs=("rank",))
    assert "refit" in str(exc.value).lower()
    assert "3.6" in str(exc.value)


def test_the_other_five_outputs_survive_a_missing_reference():
    """Only `rank` changed meaning, so only `rank` is withheld -- an old checkpoint is
    not bricked."""
    Y, R, S, A = _inputs()
    res = combine(
        Y,
        R,
        S,
        A,
        spec=_rank_spec(),
        outputs=("proba", "logit", "lift", "score", "binary"),
    )
    assert set(res.values) == {"proba", "logit", "lift", "score", "binary"}


@pytest.mark.parametrize("knots", [np.array([]), None])
def test_an_empty_reference_refuses_too(knots):
    """`prepare_knots` raises on an empty array; combine must never hand it one, and must
    not treat empty knots as a usable reference either."""
    Y, R, S, A = _inputs()
    with pytest.raises(ValueError, match="no reference-library rank"):
        combine(Y, R, S, A, spec=_rank_spec(pooled_rank_knots=knots), outputs=("rank",))


def test_a_single_distinct_knot_still_orders_molecules():
    """Degenerate but reachable: one distinct reference probability carries no resolution.

    It used to collapse to a step at 0.4, because `rank_from_knots` pins the open ends.
    A reference rank extrapolates them instead, so the result is still monotone in proba
    but no longer throws away the ordering of everything above and below the knot -- which
    is the same reason the tails are extrapolated in the non-degenerate case.
    """
    Y, R, S, A = _inputs()
    res = combine(Y, R, S, A, spec=_rank_spec(pooled_rank_knots=np.full(10, 0.4)))
    proba, rank = res.values["proba"][:, 1], res.values["rank"][:, 1]
    assert rank.min() >= 0.0 and rank.max() <= 1.0
    assert np.all(np.diff(rank[np.argsort(proba, kind="stable")]) >= 0)


def test_unparseable_rows_stay_nan():
    """A NaN probability must not come back as a confident percentile."""
    Y, R, S, A = _inputs()
    Y[3] = np.nan
    res = combine(Y, R, S, A, spec=_rank_spec(pooled_rank_knots=WIDE_KNOTS))
    assert np.isnan(res.values["rank"][3, 1])
    assert np.isfinite(res.values["rank"][np.arange(len(Y)) != 3, 1]).all()


# ------------------------------------------------------ 3. `score` through combine


def flipped_pairs(a, b):
    """Pairs *a* and *b* strictly disagree about. Ties are not disagreement."""
    sign_a = np.sign(np.subtract.outer(a, a))
    sign_b = np.sign(np.subtract.outer(b, b))
    return int(((sign_a * sign_b) < 0).sum() // 2)


@pytest.fixture
def disagreeing():
    """Two descriptors whose raw and calibrated columns rank molecules differently.

    Descriptor 0 is confident and well spread; descriptor 1 is compressed toward the middle
    and, on the raw scale, disagrees with it. That is the shape a real ensemble takes when
    its heads carry calibrators of different slopes.
    """
    rng = np.random.default_rng(0)
    n = 60
    a = np.linspace(0.05, 0.95, n)
    b = 0.5 + 0.12 * np.sin(np.linspace(0, 9, n))
    Y = np.column_stack([a, b])
    # Raw scores that are monotone per column but compressed differently, so pooling them
    # linearly does not reproduce the pooled-probability order.
    S = np.column_stack([a**3, 0.5 + 0.45 * (b - 0.5) * 6])
    S = np.clip(S, 1e-6, 1 - 1e-6)
    R = np.column_stack([rng.uniform(size=n), rng.uniform(size=n)])
    return Y, R, S


def _score_spec(score_knots=None):
    return EnsembleSpec(
        descriptor_names=("morgan", "rdkit"),
        oof_aucs=(0.80, 0.75),
        population_prior=0.3,
        pooled_score_knots=score_knots,
    )


def test_without_a_map_score_can_contradict_proba(disagreeing):
    """The behaviour the map replaces. If this stops holding the fixture has gone stale."""
    Y, R, S = disagreeing
    out = combine(Y, R, S, None, spec=_score_spec(), outputs=("proba", "score"))
    assert flipped_pairs(out.values["proba"][:, 1], out.values["score"][:, 1]) > 0, (
        "the fixture no longer produces a disagreement, so the test below proves nothing"
    )


def test_with_a_map_score_never_contradicts_proba(disagreeing):
    """Zero inverted pairs, which is the whole point of the pooled score map."""
    Y, R, S = disagreeing
    reference = combine(Y, R, S, None, spec=_score_spec(), outputs=("proba", "score"))
    p1 = reference.values["proba"][:, 1]
    s1 = reference.values["score"][:, 1]

    from lazyqsar.ensemble.reference import build_pooled_score_knots

    knots = build_pooled_score_knots(Y, R, S, None, _score_spec())
    assert knots is not None

    out = combine(Y, R, S, None, spec=_score_spec(knots), outputs=("proba", "score"))
    mapped = out.values["score"][:, 1]

    assert flipped_pairs(out.values["proba"][:, 1], mapped) == 0
    np.testing.assert_allclose(out.values["proba"][:, 1], p1, rtol=0, atol=0)
    # Still on the raw scale rather than collapsed onto proba: the map is fitted to the
    # raw values, so it cannot leave their range. How *closely* it tracks them is a
    # property of real data, not of this fixture -- here the wiggle being flattened is
    # deliberately enormous, and moving those points is the map doing its job. The real
    # measurement lives in tests/chem/test_outputs_agree_on_order.py.
    assert s1.min() <= mapped.min() and mapped.max() <= s1.max(), (
        "the mapped score left the range of the raw scores it is fitted to"
    )


def test_the_map_is_monotone_and_ascending(disagreeing):
    """`np.interp` needs ascending x, and the ordering guarantee needs non-decreasing y."""
    from lazyqsar.ensemble.reference import build_pooled_score_knots

    Y, R, S = disagreeing
    x, y = build_pooled_score_knots(Y, R, S, None, _score_spec())
    assert np.all(np.diff(x) > 0)
    assert np.all(np.diff(y) >= 0)


def test_no_map_is_built_without_the_raw_channel(disagreeing):
    """`S is None` means no descriptor could supply raw scores; there is nothing to map."""
    from lazyqsar.ensemble.reference import build_pooled_score_knots

    Y, R, _ = disagreeing
    assert build_pooled_score_knots(Y, R, None, None, _score_spec()) is None
    assert build_pooled_score_knots(None, R, None, None, _score_spec()) is None

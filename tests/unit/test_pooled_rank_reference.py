"""Building the pooled reference, and why the ensemble needed one.

``rank`` used to be a weighted mean of the descriptors' percentiles. The first test here
is the defect that made that wrong: with more than one descriptor, that mean can order two
molecules differently from the pooled probability, because an average of percentiles is
not the percentile of an average. Everything else pins the reference that replaces it.

Numpy only, and no scikit-learn anywhere: the multi-descriptor case is the one that
matters and the one a real fit cannot be relied on to produce -- which descriptors survive
the portfolio depends on the data and on the scikit-learn version. Constructing the
channels directly is what makes this deterministic.
"""

import numpy as np

from lazyqsar.ensemble import EnsembleSpec, build_pooled_rank_knots, combine
from lazyqsar.utils.ranking import rank_from_knots


def _descriptor_channels(n=200, d=3, seed=7):
    """Per-descriptor probabilities, and the percentile of each within its own column.

    The shape the real channels have: ``R`` is each descriptor's own ECDF evaluated at its
    own prediction, which is what every head actually reports.
    """
    rng = np.random.default_rng(seed)
    Y = rng.uniform(0.02, 0.98, size=(n, d))
    R = np.column_stack([rank_from_knots(Y[:, j], Y[:, j]) for j in range(d)])
    return Y, R


def _spec(d=3, **kw):
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
    spec = _spec()
    legacy = combine(Y, R, R, None, spec=spec, outputs=("rank",)).values["rank"][:, 1]
    proba = combine(Y, R, R, None, spec=spec, outputs=("proba",)).values["proba"][:, 1]

    assert not np.array_equal(
        np.argsort(legacy, kind="stable"), np.argsort(proba, kind="stable")
    ), "the fixture no longer exercises the disagreement this change exists to remove"


def test_the_reference_is_the_sorted_pooled_probability():
    Y, R = _descriptor_channels()
    spec = _spec()
    knots = build_pooled_rank_knots(Y, R, R, None, spec)
    expected = np.sort(
        combine(Y, R, R, None, spec=spec, outputs=("proba",)).values["proba"][:, 1]
    )
    np.testing.assert_array_equal(knots, expected)


def test_the_reference_makes_rank_agree_with_proba_across_descriptors():
    """The fix, at the level the defect lived on."""
    Y, R = _descriptor_channels()
    spec_without = _spec()
    spec_with = _spec(
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
    knots = build_pooled_rank_knots(Y, R, R, None, _spec())
    self_ranks = rank_from_knots(knots, knots)
    assert abs(self_ranks.mean() - 0.5) < 0.02
    counts = np.histogram(self_ranks, bins=10, range=(0.0, 1.0))[0]
    assert counts.min() > 0.5 * (len(knots) / 10), counts.tolist()


def test_the_reference_is_capped():
    Y, R = _descriptor_channels(n=500)
    knots = build_pooled_rank_knots(Y, R, R, None, _spec(), max_knots=100)
    assert len(knots) == 100
    assert np.array_equal(knots, np.sort(knots))


def test_no_reference_without_every_channel():
    """A reference pooled with different weights from the ones inference uses is worse
    than none, and nothing downstream could detect it."""
    Y, R = _descriptor_channels()
    assert build_pooled_rank_knots(Y, None, R, None, _spec()) is None
    assert build_pooled_rank_knots(Y, R, None, None, _spec()) is None
    assert build_pooled_rank_knots(None, R, R, None, _spec()) is None
    assert build_pooled_rank_knots(np.empty((0, 3)), R, R, None, _spec()) is None

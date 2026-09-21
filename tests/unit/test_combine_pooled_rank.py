"""``rank`` as the percentile of the pooled probability, and the fallback that predates it.

Before v3.5.0 the ensemble answered "where does this molecule sit" by averaging each
descriptor's answer. An average of percentiles is not the percentile of the average, so
``rank`` and ``proba`` could order two molecules differently -- and because an ECDF is
steep wherever the training scores bunch up, ``rank`` amplified the tiny float32 gap
between a model and its ONNX export into a visible one. Reading one pooled reference
makes ``rank`` a monotone view of ``proba`` and removes the amplification with it.

Numpy only, like the module under test: these run on a base install, so monotonicity is
asserted by ordering rather than by a correlation coefficient.
"""

import warnings

import numpy as np
import pytest

from lazyqsar.ensemble import EnsembleSpec, combine
from lazyqsar.utils.ranking import rank_from_knots

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


def _spec(d=3, **kw):
    return EnsembleSpec(
        descriptor_names=tuple(f"d{j}" for j in range(d)),
        oof_aucs=tuple(0.6 + 0.1 * j for j in range(d)),
        ad_hard_cutoffs=(0.2,) * d,
        **kw,
    )


def test_rank_is_the_ecdf_of_the_pooled_probability():
    """The definition, pinned: no averaging of per-descriptor ranks is involved."""
    Y, R, S, A = _inputs()
    res = combine(Y, R, S, A, spec=_spec(pooled_rank_knots=WIDE_KNOTS))
    expected = rank_from_knots(res.values["proba"][:, 1], WIDE_KNOTS)
    np.testing.assert_array_equal(res.values["rank"][:, 1], expected)
    np.testing.assert_allclose(res.values["rank"].sum(axis=1), 1.0)


def test_rank_orders_molecules_exactly_as_proba_does():
    """The invariant the change exists to create.

    Top-k by rank is then the same set as top-k by proba, and any ordering-only metric --
    AUROC, AUPRC, BEDROC -- gives the same number computed from either.
    """
    Y, R, S, A = _inputs()
    res = combine(Y, R, S, A, spec=_spec(pooled_rank_knots=WIDE_KNOTS))
    proba, rank = res.values["proba"][:, 1], res.values["rank"][:, 1]
    assert len(np.unique(proba)) == len(proba), "fixture should have no tied probas"
    by_proba = np.argsort(proba, kind="stable")
    assert np.all(np.diff(rank[by_proba]) > 0)
    np.testing.assert_array_equal(np.argsort(rank, kind="stable"), by_proba)


def test_equal_probabilities_get_equal_ranks():
    """Monotone both ways: the map cannot invent an order proba does not have."""
    Y = np.tile(np.array([[0.3, 0.6, 0.2]]), (4, 1))
    res = combine(Y, None, None, None, spec=_spec(pooled_rank_knots=WIDE_KNOTS))
    assert len(np.unique(res.values["rank"][:, 1])) == 1


def test_asking_for_rank_alone_still_works():
    """`rank` needs the pooled log-odds sum now, and nothing else computes it."""
    Y, R, S, A = _inputs()
    spec = _spec(pooled_rank_knots=WIDE_KNOTS)
    alone = combine(Y, R, S, A, spec=spec, outputs=("rank",))
    together = combine(Y, R, S, A, spec=spec, outputs=("proba", "rank"))
    np.testing.assert_array_equal(alone.values["rank"], together.values["rank"])
    assert set(alone.values) == {"rank"}


def test_without_a_pooled_reference_rank_is_the_old_weighted_mean():
    """Every checkpoint fitted before v3.5.0 is in this branch and must not move."""
    Y, R, S, A = _inputs()
    res = combine(Y, R, S, A, spec=_spec())
    np.testing.assert_array_equal(
        res.values["rank"][:, 1], (res.weights * R).sum(axis=1)
    )


def test_the_fallback_is_silent():
    """A missing reference is the normal state of an old checkpoint, not a problem."""
    Y, R, S, A = _inputs()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        combine(Y, R, S, A, spec=_spec())
    assert not caught, f"combine warned on the fallback path: {caught}"


@pytest.mark.parametrize("knots", [np.array([]), None])
def test_an_absent_reference_falls_back_rather_than_raising(knots):
    """`prepare_knots` raises on an empty array; combine must never hand it one."""
    Y, R, S, A = _inputs()
    res = combine(Y, R, S, A, spec=_spec(pooled_rank_knots=knots))
    np.testing.assert_array_equal(
        res.values["rank"][:, 1], (res.weights * R).sum(axis=1)
    )


def test_a_single_distinct_knot_still_orders_molecules():
    """Degenerate but reachable: one distinct reference probability carries no resolution.

    It used to collapse to a step at 0.4, because `rank_from_knots` pins the open ends.
    A reference rank extrapolates them instead, so the result is still monotone in proba
    but no longer throws away the ordering of everything above and below the knot -- which
    is the same reason the tails are extrapolated in the non-degenerate case.
    """
    Y, R, S, A = _inputs()
    res = combine(Y, R, S, A, spec=_spec(pooled_rank_knots=np.full(10, 0.4)))
    proba, rank = res.values["proba"][:, 1], res.values["rank"][:, 1]
    assert rank.min() >= 0.0 and rank.max() <= 1.0
    assert np.all(np.diff(rank[np.argsort(proba, kind="stable")]) >= 0)


def test_unparseable_rows_stay_nan():
    """A NaN probability must not come back as a confident percentile."""
    Y, R, S, A = _inputs()
    Y[3] = np.nan
    res = combine(Y, R, S, A, spec=_spec(pooled_rank_knots=WIDE_KNOTS))
    assert np.isnan(res.values["rank"][3, 1])
    assert np.isfinite(res.values["rank"][np.arange(len(Y)) != 3, 1]).all()

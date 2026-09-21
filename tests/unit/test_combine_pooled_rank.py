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


def test_without_a_reference_rank_refuses_rather_than_falling_back():
    """The old weighted mean of per-descriptor training percentiles is gone.

    It answered a different question. Returning it under the same name would mean one
    `rank` column meant "beats 99% of drug-like space" on one checkpoint and "beats 99% of
    its own training set" on another, with nothing in the output distinguishing them. An
    uncalibrated rank is worse than an error, because it gets believed.
    """
    Y, R, S, A = _inputs()
    with pytest.raises(ValueError, match="no reference-library rank"):
        combine(Y, R, S, A, spec=_spec(), outputs=("rank",))


def test_the_refusal_names_the_remedy():
    """An error a user cannot act on is only marginally better than a wrong number."""
    Y, R, S, A = _inputs()
    with pytest.raises(ValueError) as exc:
        combine(Y, R, S, A, spec=_spec(), outputs=("rank",))
    assert "refit" in str(exc.value).lower()
    assert "3.6" in str(exc.value)


def test_the_other_five_outputs_survive_a_missing_reference():
    """Only `rank` changed meaning, so only `rank` is withheld -- an old checkpoint is
    not bricked."""
    Y, R, S, A = _inputs()
    res = combine(
        Y, R, S, A, spec=_spec(), outputs=("proba", "logit", "lift", "score", "binary")
    )
    assert set(res.values) == {"proba", "logit", "lift", "score", "binary"}


@pytest.mark.parametrize("knots", [np.array([]), None])
def test_an_empty_reference_refuses_too(knots):
    """`prepare_knots` raises on an empty array; combine must never hand it one, and must
    not treat empty knots as a usable reference either."""
    Y, R, S, A = _inputs()
    with pytest.raises(ValueError, match="no reference-library rank"):
        combine(Y, R, S, A, spec=_spec(pooled_rank_knots=knots), outputs=("rank",))


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

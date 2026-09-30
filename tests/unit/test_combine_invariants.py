"""Invariants of ``combine()`` that must survive a deliberate change to the arithmetic.

The golden snapshot in ``test_combine_golden.py`` pins the exact numbers; these pin the
properties. When the formulas are intentionally revised, the golden file gets regenerated but
these tests should still pass -- weights are still a convex combination, probabilities still
sum to one, rows are still independent. A change that breaks one of these is a change to what
the ensemble *means*, not just to what it computes.

Numpy only, so it runs on every install.
"""

import numpy as np
import pytest

from _helpers.combine_cases import _case, _curves, _spec

from lazyqsar.ensemble import OUTPUT_NAMES, combine


def test_weights_sum_to_one():
    rng = np.random.default_rng(0)
    Y, R, S, A, kw = _case(
        rng,
        500,
        4,
        with_ad=True,
        with_rank=True,
        with_score=True,
        curves=_curves(rng, 4),
        veto="partial",
        no_skill=False,
    )
    result = combine(Y, R, S, A, spec=_spec(4, kw), outputs=("proba",))
    assert np.allclose(result.weights.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_proba_columns_sum_to_one():
    rng = np.random.default_rng(1)
    Y, R, S, A, kw = _case(
        rng,
        200,
        3,
        with_ad=False,
        with_rank=True,
        with_score=True,
        curves=None,
        veto="none",
        no_skill=False,
    )
    proba = combine(Y, R, S, A, spec=_spec(3, kw), outputs=("proba",)).values["proba"]
    assert np.allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_requesting_one_output_matches_requesting_all():
    """Output selection is an optimisation, not a behaviour change."""
    rng = np.random.default_rng(2)
    Y, R, S, A, kw = _case(
        rng,
        64,
        3,
        with_ad=True,
        with_rank=True,
        with_score=True,
        curves=_curves(rng, 3),
        veto="partial",
        no_skill=False,
    )
    # A reference, because these assert over every output and `rank` is a percentile
    # against one -- a checkpoint without it refuses rather than inventing a number.
    kw["pooled_knots"] = np.sort(rng.uniform(0.01, 0.99, size=250))
    spec = _spec(3, kw)
    everything = combine(Y, R, S, A, spec=spec, outputs=OUTPUT_NAMES).values
    for name in OUTPUT_NAMES:
        alone = combine(Y, R, S, A, spec=spec, outputs=(name,)).values[name]
        assert np.array_equal(alone, everything[name]), name


def test_float32_inputs_give_float64_results():
    """Channels may be accumulated in float32; the arithmetic still runs in float64."""
    rng = np.random.default_rng(3)
    Y, R, S, A, kw = _case(
        rng,
        50,
        3,
        with_ad=True,
        with_rank=True,
        with_score=True,
        curves=None,
        veto="none",
        no_skill=False,
    )
    spec = _spec(3, kw)
    wide = combine(Y, R, S, A, spec=spec, outputs=("proba",)).values["proba"]
    narrow = combine(
        Y.astype(np.float32),
        R.astype(np.float32),
        S.astype(np.float32),
        A.astype(np.float32),
        spec=spec,
        outputs=("proba",),
    ).values["proba"]
    assert wide.dtype == narrow.dtype == np.float64
    assert np.allclose(wide, narrow, rtol=0, atol=1e-6)


def test_rows_are_independent():
    """Splitting the input and combining the halves must match one call.

    This is the property the streaming prediction path depends on.
    """
    rng = np.random.default_rng(4)
    Y, R, S, A, kw = _case(
        rng,
        128,
        3,
        with_ad=True,
        with_rank=True,
        with_score=True,
        curves=_curves(rng, 3),
        veto="partial",
        no_skill=False,
    )
    kw["pooled_knots"] = np.sort(rng.uniform(0.01, 0.99, size=250))
    spec = _spec(3, kw)
    whole = combine(Y, R, S, A, spec=spec, outputs=OUTPUT_NAMES).values
    halves = [
        combine(Y[s], R[s], S[s], A[s], spec=spec, outputs=OUTPUT_NAMES).values
        for s in (slice(0, 51), slice(51, 128))
    ]
    for name in OUTPUT_NAMES:
        stitched = np.concatenate([h[name] for h in halves], axis=0)
        assert np.array_equal(stitched, whole[name]), name


def test_diagnostics_only_when_ad_present():
    rng = np.random.default_rng(5)
    Y, R, S, A, kw = _case(
        rng,
        20,
        3,
        with_ad=True,
        with_rank=True,
        with_score=True,
        curves=None,
        veto="none",
        no_skill=False,
    )
    spec = _spec(3, kw)
    assert combine(Y, R, S, A, spec=spec, outputs=("proba",)).diagnostics is not None
    assert combine(Y, R, S, None, spec=spec, outputs=("proba",)).diagnostics is None

    rows = combine(Y, R, S, A, spec=spec, outputs=("proba",)).diagnostics
    assert [r["name"] for r in rows] == ["d0", "d1", "d2"]
    for key in ("ad_mean", "weight_mean", "vetoed", "pred_mean"):
        assert key in rows[0]


def test_unknown_output_rejected():
    rng = np.random.default_rng(6)
    Y, R, S, A, kw = _case(
        rng,
        8,
        2,
        with_ad=False,
        with_rank=True,
        with_score=True,
        curves=None,
        veto="none",
        no_skill=False,
    )
    with pytest.raises(ValueError, match="Unknown output"):
        combine(Y, R, S, A, spec=_spec(2, kw), outputs=("probability",))


def test_spec_width_mismatch_rejected():
    rng = np.random.default_rng(7)
    Y, R, S, A, kw = _case(
        rng,
        8,
        3,
        with_ad=False,
        with_rank=True,
        with_score=True,
        curves=None,
        veto="none",
        no_skill=False,
    )
    with pytest.raises(ValueError, match="spec covers"):
        combine(
            Y,
            R,
            S,
            A,
            spec=_spec(
                2,
                {
                    **kw,
                    "oof_aucs": kw["oof_aucs"][:2],
                    "proxy_aucs": kw["proxy_aucs"][:2],
                    "ad_hard_cutoffs": None,
                },
            ),
        )

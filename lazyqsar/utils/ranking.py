"""Score-to-decision helpers shared by the fit-time estimators and the ONNX artifacts.

Both runtimes must turn a raw score into the same rank and the same label. ``numpy``-only
by design: the inference path must keep working without scikit-learn, XGBoost or RDKit
installed, so nothing heavier may be imported here.
"""

import numpy as np

# Absolute slack when comparing a score against a learned cutoff. ONNX inference runs in
# float32 and disagrees with the float64 fit-time model by ~1e-8 on the same input. Several
# estimators produce scores on an exact grid — a Random Forest vote fraction is a multiple
# of 1/n_estimators — and the cutoff is learned from those same values, so it frequently
# lands exactly on a grid point. A bare ``>=`` then flips those samples between the two
# runtimes: measured at 6.5% of labels for a 40-tree forest with cutoff 0.025. The slack is
# far below any meaningful decision margin and makes the comparison runtime-independent.
_CUTOFF_ATOL = 1e-6


def binarize(scores, threshold):
    """Return 0/1 labels for *scores* against *threshold*, tolerant of exact ties.

    Equivalent to ``scores >= threshold`` up to :data:`_CUTOFF_ATOL`, which keeps fit-time
    and ONNX predictions identical for scores sitting exactly on the cutoff.
    """
    return (np.asarray(scores, dtype=np.float64) >= threshold - _CUTOFF_ATOL).astype(
        int
    )


def subsample_knots(sorted_scores, max_knots=10_000):
    """Thin an ascending score array to at most *max_knots* points, keeping its shape.

    Evenly spaced positions, so the retained points are quantiles of the original and the
    ECDF they describe is unchanged to within the spacing. Identical to the private copy
    each base estimator uses for its own ranker knots; those predate this helper and are
    left alone so no checkpoint's knots shift under a refactor.

    Parameters
    ----------
    sorted_scores : ndarray
        Scores in ascending order.
    max_knots : int, default 10000
        Cap on the number of retained points.

    Returns
    -------
    ndarray
        *sorted_scores* unchanged when it is already short enough, else the subsample.
    """
    n = len(sorted_scores)
    if n <= max_knots:
        return sorted_scores
    idx = np.round(np.linspace(0, n - 1, max_knots)).astype(int)
    return sorted_scores[idx]


def prepare_knots(knots):
    """Collapse sorted ECDF knots to distinct values carrying mid-rank quantiles.

    ``np.interp`` with a duplicated ``xp`` returns the ``fp`` of the *last* duplicate, so a
    score falling inside a tie plateau is awarded the rank of the plateau's top edge rather
    than its middle. Random Forest is worst affected: ``predict_proba`` over ``N`` trees
    yields only ``N + 1`` distinct values, so most scores land exactly on a tie.

    Collapsing each run of equal knots to a single point at its mid-rank restores the
    standard convention ``(n_below + 0.5 * n_equal) / n``.

    Parameters
    ----------
    knots : array_like
        ECDF knots, typically the out-of-fold raw scores. Need not be sorted.

    Returns
    -------
    vals : ndarray of shape (n_distinct,)
        Distinct knot values, ascending.
    midranks : ndarray of shape (n_distinct,)
        Mid-rank quantile in [0, 1] for each distinct value.
    """
    k = np.sort(np.asarray(knots, dtype=np.float64).ravel())
    n = len(k)
    if n == 0:
        raise ValueError("Cannot build a ranker from an empty knot array.")
    vals, first = np.unique(k, return_index=True)
    last = np.searchsorted(k, vals, side="right")
    midranks = (first + last) / (2.0 * n)
    return vals, midranks


def rank_from_knots(scores, knots=None, prepared=None):
    """Return [0, 1] ECDF ranks for *scores*, interpolating between distinct knots.

    Parameters
    ----------
    scores : array_like
        Scores to rank.
    knots : array_like, optional
        ECDF knots, prepared on the fly. Ignored when *prepared* is given.
    prepared : tuple, optional
        The ``(vals, midranks)`` pair from :func:`prepare_knots`, to avoid re-deriving it
        on every call.

    Returns
    -------
    ndarray
        Ranks in [0, 1], same shape as *scores*. Scores below every knot return 0.0 and
        scores above every knot return 1.0, matching the plain-ECDF endpoints.
    """
    if prepared is None:
        if knots is None:
            raise ValueError("Provide either `knots` or `prepared`.")
        prepared = prepare_knots(knots)
    vals, midranks = prepared
    scores = np.asarray(scores, dtype=np.float64)

    if len(vals) == 1:
        # Degenerate ranker: a single distinct knot is an atom at `vals[0]`, so only the
        # strict comparisons carry information.
        ranks = np.full(scores.shape, midranks[0], dtype=np.float64)
    else:
        ranks = np.interp(scores, vals, midranks)

    # The lowest distinct knot carries its mid-rank, not 0, so `np.interp` would clamp a
    # score below the whole training range up to that mid-rank. Pin the open ends instead.
    ranks = np.where(scores < vals[0], 0.0, ranks)
    ranks = np.where(scores > vals[-1], 1.0, ranks)
    return ranks


def score_from_knots(p1, knots):
    """Map pooled probabilities onto the pre-calibration scale through *knots*.

    Linear interpolation, clamped at both ends, so the result is monotone in *p1* and
    therefore orders molecules exactly as ``proba`` does.
    """
    x, y = knots
    return np.interp(np.asarray(p1, dtype=np.float64), x, y)


def rank_from_reference(scores, knots=None, prepared=None):
    """Position against a reference library, anchored on its quartiles.

    Between the reference's first and third quartiles this **is** the exact percentile, so
    0.25, 0.50 and 0.75 are precisely the quartiles of drug-like chemical space. Outside
    that range it continues as a bounded linear function of probability::

        Q1 <= p <= Q3   ECDF(p)                        (runs 0.25 -> 0.75 by definition)
        p > Q3          0.75 + 0.25 * (p - Q3)/(1 - Q3)
        p < Q1          0.25 * p / Q1

    Above Q3 the value is therefore **not** a percentile: a molecule at 0.95 beats far more
    than 95% of drug-like space. It is a readable, bounded, order-preserving scale.

    Why not the plain ECDF everywhere. A selective model scores generic chemistry low --
    measured on an antimicrobial model, the 50,000 reference molecules spanned a pooled
    probability of only 0.065 to 0.334, while its actives sat at 0.4 to 0.95, which is two
    to eight reference-IQRs above the reference median. An ECDF pins every one of them at
    exactly 1.0, and so does any other scale calibrated to the reference's spread: a
    logistic fitted to its quartiles only moves from 0.9916 to 0.99999999 across that whole
    range. Being eight IQRs outside generic chemistry *is* the model working, so a
    calibrated measure has no choice but to say "at the top" -- which leaves a hit list as
    an undifferentiated wall of 1.000 and stops `rank` ordering molecules at the one end
    anyone looks at.

    Why the quartiles and not the range. Q1 and Q3 barely move under resampling or a refit.
    The minimum and maximum of a 50,000-sample are order statistics, the least reproducible
    numbers in the distribution, so anchoring on them would put the scale's boundaries
    somewhere different for every model. They are deliberately unused, which also leaves
    this construction with no free parameter: `proba` is already bounded, so the outer
    segments have a natural endpoint.

    Parameters
    ----------
    scores : array_like or float
        Pooled probabilities. A scalar is accepted -- the decision cutoff is expressed as a
        rank through this function.
    knots, prepared
        The reference, as raw knots or as the output of :func:`prepare_knots`.

    Returns
    -------
    ndarray
        Ranks in [0, 1], monotone in *scores*, reaching 1.0 only as the probability does.
    """
    vals, midranks = prepare_knots(knots) if prepared is None else prepared
    scores = np.asarray(scores, dtype=np.float64)

    if len(vals) == 1:
        # One distinct value carries no interior at all. Fall back to the two outer
        # segments meeting at it, which still says which side of it a molecule falls on.
        q1 = q3 = float(vals[0])
        interior = np.full(scores.shape, 0.5)
    else:
        # Inverse interpolation: the probabilities at which the reference's own ECDF
        # crosses 0.25 and 0.75.
        q1 = float(np.interp(0.25, midranks, vals))
        q3 = float(np.interp(0.75, midranks, vals))
        interior = np.interp(scores, vals, midranks)

    # `np.where`, never boolean assignment: `np.interp` of a scalar returns a numpy scalar,
    # which has no item assignment.
    ranks = interior

    if q3 < 1.0 and q3 >= q1:
        upper = 0.75 + 0.25 * (scores - q3) / (1.0 - q3)
        ranks = np.where(scores > q3, upper, ranks)

    if q1 > 0.0 and q3 >= q1:
        lower = 0.25 * scores / q1
        ranks = np.where(scores < q1, lower, ranks)

    return np.clip(ranks, 0.0, 1.0)

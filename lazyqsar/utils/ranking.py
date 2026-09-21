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

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

# The reference percentiles the rank scale is anchored on, and the rank each receives.
#
# Tail percentiles rather than quartiles. A bioactivity model's product is the top of the
# list, and resolving inactives is worth nothing: measured across six antimicrobial models,
# the top 1% of a screened library falls between the reference's p98.9 and p100, while
# anchoring on the quartiles spends three quarters of the axis below the library's median.
# On one of those models the shipped quartile scale gave the 113 best-scoring compounds a
# span of 0.003 of the axis -- effectively one value; these anchors give them 0.096.
#
# Each step is a 10x shrink of the tail, so 0.50 reads "beats 90% of drug-like space",
# 0.65 "beats 99%", 0.75 "beats 99.9%".
TAIL_ANCHORS = ((50.0, 0.25), (90.0, 0.50), (99.0, 0.65), (99.9, 0.75))

# Where p95 of the model's own out-of-fold actives lands, when it clears the last reference
# anchor. Past the reference's p99.9 the library is too sparse to resolve anything -- for a
# selective model the whole top 1% of a screen can sit inside a single knot interval -- so
# the top of the scale is pinned on molecules whose labels are known instead.
ACTIVES_RANK = 0.95

# The decision boundary, as a point on this scale. 0.65 is the reference's p99, so "a hit
# beats 99% of drug-like space" and the generic hit rate is 1% by construction, rather than
# whatever a balanced-accuracy search against assay-matched inactives happens to produce
# (measured: 0.60 to 0.99 across six real models, i.e. most of drug-like space called active).
#
# Deliberately at or below the last reference anchor, so inverting it never touches the
# out-of-fold actives anchor. The cutoff is then a pure function of the reference library and
# does not wobble on a model fitted from 24 actives. Keep any retuned value <= 0.75.
DECISION_RANK = 0.65

# The share of the reference library the cutoff is *defined* to admit, derived from the
# anchors rather than written down again: DECISION_RANK sits on a TAIL_ANCHORS percentile,
# so this is 1 - p/100 for that anchor. Computed so that moving an anchor or the constant
# cannot leave a stale 0.01 behind. None if DECISION_RANK is ever moved off an anchor.
DECISION_HIT_RATE = next(
    (1.0 - q / 100.0 for q, rank in TAIL_ANCHORS if rank == DECISION_RANK), None
)

# The value of the ``decision_cutoff_source`` metadata key that means "this checkpoint's
# cutoff is :data:`DECISION_RANK` inverted against its reference library". `binary` and the
# artifacts' `predict` threshold on the cutoff only when they see this, so a checkpoint
# fitted before the cutoff moved onto the rank scale keeps the labels it always produced.
#
# Lives here rather than in `ensemble.combine` because the inference artifacts branch on it
# and are documented as numpy-only; this module is the shared floor both tiers already
# import.
DECISION_CUTOFF_SOURCE = "reference_rank"

# Raised when there is no reference library to rank against. Here, not in
# `ensemble.combine` or `agnostic`, for the same reason as DECISION_CUTOFF_SOURCE above:
# the numpy-only inference artifacts raise it, and `lazyqsar/artifacts/` must not import
# anything heavier than this module -- it used to reach up into `lazyqsar.agnostic`.
#
# One message, covering both ways of getting here. There were two, and the artifact raised
# whichever one its module happened to import, so a task-level checkpoint was told to pass
# `reference_X=` (not applicable) and a saved descriptor-matrix model was told to upgrade
# (already current). Naming both conditions is shorter than guessing between them.
NO_REFERENCE_MESSAGE = (
    "predict_rank needs a reference library: this model has no reference-library rank. "
    "`rank` is a position against a fixed library of drug-like molecules, so there is "
    "nothing to measure against.\n"
    "Two ways to end up here:\n"
    "  - Fitted with lazyqsar < 3.6. Those checkpoints carry a percentile against their "
    "own training set instead, which is not comparable and is not reported as though it "
    "were. Refit with lazyqsar>=3.6.\n"
    "  - Fitted through the descriptor-matrix entry point (`LazyClassifier`), which never "
    "sees the molecules and so cannot featurize the library itself. Refit passing "
    "`reference_X=` (or `reference_h5_file=`): get the molecule list with `lazyqsar "
    "reference smiles --output ref.csv` (or `lazyqsar.reference.reference_smiles()`), "
    "featurize it in that order, and pass the matrix.\n"
    "Either way `proba`, `logit`, `lift`, `score` and `binary` are unaffected."
)


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


def _collapse_anchors(points):
    """Sort *points* by probability and keep one per distinct value, carrying the top rank.

    ``np.interp`` needs a strictly increasing ``xp``, and a narrow reference library can put
    two anchor percentiles on the same probability -- one measured library spans 0.17 to
    0.26 across its whole range. Keeping the higher rank leaves the map monotone and the
    inverse single-valued: a rank between the two collapsed values inverts to that one
    probability, which is the honest answer, because the library cannot tell those ranks
    apart.
    """
    out = []
    for x, y in sorted(points):
        if out and x <= out[-1][0]:
            out[-1] = (out[-1][0], max(out[-1][1], y))
        else:
            out.append((x, y))
    xs = np.array([p[0] for p in out], dtype=np.float64)
    ys = np.array([p[1] for p in out], dtype=np.float64)
    return xs, ys


def reference_anchor_table(knots=None, prepared=None, anchor_high=None):
    """The (probability -> rank) anchor points, and whether the actives anchor was used.

    One table, three readers: :func:`rank_from_reference` interpolates it forward,
    :func:`proba_from_reference_rank` interpolates it backward, and the fit-time anchor
    decision reads *used* instead of re-deriving the comparison. Holding them to one
    definition is what stops the forward and inverse maps drifting apart.

    Parameters
    ----------
    knots : array_like, optional
        Reference pooled probabilities. Ignored when *prepared* is given.
    prepared : tuple, optional
        The ``(vals, midranks)`` pair from :func:`prepare_knots`.
    anchor_high : float, optional
        ``p95`` of the model's out-of-fold actives, or ``None``.

    Returns
    -------
    xs, ys : ndarray
        Probabilities and the rank each receives, both strictly increasing.
    used : bool
        Whether *anchor_high* entered the table. ``False`` means the model's known actives
        do not reach the top 0.1% of drug-like chemical space, so pinning
        :data:`ACTIVES_RANK` to them would fold the table back on itself -- a *lower*
        probability would receive a *higher* rank, breaking the one invariant ``rank`` must
        keep. The table then runs straight from the last reference anchor to certainty. Worth
        reporting rather than hiding: it is a statement about the model, not about the scale.

    Notes
    -----
    The anchor probabilities are read off the collapsed knots by inverse interpolation, the
    same way :func:`rank_from_reference` recovers the anchors themselves -- not with
    ``np.percentile`` on the raw array. The two disagree on a library with tied knots, and
    only the former is consistent with the ECDF the forward map interpolates.
    """
    vals, midranks = prepare_knots(knots) if prepared is None else prepared
    vals = np.asarray(vals, dtype=np.float64)
    midranks = np.asarray(midranks, dtype=np.float64)

    points = [(0.0, 0.0)]
    points += [
        (float(np.interp(q / 100.0, midranks, vals)), rank) for q, rank in TAIL_ANCHORS
    ]
    last_reference = points[-1][0]

    used = anchor_high is not None and last_reference < float(anchor_high) < 1.0
    if used:
        points.append((float(anchor_high), ACTIVES_RANK))
    points.append((1.0, 1.0))

    xs, ys = _collapse_anchors(points)
    return xs, ys, bool(used)


def proba_from_reference_rank(rank, knots=None, prepared=None, anchor_high=None):
    """The probability that :func:`rank_from_reference` maps to *rank*.

    Exact inverse, because both directions interpolate the one table from
    :func:`reference_anchor_table`, whose ``ys`` are strictly increasing.

    This is what turns a decision expressed on the rank scale -- :data:`DECISION_RANK` --
    into the probability the model actually thresholds on, and from there into every other
    unit the model reports.

    Parameters
    ----------
    rank : array_like or float
        Position on the rank scale.
    knots, prepared, anchor_high
        As :func:`reference_anchor_table`.

    Returns
    -------
    ndarray
        Pooled probabilities, same shape as *rank*.
    """
    xs, ys, _ = reference_anchor_table(knots, prepared, anchor_high)
    return np.interp(np.asarray(rank, dtype=np.float64), ys, xs)


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


def rank_from_reference(scores, knots=None, prepared=None, anchors=None):
    """Position against a reference library, anchored on its upper tail.

    One interpolation over :func:`reference_anchor_table`::

        ref p50   -> 0.25      "beats half of drug-like space"
        ref p90   -> 0.50      "beats 90%"
        ref p99   -> 0.65      "beats 99%"
        ref p99.9 -> 0.75      "beats 99.9%"
        p95 of the model's out-of-fold actives -> 0.95, when it clears ref p99.9

    Each step is a 10x shrink of the tail, so the axis is spent where a bioactivity model's
    product is. Measured across six antimicrobial models, the top 1% of a screened library
    sits between the reference's p98.9 and p100; the earlier quartile anchoring gave that
    top 1% a mean span of 0.069 of the axis against 0.215 here, and on one model the 113
    best-scoring compounds shared a span of 0.003 -- effectively a single value.

    The cost, taken deliberately: the bottom of the scale is flattened. Roughly a third of a
    generic library can land below rank 0.25. That is the right trade for a model whose job
    is a hit list, but it means ``rank`` carries little information about *how* inactive
    something is, and nothing downstream should read low ranks quantitatively.

    Above the reference's maximum the library has nothing left to say, which is what the
    out-of-fold actives anchor is for -- and when a model's actives do not reach the
    reference's p99.9 it is dropped rather than folding the map back on itself. See
    :func:`reference_anchor_table`.

    Parameters
    ----------
    scores : array_like or float
        Pooled probabilities. A scalar is accepted -- the decision cutoff is expressed as a
        rank through this function.
    knots, prepared
        The reference, as raw knots or as the output of :func:`prepare_knots`.
    anchors : tuple, optional
        ``(p05_inactives, p95_actives)``. **Only the second element shapes the scale.** The
        first is still measured and recorded, because where a model's inactives sit is worth
        reporting, but the tail table has no low anchor: the region it used to govern is now
        inside the ``0 -> ref p50`` segment.

    Returns
    -------
    ndarray
        Ranks in [0, 1], monotone in *scores*.
    """
    xs, ys, _ = reference_anchor_table(
        knots, prepared, None if anchors is None else anchors[1]
    )
    scores = np.asarray(scores, dtype=np.float64)
    return np.clip(np.interp(scores, xs, ys), 0.0, 1.0)

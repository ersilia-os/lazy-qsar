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

    One table, three readers: :func:`rank_from_reference` takes its anchors (and, above the
    reference maximum, its segments), :func:`proba_from_reference_rank` inverts the same
    scale, and the fit-time anchor decision reads *used* instead of re-deriving the
    comparison. Holding them to one definition is what stops the forward and inverse maps
    drifting apart. The probabilities in the table are the anchor points of the scale; the
    smooth curve between them is built from the reference percentiles, not from these
    straight segments.

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


# Rank gained per decade of tail past p99.9, inside the reference: half the 0.65 -> 0.75 pace.
_TOP_PACE = 0.05


def _pchip_slopes(x, y):
    """Node slopes of the monotone cubic through ``(x, y)`` (Fritsch-Carlson, as scipy's PCHIP).

    Here rather than imported because this module must stay numpy-only. With non-decreasing
    *y* the resulting curve never overshoots, so it is monotone wherever the data are.
    """
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros_like(y)
    for k in range(1, len(x) - 1):
        if delta[k - 1] * delta[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    for end, (h0, h1, d0, d1) in (
        (0, (h[0], h[1], delta[0], delta[1])),
        (-1, (h[-1], h[-2], delta[-1], delta[-2])),
    ):
        s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(s) != np.sign(d0):
            s = 0.0
        elif np.sign(d0) != np.sign(d1) and abs(s) > abs(3 * d0):
            s = 3 * d0
        d[end] = s
    return d


def _pchip_eval(x, y, d, t):
    """Evaluate the cubic Hermite curve with nodes ``(x, y)`` and slopes *d* at *t*."""
    k = np.clip(np.searchsorted(x, t, side="right") - 1, 0, len(x) - 2)
    h = x[k + 1] - x[k]
    s = (t - x[k]) / h
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    return h00 * y[k] + h10 * h * d[k] + h01 * y[k + 1] + h11 * h * d[k + 1]


def _tail_depth(q):
    """``-log10(1 - q)``: 0 at the bottom of the reference, 1 at p90, 2 at p99, 3 at p99.9."""
    return -np.log10(np.clip(1.0 - np.asarray(q, dtype=np.float64), 1e-15, 1.0))


def _smooth_scale(prepared, anchor_high):
    """Everything the smooth rank needs from a reference, or ``None`` to use the plain table.

    Returns ``(vals, midranks, t_nodes, r_nodes, slopes, upper_xs, upper_ys)``: the reference
    ECDF, the monotone curve through the tail anchors in tail-depth space, and the part of the
    anchor table above the reference maximum (unchanged from the piecewise scale).

    ``None`` when the reference cannot carry the curve: a single distinct value, or so few
    distinct values that the top of the ECDF does not reach past p99.9.
    """
    vals, midranks = (np.asarray(a, dtype=np.float64) for a in prepared)
    if len(vals) < 2:
        return None
    t_top = float(_tail_depth(midranks[-1]))
    t_anchors = [float(_tail_depth(q / 100.0)) for q, _ in TAIL_ANCHORS]
    if t_top <= t_anchors[-1]:
        return None
    xs, ys, used = reference_anchor_table(prepared=prepared, anchor_high=anchor_high)
    t_nodes = [0.0] + t_anchors
    r_nodes = [0.0] + [r for _, r in TAIL_ANCHORS]
    r_last = TAIL_ANCHORS[-1][1]
    if used and anchor_high < vals[-1]:
        # The actives anchor falls inside the reference: it becomes a node of the curve, at
        # its own percentile, and the reference maximum keeps the rank the table gives it.
        t_act = float(_tail_depth(np.interp(anchor_high, vals, midranks)))
        if t_anchors[-1] < t_act < t_top:
            t_nodes.append(t_act)
            r_nodes.append(ACTIVES_RANK)
        r_top = float(np.interp(vals[-1], xs, ys))
    else:
        # Past p99.9 the curve keeps rising, half the pace of the last anchors per decade of
        # tail, instead of flattening and piling the reference's top 0.1% onto one rank.
        # Capped at a third of the room left below the next point of the table (the actives
        # anchor, or certainty), so molecules beyond the reference keep room to spread.
        # A reference that already reaches probability 1.0 has no point above it: the next
        # point is certainty, the last one of the table. A score of exactly 1.0 then gets
        # `r_top`, the rank of the reference maximum, not 1.0.
        y_next = float(
            ys[min(np.searchsorted(xs, vals[-1], side="right"), len(ys) - 1)]
        )
        extra = t_top - t_anchors[-1]
        r_top = r_last + min(_TOP_PACE * extra, (y_next - r_last) / 3.0)
    if r_top <= r_nodes[-1]:
        return None
    t_nodes = np.array(t_nodes + [t_top])
    r_nodes = np.array(r_nodes + [r_top])
    slopes = _pchip_slopes(t_nodes, r_nodes)
    above = xs > vals[-1]
    upper_xs = np.concatenate([[vals[-1]], xs[above]])
    upper_ys = np.concatenate([[r_top], ys[above]])
    return vals, midranks, t_nodes, r_nodes, slopes, upper_xs, upper_ys


def _smooth_rank(scores, scale):
    """Forward map of the smooth scale; see :func:`rank_from_reference`."""
    vals, midranks, t_nodes, r_nodes, slopes, upper_xs, upper_ys = scale
    scores = np.asarray(scores, dtype=np.float64)
    q = np.interp(scores, vals, midranks)
    ranks = _pchip_eval(
        t_nodes, r_nodes, slopes, np.clip(_tail_depth(q), 0.0, t_nodes[-1])
    )
    # Below the reference minimum the ECDF has nothing to say; run linearly from 0 so those
    # molecules keep their order instead of tying at the curve's start.
    r_bottom = float(_pchip_eval(t_nodes, r_nodes, slopes, _tail_depth(midranks[0])))
    if vals[0] > 0:
        below = scores < vals[0]
        ranks = np.where(below, r_bottom * np.clip(scores, 0.0, None) / vals[0], ranks)
    above = scores > vals[-1]
    ranks = np.where(above, np.interp(scores, upper_xs, upper_ys), ranks)
    return np.clip(ranks, 0.0, 1.0)


def _smooth_inverse(rank, scale, grid=20001):
    """Inverse of :func:`_smooth_rank`, exact at the anchors and to ~1e-7 between them."""
    vals, midranks, t_nodes, r_nodes, slopes, upper_xs, upper_ys = scale
    rank = np.asarray(rank, dtype=np.float64)
    # The curve is monotone, so invert it on a dense grid that contains every node exactly.
    t_grid = np.union1d(np.linspace(0.0, t_nodes[-1], grid), t_nodes)
    r_grid = _pchip_eval(t_nodes, r_nodes, slopes, t_grid)
    r_grid = np.maximum.accumulate(r_grid)
    t = np.interp(rank, r_grid, t_grid)
    proba = np.interp(1.0 - 10.0 ** (-t), midranks, vals)
    r_bottom = float(_pchip_eval(t_nodes, r_nodes, slopes, _tail_depth(midranks[0])))
    if vals[0] > 0 and r_bottom > 0:
        proba = np.where(rank < r_bottom, rank / r_bottom * vals[0], proba)
    proba = np.where(rank > upper_ys[0], np.interp(rank, upper_ys, upper_xs), proba)
    return proba


def proba_from_reference_rank(rank, knots=None, prepared=None, anchor_high=None):
    """The probability that :func:`rank_from_reference` maps to *rank*.

    The inverse of the smooth scale: exact at every anchor (so :data:`DECISION_RANK`
    inverts to the reference's p99, as it always has) and accurate to about ``1e-7`` of rank
    between them. A reference too small to carry the curve falls back to the piecewise
    table, inverted exactly.

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
    prepared = prepare_knots(knots) if prepared is None else prepared
    scale = _smooth_scale(prepared, anchor_high)
    if scale is not None:
        return _smooth_inverse(rank, scale)
    xs, ys, _ = reference_anchor_table(prepared=prepared, anchor_high=anchor_high)
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

    The score is first read as its percentile ``q`` against the reference, then ``q`` is
    mapped through a smooth monotone curve that passes exactly through the tail anchors::

        ref p50   -> 0.25      "beats half of drug-like space"
        ref p90   -> 0.50      "beats 90%"
        ref p99   -> 0.65      "beats 99%"
        ref p99.9 -> 0.75      "beats 99.9%"
        p95 of the model's out-of-fold actives -> 0.95, when it clears ref p99.9

    The curve is a monotone cubic (PCHIP) in tail depth ``-log10(1 - q)``, so its slope has
    no corners. Up to 3.6 the scale was linear in probability between anchors, and every
    anchor was a corner: molecules piled up just below it and the density dropped sharply
    above (6.6x at 0.25 on the reference library), which reads as spikes in any histogram of
    ``rank``. The anchors, the order of molecules and the decision cutoff are unchanged; only
    the spacing between anchors moved. Below the reference minimum the scale runs linearly
    from 0, and above its maximum it is the piecewise table exactly as before.

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
    anchor_high = None if anchors is None else anchors[1]
    prepared = prepare_knots(knots) if prepared is None else prepared
    scale = _smooth_scale(prepared, anchor_high)
    if scale is not None:
        return _smooth_rank(scores, scale)
    # A reference too small to carry the curve keeps the piecewise table.
    xs, ys, _ = reference_anchor_table(prepared=prepared, anchor_high=anchor_high)
    scores = np.asarray(scores, dtype=np.float64)
    return np.clip(np.interp(scores, xs, ys), 0.0, 1.0)

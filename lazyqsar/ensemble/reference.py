"""Build the pooled reference distribution that ``rank`` is measured against.

``rank`` answers "where does this molecule sit relative to the model's training
distribution". Until v3.5.0 the ensemble answered it by averaging each descriptor's
answer, which is not the same question: an average of percentiles is not the percentile
of the average, so ``rank`` could order two molecules one way and ``proba`` the other.
Ranking the pooled probability against one pooled reference makes ``rank`` a monotone
view of ``proba`` instead, which is what every consumer already assumes it is.

The reference is the model's own out-of-fold training distribution -- the same thing each
head's knots already describe, one level up. Ranking against an external fixed library is
a different feature and is deliberately not what this builds.

Deliberately numpy-only, like the rest of ``lazyqsar.ensemble``: the knots it produces are
re-applied on the inference path, which runs without scikit-learn or RDKit.

What the reference is not
-------------------------
It is out-of-fold where deployment is full-data; the gating weights and
applicability-domain scores behind it are in-sample, so fewer descriptors are vetoed than
at inference; and it is computed from scikit-learn floats while inference reads ONNX
floats. Each of those shifts an absolute rank value slightly, and the tails most of all.
None of them touches the property the pooled reference exists to provide: both runtimes
read the *same* knots, so in both of them ``rank`` is a monotone function of ``proba``.
"""

import numpy as np

from ..utils.ranking import subsample_knots
from .combine import combine

_MAX_KNOTS = 10_000


def build_pooled_rank_knots(Y, R, S, A, spec, max_knots=_MAX_KNOTS):
    """Return ascending ECDF knots for the pooled probability, or None.

    Parameters
    ----------
    Y, R, S, A : ndarray or None
        Out-of-fold per-descriptor channels on the training rows, shaped
        ``(n_train, n_descriptors)`` and column-aligned to *spec*. ``A`` is the only one
        that may legitimately be ``None`` (no applicability domain).
    spec : EnsembleSpec
        The weighting parameters inference will use. Building the reference with any
        other weighting calibrates the ECDF against a distribution nothing produces.
    max_knots : int, default 10000
        Cap on stored knots.

    Returns
    -------
    ndarray or None
        Ascending float64 knots, or ``None`` when the inputs cannot support a reference
        that matches what inference computes.
    """
    if Y is None or R is None or S is None:
        # A reference built without the rank channel would be pooled with near-uniform
        # weights while inference pools with applicability-domain-conditioned ones. A
        # silently mis-calibrated ECDF is still monotone and still in [0, 1], so nothing
        # downstream could detect it; no reference at all is the safer failure.
        return None
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[0] == 0:
        return None

    p1 = combine(Y, R, S, A, spec=spec, outputs=("proba",)).values["proba"][:, 1]
    p1 = p1[np.isfinite(p1)]
    if p1.size == 0:
        return None
    return subsample_knots(np.sort(p1), max_knots=max_knots)


def build_pooled_score_knots(Y, R, S, A, spec, max_knots=_MAX_KNOTS):
    """Return the pooled probability -> pooled raw score map, or None.

    ``score`` reports the model's pre-calibration scale. Until now it got there by pooling
    the raw per-descriptor scores independently, which is not a monotone view of the pooled
    probability -- measured on the ChEMBL fixtures, ``score`` disagreed with ``proba`` about
    the order of 581 of 79,800 pairs.

    The cause is not calibration reordering anything: every head's calibrator is monotone
    and never reorders that head's own molecules. It is that the heads get *different*
    curves, so calibration changes how far apart each head's opinions sit -- how loudly it
    votes -- and a weighted average of differently-stretched monotone curves is not a
    monotone function of the weighted average of the originals. No way of pooling raw values
    fixes that; four were measured and the best still left 550 flipped pairs.

    So ``score`` is derived from the pooled probability instead, through this map. Monotone
    by construction, which makes it order-identical with ``proba`` -- while the values stay
    where they were, because the map is fitted to reproduce them (mean shift 0.0012,
    maximum 0.0318 on the fixtures).

    Parameters
    ----------
    Y, R, S, A : ndarray or None
        Out-of-fold per-descriptor channels on the training rows, as
        :func:`build_pooled_rank_knots` takes them. ``S`` carries the raw scores this map
        reproduces, so unlike the rank reference it is load-bearing rather than merely
        required for correct weighting.
    spec : EnsembleSpec
        The weighting inference will use, for the same reason as the rank reference.
    max_knots : int, default 10000
        Cap on stored knots.

    Returns
    -------
    tuple of (ndarray, ndarray), or None
        ``(probability, score)`` knots, probability ascending and score non-decreasing, or
        ``None`` when the inputs cannot support a map matching what inference computes.
    """
    if Y is None or R is None or S is None:
        return None
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[0] == 0:
        return None

    result = combine(Y, R, S, A, spec=spec, outputs=("proba", "score"))
    p1 = result.values["proba"][:, 1]
    s1 = result.values["score"][:, 1]
    keep = np.isfinite(p1) & np.isfinite(s1)
    p1, s1 = p1[keep], s1[keep]
    if p1.size == 0:
        return None

    order = np.argsort(p1, kind="stable")
    x = p1[order]
    # Running maximum, so the map is non-decreasing even where the raw scores disagree
    # locally with the probabilities. That disagreement is exactly what this removes.
    y = np.maximum.accumulate(s1[order])
    # One y per distinct x, or np.interp would pick arbitrarily between duplicates.
    x, first = np.unique(x, return_index=True)
    y = np.maximum.accumulate(y[first])
    if x.size < 2:
        return None
    # Thinned by shared positions rather than by two independent calls to
    # `subsample_knots`: the pairing is the map, and picking positions twice invites it to
    # drift apart the moment that helper's rule changes.
    if x.size > max_knots:
        idx = np.round(np.linspace(0, x.size - 1, max_knots)).astype(int)
        x, y = x[idx], y[idx]
    return x, y

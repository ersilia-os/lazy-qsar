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

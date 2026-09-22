"""Fold per-descriptor predictions into a model's final outputs.

Everything here is row-independent: sample *i*'s outputs depend only on row *i* of the
inputs. That is what makes the streaming prediction path safe to chunk, and it is worth
preserving — an output computed across the batch would silently become chunk-relative.

Deliberately numpy-only. No logging either: diagnostics come back as data on
:class:`CombineResult` and the caller decides whether to render them, which is what lets
one function serve both the fit-time and inference call sites.

How each output is pooled
-------------------------
``proba``, ``logit``, ``lift`` and ``binary`` come from one weighted sum in logit space.
``score`` is a weighted mean of the raw scores. ``rank`` is *not* a weighted mean of the
per-descriptor ranks -- averaging percentiles does not give the percentile of the average,
and the two orderings genuinely disagreed -- it is the pooled probability read off one
pooled out-of-fold reference, so it is a monotone view of ``proba``. Checkpoints fitted
before that reference existed fall back to the old weighted mean.

Weights, not just averages
--------------------------
Descriptors are not equally trustworthy, so they are combined with a per-sample weight
matrix rather than a flat mean. Each descriptor's weight blends a global skill term
(``max(0, mean(oof_auc, proxy_auc) - 0.5)``) with a per-sample reliability term derived
from its rank, and an applicability-domain score can veto a descriptor outright for
samples that fall outside its training distribution. With no AD scores available there is
nothing to condition on and the weights collapse to uniform.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..utils.ranking import (
    prepare_knots,
    rank_from_reference,
    score_from_knots,
)

OUTPUT_NAMES = ("proba", "logit", "rank", "score", "lift", "binary")

_DEFAULT_CUTOFF = 0.5
_EPS = 1e-7

POOLED_RANKER_KEY = "pooled_ranker"
POOLED_SCORER_KEY = "pooled_scorer"

# The only `pooled_ranker.source` that means a percentile against drug-like chemical
# space. Anything else -- notably "oof", written by v3.5.x -- is a training-set
# percentile and is not comparable with one.
REFERENCE_SOURCE = "reference_library"

NO_REFERENCE_MESSAGE = (
    "This checkpoint has no reference-library rank. `rank` is a percentile against a "
    "fixed library of drug-like molecules; checkpoints fitted before v3.6 carry a "
    "percentile against their own training set instead, which is not comparable and is "
    "not reported as though it were. Refit with lazyqsar>=3.6 to get `rank`, or use "
    "`proba`, `logit`, `lift`, `score` or `binary`, which are unaffected."
)


def read_pooled_rank_knots(metadata):
    """Pull the pooled rank reference out of a task-level ``metadata.json``, or None.

    One reader, because three call sites parse that file -- ``EnsembleSpec.from_metadata``
    and the two loaders in ``lazyqsar.qsar`` -- and a checkpoint that reaches one of them
    without the key silently falls back to the pre-v3.5.0 rank instead of failing.

    Parameters
    ----------
    metadata : dict or None
        Parsed ``metadata.json``. Every key is optional.

    Only a reference-library block is accepted. Checkpoints fitted before v3.6 carry
    out-of-fold knots under this same key, and the two are indistinguishable once read --
    both monotone, both in [0, 1] -- so reading an old one would report "beats 99% of
    drug-like space" about a training-set percentile. Those checkpoints are treated as
    having no reference, and asking them for ``rank`` raises.

    Parameters
    ----------
    metadata : dict or None
        Parsed ``metadata.json``. Every key is optional.

    Returns
    -------
    ndarray or None
        Ascending float64 knots, or ``None`` when the key is absent, empty, or not a
        reference-library block.
    """
    block = (metadata or {}).get(POOLED_RANKER_KEY) or {}
    knots = block.get("knots")
    if knots is None or len(knots) == 0:
        return None
    if block.get("source") != REFERENCE_SOURCE:
        return None
    return np.asarray(knots, dtype=np.float64)


def read_pooled_rank_anchors(metadata):
    """Pull the rank scale's tail anchors out of a task-level ``metadata.json``.

    ``(p05_inactives, p95_actives)`` in probability units, either of which may be ``None``.
    They cannot be derived at predict time -- only the reference knots travel in the
    checkpoint, not the out-of-fold molecules -- so they have to be stored.

    A checkpoint without them ranks exactly as one fitted before anchoring existed, because
    :func:`lazyqsar.utils.ranking.rank_from_reference` falls back per side.
    """
    block = (metadata or {}).get(POOLED_RANKER_KEY) or {}
    if block.get("source") != REFERENCE_SOURCE:
        return None
    low, high = block.get("anchor_low"), block.get("anchor_high")
    if low is None and high is None:
        return None
    return (
        None if low is None else float(low),
        None if high is None else float(high),
    )


def read_pooled_score_knots(metadata):
    """Pull the pooled score map out of a task-level ``metadata.json``, or None.

    Returns
    -------
    tuple of (ndarray, ndarray), or None
        ``(probability, score)`` knots, or ``None`` when the key is absent, empty or
        malformed -- in which case ``score`` keeps its pre-3.5.0 independent pooling.
    """
    block = (metadata or {}).get(POOLED_SCORER_KEY) or {}
    x, y = block.get("knots_x"), block.get("knots_y")
    if x is None or y is None or len(x) == 0 or len(x) != len(y):
        return None
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


@dataclass(frozen=True)
class EnsembleSpec:
    """Weighting parameters for one task, sliced to the active descriptors.

    Every sequence is already restricted to the descriptors that will actually be scored,
    in the same order as the columns of ``Y``/``R``/``S``/``A``. Slicing once here rather
    than re-indexing inside the weighting loops removes a whole class of
    index-misalignment bug.

    Attributes
    ----------
    descriptor_names : tuple of str
        Active descriptor names, column order.
    oof_aucs, proxy_aucs : tuple of (float or None), or None
        Per-descriptor skill estimates. ``None`` for the whole field means "no
        information"; a ``None`` entry means that descriptor has none.
    rank_error_curves : tuple of ((ndarray, ndarray) or None), or None
        Per-descriptor rank -> expected-error curves. Used only when every active
        descriptor has one; otherwise reliability falls back to ``|rank - 0.5| * 2``.
    ad_hard_cutoffs : tuple of float, or None
        Per-descriptor applicability-domain floor. A sample scoring below it has that
        descriptor's weight set to zero.
    population_prior : float
        Fraction of positives in the training set, the denominator for ``lift``.
    decision_cutoff : float
        Probability threshold for ``binary``.
    pooled_rank_knots : ndarray, or None
        Ascending ECDF knots over the pooled out-of-fold probability, built at fit time
        by :func:`lazyqsar.ensemble.reference.build_pooled_rank_knots`. When present,
        ``rank`` is this ECDF evaluated at the pooled probability. ``None`` -- the state
        of every checkpoint fitted before v3.5.0 -- keeps the earlier weighted mean of
        per-descriptor ranks. Not per-descriptor, so it is never sliced: it is written
        after the active set is settled and describes exactly that set.
    """

    descriptor_names: tuple[str, ...] = ()
    oof_aucs: tuple[float | None, ...] | None = None
    proxy_aucs: tuple[float | None, ...] | None = None
    rank_error_curves: tuple[tuple[np.ndarray, np.ndarray] | None, ...] | None = None
    ad_hard_cutoffs: tuple[float, ...] | None = None
    population_prior: float = 0.5
    decision_cutoff: float = _DEFAULT_CUTOFF
    pooled_rank_knots: np.ndarray | None = None
    pooled_rank_anchors: tuple | None = None
    pooled_score_knots: tuple[np.ndarray, np.ndarray] | None = None

    @classmethod
    def from_metadata(
        cls, metadata: dict, descriptor_names: Sequence[str]
    ) -> tuple[EnsembleSpec, list[str]]:
        """Build a spec from a task-level ``metadata.json``.

        Parameters
        ----------
        metadata : dict
            Contents of the task's ``metadata.json``. Every key is optional; a checkpoint
            saved before a field existed degrades to the same defaults the loaders used.
        descriptor_names : sequence of str
            Descriptor directories present on disk, in column order.

        Returns
        -------
        spec : EnsembleSpec
            Sliced to the active descriptors.
        active_names : list of str
            The subset of *descriptor_names* the model actually scores.

        Notes
        -----
        The skill term reads ``quality_aucs`` and falls back to ``oof_aucs``. Those are
        different numbers — ``quality = 2 * oof - train`` penalises descriptors that
        overfit — and the two historical loaders disagreed about which to use. The ONNX
        loader used ``quality``, so that is the deployed convention and the one kept here.
        """
        metadata = metadata or {}
        quality_map = metadata.get("quality_aucs", {}) or {}
        oof_map = metadata.get("oof_aucs", {}) or {}
        proxy_map = metadata.get("proxy_aucs", {}) or {}
        active_map = metadata.get("active_descriptors", {}) or {}
        cutoff_map = metadata.get("ad_hard_cutoffs", {}) or {}
        curve_map = metadata.get("rank_error_curves", {}) or {}

        active_names = [d for d in descriptor_names if active_map.get(d, True)]
        if not active_names:
            active_names = list(descriptor_names)

        curves = None
        if curve_map:
            curves = tuple(
                (np.asarray(curve_map[d][0]), np.asarray(curve_map[d][1]))
                if d in curve_map
                else None
                for d in active_names
            )

        prior = metadata.get("population_prior", 0.5)
        pooled_knots = read_pooled_rank_knots(metadata)
        pooled_anchors = read_pooled_rank_anchors(metadata)
        pooled_score = read_pooled_score_knots(metadata)
        # decision_cutoff is deliberately NOT read from metadata["decision_cutoff_proba"].
        # That learned, balanced-accuracy-optimal threshold exists in every checkpoint but
        # has never been used by either prediction path, and adopting it would move the
        # binary output on every deployed model. Whether to switch is its own change, with
        # its own held-out evaluation; until then binary means proba >= 0.5 as it always has.

        return (
            cls(
                descriptor_names=tuple(active_names),
                oof_aucs=tuple(
                    quality_map.get(d, oof_map.get(d, 1.0)) for d in active_names
                ),
                proxy_aucs=tuple(proxy_map.get(d) for d in active_names),
                rank_error_curves=curves,
                ad_hard_cutoffs=(
                    tuple(cutoff_map.get(d, 0.0) for d in active_names)
                    if cutoff_map
                    else None
                ),
                population_prior=float(prior if prior is not None else 0.5),
                pooled_rank_knots=pooled_knots,
                pooled_rank_anchors=pooled_anchors,
                pooled_score_knots=pooled_score,
            ),
            active_names,
        )


@dataclass
class CombineResult:
    """Outputs plus the intermediates a caller may want to report.

    Attributes
    ----------
    values : dict
        Requested outputs. Each is ``(n_samples, 2)`` — ``[negative, positive]`` — except
        ``binary``, which is ``(n_samples,)`` of int.
    weights : ndarray of shape (n_samples, n_descriptors)
        The per-sample weight matrix, rows summing to 1.
    base : ndarray of shape (n_descriptors,)
        Global skill score per descriptor, before per-sample adjustment.
    ranks : ndarray of shape (n_samples, n_descriptors)
        Ranks actually used, after the missing-rank fallback.
    diagnostics : list of dict, or None
        Per-descriptor summary rows, or ``None`` when no AD scores were supplied. Data
        only; rendering is the caller's job.
    """

    values: dict[str, np.ndarray] = field(default_factory=dict)
    weights: np.ndarray | None = None
    base: np.ndarray | None = None
    oof_percentiles: np.ndarray | None = None
    diagnostics: list[dict] | None = None


def build_weight_matrix(Y, R, A, spec: EnsembleSpec):
    """Return the normalised per-sample weight matrix and the global skill scores.

    Parameters
    ----------
    Y : ndarray of shape (n_samples, n_descriptors)
        Calibrated probabilities. Only its shape is used.
    R : ndarray or None
        Per-descriptor ranks, or ``None`` when unavailable.
    A : ndarray or None
        Applicability-domain scores, or ``None`` when the checkpoint has no AD artifacts.
        With no AD there is nothing to condition the weights on, so they go uniform.
    spec : EnsembleSpec
        Already sliced to the active descriptors.

    Returns
    -------
    W : ndarray of shape (n_samples, n_descriptors)
        Rows sum to 1.
    base : ndarray of shape (n_descriptors,)
        ``max(0, mean(oof_auc, proxy_auc) - 0.5)`` per descriptor, or all ones when no
        descriptor carries usable skill information.
    """
    B, D = Y.shape
    oof_aucs = spec.oof_aucs
    proxy_aucs = spec.proxy_aucs

    base_scores = []
    for j in range(D):
        vals = []
        if oof_aucs and oof_aucs[j] is not None:
            vals.append(float(oof_aucs[j]))
        if proxy_aucs and proxy_aucs[j] is not None:
            vals.append(float(proxy_aucs[j]))
        base_scores.append(max(0.0, float(np.mean(vals)) - 0.5) if vals else 0.0)
    base = np.array(base_scores, dtype=np.float64)
    if base.sum() == 0:
        base = np.ones(D, dtype=np.float64)

    if A is None:
        return np.full((B, D), 1.0 / D, dtype=np.float64), base

    curves = spec.rank_error_curves
    if R is not None:
        if curves and all(c is not None for c in curves):
            reliability = np.zeros((B, D), dtype=np.float64)
            for j, (r_knots, e_knots) in enumerate(curves):
                reliability[:, j] = 1.0 - np.interp(R[:, j], r_knots, e_knots)
        else:
            reliability = np.abs(R - 0.5) * 2
        W = 0.5 * base[np.newaxis, :] + 0.5 * reliability
    else:
        W = np.tile(base, (B, 1))

    if spec.ad_hard_cutoffs is not None:
        for j, cutoff in enumerate(spec.ad_hard_cutoffs):
            W[A[:, j] < cutoff, j] = 0.0

    # A sample vetoed on every descriptor has nothing left to weight with; fall back to
    # global skill rather than emitting NaN.
    all_ood = W.sum(axis=1) == 0
    if all_ood.any():
        W[all_ood] = base

    W /= W.sum(axis=1, keepdims=True)
    return W, base


def mask_rows(values: dict, rows) -> dict:
    """Overwrite *rows* with NaN in every output array, in place.

    Used for molecules that could not be parsed. Their descriptor row is all-NaN, and the
    imputer inside the exported preprocessor would otherwise replace it with the training
    median -- turning an unparseable string into an ordinary-looking score. Writing NaN
    keeps the row aligned with the input while making the gap unmissable.

    ``binary`` is integer-valued and cannot hold NaN, so it is promoted to float. That is
    the intended trade: a caller checking ``== 1`` still behaves correctly, and a NaN is
    visible where a 0 would have been silently wrong.
    """
    rows = np.asarray(rows, dtype=int)
    if rows.size == 0:
        return values
    for name, arr in list(values.items()):
        arr = np.asarray(arr)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64)
        arr[rows] = np.nan
        values[name] = arr
    return values


def _as_2d(X, name):
    if X is None:
        return None
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (n_samples, n_descriptors); got {arr.shape}"
        )
    return arr


def combine(Y, R=None, S=None, A=None, *, spec, outputs=OUTPUT_NAMES, cutoff=None):
    """Combine per-descriptor predictions into a model's outputs.

    Parameters
    ----------
    Y : array-like of shape (n_samples, n_descriptors)
        Calibrated probability of the positive class, per descriptor.
    R : array-like or None
        Per-descriptor ranks. ``None`` — meaning at least one descriptor could not supply
        them — falls back to a constant 0.5, which also flattens the reliability term.
    S : array-like or None
        Per-descriptor raw, pre-calibration scores. ``None`` falls back to *Y*.
    A : array-like or None
        Per-descriptor applicability-domain scores, or ``None`` when the checkpoint has
        none.
    spec : EnsembleSpec
        Weighting parameters, sliced to match the columns of the arrays above.
    outputs : sequence of str
        Which of :data:`OUTPUT_NAMES` to compute.
    cutoff : float, optional
        Probability threshold for ``binary``. Defaults to ``spec.decision_cutoff``.

    Returns
    -------
    CombineResult

    Notes
    -----
    Inputs are upcast to float64 before any arithmetic, so the combination itself is
    exact regardless of how the caller stored the channels. The upcast does not recover
    precision the caller has already spent: ``ensemble.channels`` accumulates in float32
    to halve peak memory, and that rounding does change the result. Measured across the
    28 scenarios in ``tests/_helpers/combine_cases.py``, float32 channels move ``logit``
    by up to 9.4e-07, ``lift`` by 6.7e-07, ``proba`` by 1.5e-07 and ``rank``/``score`` by
    3e-08, leaving ``binary`` identical. That is the accepted trade, and it is an order
    of magnitude below the ONNX export gap the same pipeline already carries -- but it is
    a real difference, not a free one.
    """
    unknown = set(outputs) - set(OUTPUT_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown output(s) {sorted(unknown)}; choose from {list(OUTPUT_NAMES)}"
        )

    Y = _as_2d(Y, "Y")
    R = _as_2d(R, "R")
    S = _as_2d(S, "S")
    A = _as_2d(A, "A")
    B, D = Y.shape
    if spec.descriptor_names and len(spec.descriptor_names) != D:
        raise ValueError(
            f"spec covers {len(spec.descriptor_names)} descriptors but Y has {D} columns"
        )
    W, base = build_weight_matrix(Y, R, A, spec)

    diagnostics = None
    if A is not None:
        names = spec.descriptor_names or tuple(str(j) for j in range(D))
        oof_aucs, proxy_aucs = spec.oof_aucs, spec.proxy_aucs
        diagnostics = [
            {
                "name": names[j],
                "oof_auc": float(oof_aucs[j])
                if (oof_aucs and oof_aucs[j] is not None)
                else float("nan"),
                "proxy_auc": float(proxy_aucs[j])
                if (proxy_aucs and proxy_aucs[j] is not None)
                else None,
                "ad_mean": float(A[:, j].mean()),
                "ad_std": float(A[:, j].std()),
                "ad_min": float(A[:, j].min()),
                "ad_max": float(A[:, j].max()),
                "weight_mean": float(W[:, j].mean()),
                "weight_std": float(W[:, j].std()),
                "vetoed": int((W[:, j] == 0).sum()),
                "pred_mean": float(Y[:, j].mean()),
            }
            for j in range(D)
        ]

    if R is None:
        R = np.full((B, D), 0.5, dtype=np.float64)

    values: dict[str, np.ndarray] = {}
    wanted = set(outputs)

    # `rank` is the pooled probability read off the pooled reference, so it needs the
    # log-odds sum too -- and an `outputs=("rank",)` call would otherwise leave `p1`
    # unbound.
    pooled_knots = getattr(spec, "pooled_rank_knots", None)
    pooled_rank = "rank" in wanted and pooled_knots is not None and len(pooled_knots)
    # `score` is the pooled probability read back onto the pre-calibration scale, for the
    # same reason `rank` is read off the pooled reference: pooling raw scores
    # independently is not a monotone view of the pooled probability, so the two could
    # order the same pair of molecules differently. Needs the log-odds sum too.
    score_knots = getattr(spec, "pooled_score_knots", None)
    pooled_score = "score" in wanted and score_knots is not None

    # proba, logit, lift and binary all derive from one weighted log-odds sum; computing
    # it once is what makes asking for several outputs cost no more than asking for one.
    needs_logit = (
        bool(wanted & {"proba", "logit", "lift", "binary"})
        or pooled_rank
        or pooled_score
    )
    if needs_logit:
        logits = np.log(np.clip(Y, _EPS, 1 - _EPS) / np.clip(1 - Y, _EPS, 1 - _EPS))
        l1 = (W * logits).sum(axis=1)
        p1 = 1.0 / (1.0 + np.exp(-l1))

    if "proba" in wanted:
        values["proba"] = np.vstack((1 - p1, p1)).T
    if "logit" in wanted:
        values["logit"] = np.vstack((-l1, l1)).T
    if "rank" in wanted:
        if not pooled_rank:
            # No silent fallback. The weighted mean of per-descriptor training percentiles
            # this used to compute answers a different question, and returning it under the
            # same name would mean one `rank` column meant "beats 99% of drug-like space"
            # on one checkpoint and "beats 99% of its own training set" on another, with
            # nothing in the output to tell them apart. An uncalibrated rank is worse than
            # an error, because it gets believed.
            raise ValueError(NO_REFERENCE_MESSAGE)
        # `rank_from_reference`, not `rank_from_knots`: the knots come from a reference
        # library, whose pooled probabilities stop well below 1 for any selective model
        # (measured: 0.065 to 0.334). Clamping would tie every active at exactly 1.0 and
        # break the invariant that rank orders molecules exactly as proba does.
        r1 = rank_from_reference(
            p1,
            prepared=prepare_knots(pooled_knots),
            anchors=getattr(spec, "pooled_rank_anchors", None),
        )
        values["rank"] = np.vstack((1 - r1, r1)).T
    if "score" in wanted:
        if pooled_score:
            s1 = score_from_knots(p1, score_knots)
        else:
            # Pre-3.5.0 checkpoints carry no score map, so they keep the independent
            # pooling and the ordering that comes with it. `S is None` on top of that
            # means at least one descriptor could not supply a raw score, and the
            # documented fallback is the calibrated one. Resolved here rather than with an
            # `S = Y.copy()` beside the other upcasts, because that copy ran for every
            # call -- a full (n_samples, n_descriptors) float64 array allocated and never
            # read unless `score` was among the outputs, which for the default `proba`
            # request it never is.
            s1 = (W * (Y if S is None else S)).sum(axis=1)
        values["score"] = np.vstack((1 - s1, s1)).T
    if "lift" in wanted:
        prior = spec.population_prior
        values["lift"] = np.column_stack(
            [(1 - p1) / max(1 - prior, _EPS), p1 / max(prior, _EPS)]
        )
    if "binary" in wanted:
        threshold = spec.decision_cutoff if cutoff is None else float(cutoff)
        values["binary"] = (p1 >= threshold).astype(int)

    return CombineResult(
        values=values,
        weights=W,
        base=base,
        oof_percentiles=R,
        diagnostics=diagnostics,
    )

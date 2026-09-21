"""Fold per-descriptor predictions into a model's final outputs.

Everything here is row-independent: sample *i*'s outputs depend only on row *i* of the
inputs. That is what makes the streaming prediction path safe to chunk, and it is worth
preserving — an output computed across the batch would silently become chunk-relative.

Deliberately numpy-only. No logging either: diagnostics come back as data on
:class:`CombineResult` and the caller decides whether to render them, which is what lets
one function serve both the fit-time and inference call sites.

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

OUTPUT_NAMES = ("proba", "logit", "rank", "score", "lift", "binary")

_DEFAULT_CUTOFF = 0.5
_EPS = 1e-7


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
    """

    descriptor_names: tuple[str, ...] = ()
    oof_aucs: tuple[float | None, ...] | None = None
    proxy_aucs: tuple[float | None, ...] | None = None
    rank_error_curves: tuple[tuple[np.ndarray, np.ndarray] | None, ...] | None = None
    ad_hard_cutoffs: tuple[float, ...] | None = None
    population_prior: float = 0.5
    decision_cutoff: float = _DEFAULT_CUTOFF

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
    ranks: np.ndarray | None = None
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
    if S is None:
        S = Y.copy()

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

    # proba, logit, lift and binary all derive from one weighted log-odds sum; computing
    # it once is what makes asking for several outputs cost no more than asking for one.
    needs_logit = bool(wanted & {"proba", "logit", "lift", "binary"})
    if needs_logit:
        logits = np.log(np.clip(Y, _EPS, 1 - _EPS) / np.clip(1 - Y, _EPS, 1 - _EPS))
        l1 = (W * logits).sum(axis=1)
        p1 = 1.0 / (1.0 + np.exp(-l1))

    if "proba" in wanted:
        values["proba"] = np.vstack((1 - p1, p1)).T
    if "logit" in wanted:
        values["logit"] = np.vstack((-l1, l1)).T
    if "rank" in wanted:
        r1 = (W * R).sum(axis=1)
        values["rank"] = np.vstack((1 - r1, r1)).T
    if "score" in wanted:
        s1 = (W * S).sum(axis=1)
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
        values=values, weights=W, base=base, ranks=R, diagnostics=diagnostics
    )

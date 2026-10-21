"""SMILES-level classification: ``LazyClassifierQSAR``, its ONNX wrapper, ``LazyQSAR``.

The recipe shared with regression -- descriptor screen, one agnostic model and one
applicability domain per descriptor, pruning, per-sample combination, checkpoints -- is
in :mod:`lazyqsar.qsar_base`. This module adds what only classification has: the
reference library behind ``rank``, the decision cutoff on that scale, the pooled score
map, and the out-of-fold diagnostics.
"""

import json
import os

import numpy as np

from .ensemble import (
    OUTPUT_NAMES,
    EnsembleSpec,
    build_pooled_score_knots,
    combine,
)
from .registry import (  # noqa: F401  (re-exported for backwards compatibility)
    DESCRIPTOR_TYPES,
    DESCRIPTORS_MODE,
    get_descriptor_type,
)
from .ensemble.combine import (
    DECISION_CUTOFF_SOURCE,
    read_decision_cutoff,
    read_pooled_rank_knots,
    read_pooled_score_knots,
)
from .qsar_base import (  # noqa: F401  (re-exported: callers have always looked here)
    _ArtifactWrapperBase,
    _EnsemblePredictMixin,
    _QSARBase,
    _check_task,
    _descriptor_dirs,
    _empty_channels,
    _load_descriptor_stack,
    _positions_to_load,
    _save_maybe_zipped,
    _smiles_md5,
    _stack_channels,
    _unpacked,
    as_smiles_list,
    invalid_smiles_indices,
    validate_smiles,
)
from .utils.archives import read_json
from .utils.splits import check_trainable
from .utils.logging import logger
from .utils.ranking import (
    DECISION_HIT_RATE,
    DECISION_RANK,
    prepare_knots,
    proba_from_reference_rank,
    rank_from_reference,
    reference_anchor_table,
    score_from_knots,
    subsample_knots,
)

# Matches `combine`'s guard on the lift denominator, so the reported cutoff is the number
# the `lift` output is compared against.
_LIFT_EPS = 1e-7

# Descriptor pruning on out-of-fold AUC: below the floor, or more than the gap below the
# best descriptor.
_PRUNE_FLOOR = 0.55
_PRUNE_GAP = 0.10


def _has_onnx(descriptor_dir):
    """Whether a saved descriptor directory contains any ONNX graph.

    The search is recursive because the graphs live one level down, under ``batch_0/``
    and its siblings — a descriptor directory itself holds only ``featurizer.json``,
    ``metadata.json`` and those batch folders. A non-recursive check therefore never
    matched, and every checkpoint silently loaded through the raw path.
    """
    for _, _, files in os.walk(descriptor_dir):
        if any(f.endswith(".onnx") for f in files):
            return True
    return False


def _quality_weight(oof_auc, train_auc):
    """The skill term the ensemble weights descriptors by: ``2*oof - train``, clipped.

    The subtraction penalises a descriptor that fits its training data far better than its
    held-out folds. Two ways it used to misbehave:

    *Unclipped*, it is not bounded by 1. ``train < oof`` happens on small data, and a
    failed training-AUC computation used to stand in as 0.5, which on a strong descriptor
    gives ``2*0.9 - 0.5 = 1.3`` -- so the failure earned roughly double the weight of a
    genuinely good descriptor. The clip makes the term mean what its name says.

    *With an unknown input*, there is no honest gap to charge, so the descriptor is
    weighted on its out-of-fold AUC alone rather than on a guess.
    """
    if oof_auc is None:
        return None
    if train_auc is None:
        return float(np.clip(oof_auc, 0.0, 1.0))
    return float(np.clip(2.0 * oof_auc - train_auc, 0.0, 1.0))


def _optional(fn, X):
    """Call *fn(X)* and take the positive column, or return None if it is unavailable.

    Not every head exposes every channel, and a checkpoint saved before one existed has
    no data for it. The caller treats a single None as "this channel is unavailable for
    the whole ensemble" and falls back accordingly.

    A genuine failure — a corrupt ONNX graph, a shape mismatch — looks the same here, and
    silently degrades the ensemble rather than raising, so it is logged.
    """
    try:
        return fn(X)[:, 1]
    except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
        logger.warning(
            f"{getattr(fn, '__name__', fn)} unavailable, falling back: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


class _ClassifierPredictMixin(_EnsemblePredictMixin):
    """The six ``predict_*`` methods, implemented once over :func:`combine`."""

    def _wanted_outputs(self, spec) -> tuple:
        # `combine` raises for `rank` on a checkpoint with no reference library, and this
        # call asks for every output at once -- so without this the raise would take
        # `predict_proba` down with it on every checkpoint fitted before v3.6.
        # `predict_rank` raises it deliberately instead.
        if getattr(spec, "pooled_rank_knots", None) is None:
            return tuple(n for n in OUTPUT_NAMES if n != "rank")
        return OUTPUT_NAMES

    def predict_proba(self, smiles_list):
        """Calibrated ensemble probabilities, shape (n_samples, 2)."""
        return self._combined(smiles_list).values["proba"]

    def predict_logit(self, smiles_list):
        """Log-odds of the calibrated probabilities, shape (n_samples, 2)."""
        return self._combined(smiles_list).values["logit"]

    def predict_rank(self, smiles_list):
        """Position against the reference library, shape (n_samples, 2).

        Measured against a fixed 50,000-molecule sample of drug-like chemical space, not
        against this model's own training set, which is what versions before 3.6 reported.

        **Not a raw percentile.** The probability is read through the anchor table in
        :mod:`lazyqsar.utils.ranking`, which pins four reference percentiles to fixed
        ranks so that the same rank means the same thing on every model:

        =====================  =====
        beats this much of     rank
        the reference library
        =====================  =====
        50%                    0.25
        90%                    0.50
        99%                    0.65
        99.9%                  0.75
        =====================  =====

        So ``0.65`` -- not ``0.99`` -- is the molecule that beats 99% of drug-like space,
        and it is also :data:`~lazyqsar.utils.ranking.DECISION_RANK`, the cutoff
        ``predict`` uses. Above 0.75 the scale runs on to the model's own strongest
        out-of-fold actives at 0.95 and to certainty at 1.0, so values there say "beyond
        anything the reference contains" rather than naming a percentile. Between the
        anchors the scale follows a smooth monotone curve through the reference percentiles
        (since 3.7; linear in probability before), so a histogram of ``rank`` has no steps
        at the anchors. The bottom half of the scale is still deliberately compressed: half
        of a generic library lands below 0.25.

        Raises ``ValueError`` on a checkpoint that carries no reference library. The other
        five outputs still work; only this one changed meaning.
        """
        values = self._combined(smiles_list).values
        if "rank" not in values:
            from .utils.ranking import NO_REFERENCE_MESSAGE

            raise ValueError(NO_REFERENCE_MESSAGE)
        return values["rank"]

    def predict_score(self, smiles_list):
        """Weighted raw (pre-calibration) scores, shape (n_samples, 2)."""
        return self._combined(smiles_list).values["score"]

    def predict_lift(self, smiles_list):
        """Probability over the population prior, shape (n_samples, 2)."""
        return self._combined(smiles_list).values["lift"]

    def predict(self, smiles_list, threshold=None, cutoff=None):
        """Binary labels, shape (n_samples,).

        With no threshold given this returns :func:`combine`'s own ``binary`` output, which
        thresholds on the checkpoint's decision cutoff -- a fixed point on the reference rank
        scale, so the model calls about 1% of drug-like chemical space active. A checkpoint
        without one falls back to ``proba >= 0.5``, as this method always did.

        That matters because this method used to recompute the label from ``proba`` against a
        hardcoded 0.5, ignoring the ``binary`` that :meth:`_combined` had just computed. The
        Python API and ``--predict_type binary`` could therefore disagree.

        ``cutoff`` is accepted as an alias for ``threshold``: the two classes this mixin
        replaced spelled the same argument differently, and both spellings are in use.
        Passing either overrides the checkpoint's cutoff.
        """
        if cutoff is not None:
            threshold = cutoff
        if threshold is None:
            return self._combined(smiles_list).values["binary"]
        p1 = self._combined(smiles_list).values["proba"][:, 1]
        labels = p1 >= threshold
        # NaN >= threshold is False, which would quietly turn an unparseable molecule into
        # a confident negative -- the one outcome this path must not produce. Promote to
        # float and carry the NaN through, exactly as ensemble.combine.mask_rows does, and
        # only when there is something to carry, so the usual return stays integer.
        if np.isnan(p1).any():
            out = labels.astype(float)
            out[np.isnan(p1)] = np.nan
            return out
        return labels.astype(int)


def _rank_band(ranks):
    """Quartiles of a set of ranks, or ``None`` when the class is empty.

    Quartiles rather than a mean: the point is to say where known molecules *sit*, and a
    band a user can compare a single compound against is more useful than a centre.
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    ranks = ranks[np.isfinite(ranks)]
    if ranks.size == 0:
        return None
    p25, p50, p75 = (float(v) for v in np.percentile(ranks, [25, 50, 75]))
    return {"n": int(ranks.size), "rank_p25": p25, "rank_p50": p50, "rank_p75": p75}


def _anchors_from_metadata(meta):
    """The stored anchor record, in the shape a fitted model holds it."""
    block = (meta or {}).get("pooled_ranker") or {}
    if block.get("anchor_low") is None and block.get("anchor_high") is None:
        return None
    return {
        k: block.get(k)
        for k in (
            "anchor_low",
            "anchor_high",
            "anchor_low_used",
            "anchor_high_used",
            "n_actives",
            "n_inactives",
        )
    }


def _anchor_pair(anchors):
    """``(low, high)`` from the stored anchor record, honouring the usability flags.

    ``high`` is ``None`` when p95 of the out-of-fold actives did not clear the reference's
    p99.9, so the scale runs straight from there to certainty. ``low`` is always ``None``:
    the tail table has no low anchor, and `rank_from_reference` reads only the second
    element. Kept as a pair because that is the shape stored in every checkpoint.
    """
    if not anchors:
        return None
    low = anchors.get("anchor_low") if anchors.get("anchor_low_used") else None
    high = anchors.get("anchor_high") if anchors.get("anchor_high_used") else None
    return None if low is None and high is None else (low, high)


def _decision_cutoff_from_metadata(meta):
    """The fitted model's cutoff record, rebuilt from a task-level ``metadata.json``.

    Shaped like :meth:`LazyClassifierQSAR._build_decision_cutoff`'s return so that a loaded
    model and a freshly fitted one carry the same attribute. ``None`` for a checkpoint whose
    cutoff is not on the reference rank scale, which keeps ``binary`` at ``proba >= 0.5``.
    """
    proba = read_decision_cutoff(meta)
    if proba is None:
        return None
    return {
        "rank": meta.get("decision_cutoff_rank"),
        "proba": proba,
        "logit": meta.get("decision_cutoff_logit"),
        "lift": meta.get("decision_cutoff_lift"),
        "source": DECISION_CUTOFF_SOURCE,
    }


def _spec_from_attributes(
    names,
    active_indices,
    oof_aucs,
    proxy_aucs,
    curves,
    cutoffs,
    prior,
    pooled_knots=None,
    pooled_score_knots=None,
    pooled_anchors=None,
    decision_cutoff=None,
):
    """Build an :class:`EnsembleSpec` from the per-descriptor attribute lists.

    The lists are indexed by *full* descriptor position; the spec is sliced down to the
    active ones so the weighting code never has to re-index. *pooled_knots* and
    *pooled_score_knots* are the arguments that are not per-descriptor: each describes the
    pooled probability of the active set as a whole, so they pass through unsliced -- as
    do *pooled_anchors*, which pin the top of the scale, and *decision_cutoff*.

    *decision_cutoff* of ``None`` leaves :class:`EnsembleSpec`'s own default in place, which
    is the historical ``proba >= 0.5``. Callers that only want ``proba`` need not supply it.
    """

    def sliced(seq):
        return tuple(seq[i] for i in active_indices) if seq else None

    extra = (
        {} if decision_cutoff is None else {"decision_cutoff": float(decision_cutoff)}
    )
    return EnsembleSpec(
        descriptor_names=tuple(names[i] for i in active_indices),
        oof_aucs=sliced(oof_aucs),
        proxy_aucs=sliced(proxy_aucs),
        rank_error_curves=sliced(curves),
        ad_hard_cutoffs=sliced(cutoffs),
        population_prior=prior,
        pooled_rank_knots=pooled_knots,
        pooled_score_knots=pooled_score_knots,
        pooled_rank_anchors=pooled_anchors,
        **extra,
    )


class ArtifactWrapper(_ClassifierPredictMixin, _ArtifactWrapperBase):
    """
    ONNX inference wrapper for a multi-descriptor LazyClassifierQSAR model.

    Returned by ``LazyClassifierQSAR.load()`` / ``load_onnx()``. Holds one ONNX artifact
    per descriptor type and exposes the same predict_* API as the training-time
    ``LazyClassifierQSAR``. The ensemble spec comes from
    :meth:`EnsembleSpec.from_metadata`, the same reader the inference runner uses.
    """

    task = "classification"

    def _wanted_channels(self) -> set:
        # `s` only when this checkpoint has no pooled score map. With one, `score` is
        # read off the pooled probability, so asking for the raw channel would run every
        # graph a second time for a value nothing reads.
        if self.spec.pooled_score_knots is not None:
            return {"y", "r", "a"}
        return {"y", "r", "s", "a"}

    @property
    def oof_aucs(self):
        """Per-descriptor skill the ensemble weights by (quality AUC), all descriptors."""
        meta = self.metadata
        quality = meta.get("quality_aucs") or {}
        oof = meta.get("oof_aucs") or {}
        return [quality.get(d, oof.get(d, 1.0)) for d in self.descriptor_types]

    @property
    def population_prior(self):
        return self.spec.population_prior

    @property
    def pooled_rank_knots(self):
        return self.spec.pooled_rank_knots

    @property
    def pooled_score_knots(self):
        return self.spec.pooled_score_knots

    @property
    def pooled_rank_anchors(self):
        return self.spec.pooled_rank_anchors

    @property
    def decision_cutoff(self):
        return read_decision_cutoff(self.metadata)


class LazyClassifierQSAR(_ClassifierPredictMixin, _QSARBase):
    """
    SMILES-aware binary classifier with built-in descriptor computation.

    Trains one ``LazyClassifier`` per descriptor type and combines their
    predictions via an AUC-weighted ensemble that accounts for per-sample
    prediction confidence (rank-based reliability).

    Parameters
    ----------
    mode : str
        Descriptor set to use:
          - ``"fast"``  — Morgan fingerprints only (no DL models)
          - ``"slow"``  — CDDD, Chemeleon, CLAMP, Morgan, RDKit

    Attributes (after fit)
    ----------------------
    descriptor_types : list[str]
        Names of the descriptor types used.
    classifiers_ : list[LazyClassifier]
        One fitted ``LazyClassifier`` per descriptor.
    oof_aucs_ : list[float | None]
        Per-descriptor OOF AUC.
    proxy_aucs_ : list[float | None]
        Per-descriptor proxy AUC (from a held-out split after fit).
    population_prior_ : float
        Fraction of positives in the training set.
    """

    task = "classification"
    _skill_attrs = ("oof_aucs_", "train_aucs_", "quality_aucs_", "proxy_aucs_")
    _metric_label = "AUC"
    _prune_floor = _PRUNE_FLOOR
    _prune_gap = _PRUNE_GAP

    def _model_class(self):
        from .agnostic import LazyClassifier

        return LazyClassifier

    @staticmethod
    def _model_skill(model):
        return model.oof_auc_, model.train_auc_

    @staticmethod
    def _quality(oof, train):
        return _quality_weight(oof, train)

    @staticmethod
    def _rank_error_curve(model, X, y, channels):
        """Rank -> error curve from training predictions (20 knots, windowed).

        Training-set predictions are optimistic but sufficient as a relative reliability
        signal: high ``|rank - 0.5|`` should map to low error.
        """
        try:
            _p = model.predict_proba(X=X)[:, 1]
            _r = model._oof_percentile(X=X)[:, 1]
            _err = np.abs(_p - y.astype(float))
            _sidx = np.argsort(_r)
            _rs, _es = _r[_sidx], _err[_sidx]
            _nk = min(20, len(_rs))
            _ki = np.round(np.linspace(0, len(_rs) - 1, _nk)).astype(int)
            _hw = max(1, len(_rs) // (_nk * 2))
            _r_knots = _rs[_ki]
            _e_knots = np.array(
                [_es[max(0, k - _hw) : k + _hw + 1].mean() for k in _ki]
            )
            return (_r_knots.tolist(), _e_knots.tolist())
        except Exception:
            return None

    def _before_descriptor_fits(self) -> None:
        # Before any training: the reference is read only at the end of fit, and a missing
        # one would otherwise cost the whole fit to discover.
        from .reference import require_fetchable

        require_fetchable(self.descriptor_types)

    def fit(self, smiles_list, y, precomputed=None, validate=True):
        """Fit one classifier per applicable descriptor.

        Parameters
        ----------
        smiles_list : list of str
            Training compounds.
        y : array-like
            Binary labels.
        precomputed : dict, optional
            ``{descriptor_name: feature_matrix}`` aligned with *smiles_list*. Lets a
            caller that has already featurized skip doing it again — the multi-task CLI
            fit computes each descriptor once over the union of every task and passes the
            per-task slice here, which is what keeps that one pass from becoming one per
            task.
        validate : bool
            Parse-check the SMILES. Callers that have already validated a superset can
            skip the repeat work.
        """
        smiles_list = as_smiles_list(smiles_list)
        # Clear any cached state from a previous fit.
        self._feature_cache.clear()
        self._ensemble_cache.clear()

        y = np.array(y, dtype=int)
        # Before the descriptor portfolio runs, for the same reason `require_fetchable`
        # is checked early: both conditions are knowable up front and both would
        # otherwise surface only after the whole featurization pass.
        check_trainable(y, where="LazyClassifierQSAR.fit")
        if validate:
            validate_smiles(smiles_list)
        n = len(smiles_list)
        pos_rate = float(y.mean())
        self.population_prior_ = pos_rate
        self.n_compounds_ = n
        self.n_actives_ = int((y == 1).sum())

        proxy = self._select_descriptors(smiles_list, y, precomputed)
        logger.rule("LazyClassifierQSAR")
        logger.info(
            f"mode={self.mode}  descriptors={self.descriptor_types}  "
            f"n={n:,}  pos_rate={pos_rate:.1%}"
        )
        _oof_channels, _train_ad, desc_rows = self._fit_descriptors(
            smiles_list, y, proxy
        )

        # Cleared before the build, not after: `_channels` passes `pooled_rank_knots_`
        # into the spec, so a refit that reused the attribute would calibrate the new
        # reference against the previous fit's knots.
        self.pooled_rank_knots_ = None
        self.reference_meta_ = None
        _stacked = self._oof_channels_stacked(_oof_channels, _train_ad)
        self.pooled_score_knots_ = self._build_pooled_score_map(_stacked)
        self.pooled_rank_knots_, self.reference_meta_ = (
            self._build_reference_rank_knots()
        )
        # Before the diagnostics, because the band is expressed on the scale the anchors
        # help define.
        self.pooled_rank_anchors_ = self._build_rank_anchors(_stacked, y)
        # Before the diagnostics too: `sensitivity_at_cutoff` needs the cutoff.
        self.decision_cutoff_ = self._build_decision_cutoff()
        self.oof_diagnostics_ = self._build_oof_diagnostics(_stacked, y)

        logger.rule()
        logger.descriptor_table(desc_rows)
        self._log_oof_diagnostics()

    def _log_oof_diagnostics(self):
        """Print the advisory numbers where the person who can act on them is watching."""
        diag = getattr(self, "oof_diagnostics_", None)
        if not diag:
            return
        for label in ("actives", "inactives"):
            band = diag.get(label)
            if band:
                logger.info(
                    f"{label:>9} (n={band['n']}) rank "
                    f"{band['rank_p25']:.3f} / {band['rank_p50']:.3f} / "
                    f"{band['rank_p75']:.3f}  (p25/p50/p75)"
                )
        auc = diag.get("screening_auc")
        if auc is not None:
            logger.info(
                f"  screening AUC {auc:.3f}  (actives vs the reference library -- "
                "whether actives rise above generic chemistry, which oof_auc does not ask)"
            )
        hit = diag.get("generic_hit_rate")
        sens = diag.get("sensitivity_at_cutoff")
        if hit is not None and sens is not None:
            # As a pair: the hit rate alone is now fixed by the cutoff and says nothing
            # about the model, while the two together are the screening trade-off.
            logger.info(
                f"  at rank {diag.get('decision_cutoff_rank')}, calls {hit:.2%} of "
                f"drug-like chemical space active, catching {sens:.1%} of its own "
                "known actives"
            )
        elif hit is not None:
            logger.info(
                f"  this model would call {hit:.2%} of drug-like chemical space active"
            )

    def _oof_channels_stacked(self, oof_channels, train_ad):
        """Stack the per-descriptor out-of-fold channels into ``(Y, R, S, A, spec)``.

        Extracted because two things need it -- the ``score`` map and the out-of-fold
        diagnostics -- and the stacking has to match what :meth:`_channels` does exactly.
        Two copies of it would be two chances to drift.

        Must run after the active set is settled: a stack over three descriptors does not
        describe the pooled probability of two.

        Returns ``None`` unless *every* active descriptor supplied out-of-fold channels.
        Anything assembled from a subset would still be monotone and still lie in range, so
        nothing downstream could tell it had been built against the wrong distribution.
        """
        active_indices = [i for i, a in enumerate(self.active_descriptors_) if a]
        if not active_indices:
            return None
        cols = [oof_channels[i] for i in active_indices]
        if any(c is None for c in cols):
            logger.debug(
                "No pooled out-of-fold stack: at least one active descriptor has no "
                "out-of-fold predictions."
            )
            return None

        Y = np.column_stack([c[0] for c in cols])
        R = np.column_stack([c[1] for c in cols])
        S = np.column_stack([c[2] for c in cols])
        A = (
            np.column_stack([train_ad[i] for i in active_indices])
            if len(train_ad) == len(self.active_descriptors_)
            else None
        )
        spec = _spec_from_attributes(
            self.descriptor_types,
            active_indices,
            self.quality_aucs_,
            self.proxy_aucs_,
            self._rank_error_curves_,
            self.ad_hard_cutoffs_,
            self.population_prior_,
        )
        return Y, R, S, A, spec

    def _build_pooled_score_map(self, stacked):
        """Learn the pooled out-of-fold map ``score`` is read back through.

        Out-of-fold, not reference-library, and deliberately so: this is a
        probability-to-raw-score map, a property of *this model's* calibrators rather than
        of any population, and the training data covers the high-probability region a
        generic library barely reaches.
        """
        if stacked is None:
            return None
        return build_pooled_score_knots(*stacked)

    def _build_rank_anchors(self, stacked, y):
        """Where the top of the rank scale is pinned: the model's own known actives.

        ``p95`` of the out-of-fold actives becomes rank 0.95, so the top of the scale is
        reachable whatever the model's probability ceiling. Without it the scale runs
        straight from the reference's p99.9 to 1.0, which assumes a model can reach
        certainty -- and most cannot: calibrators clip to the range seen in training and a
        calibrated probability is bounded by how rare actives are.

        The decision is delegated to :func:`reference_anchor_table` rather than repeated
        here, so the anchor this records is by construction the anchor the scale uses.

        ``p05`` of the inactives is still measured and recorded -- where a model's inactives
        sit is worth reporting -- but it no longer shapes the scale: the tail table has no
        low anchor, and the region it used to govern now lies inside the
        ``0 -> reference p50`` segment.
        """
        knots = getattr(self, "pooled_rank_knots_", None)
        if stacked is None or knots is None or not len(knots):
            return None

        p1 = combine(*stacked[:4], spec=stacked[4], outputs=("proba",)).values["proba"][
            :, 1
        ]
        y = np.asarray(y).ravel()
        if len(y) != len(p1):
            return None

        prepared = prepare_knots(knots)
        act, inact = p1[y == 1], p1[y == 0]
        high = float(np.percentile(act, 95)) if act.size else None
        low = float(np.percentile(inact, 5)) if inact.size else None

        _, _, high_used = reference_anchor_table(prepared=prepared, anchor_high=high)
        if high is not None and not high_used:
            last_reference = float(np.interp(0.999, prepared[1], prepared[0]))
            logger.warning(
                f"Upper rank anchor unused: p95 of the out-of-fold actives ({high:.3f}) "
                f"does not exceed the reference's 99.9th percentile "
                f"({last_reference:.3f}) -- this model's known actives do not reach the "
                "top 0.1% of drug-like chemical space. The top of the scale falls back to "
                "a straight line to certainty; see oof_diagnostics.screening_auc."
            )
        if act.size and act.size < 20:
            logger.warning(
                f"Only {act.size} out-of-fold actives anchor the top of the rank scale; "
                "it will move noticeably if the model is refitted."
            )

        return {
            "anchor_low": low,
            "anchor_high": high,
            # Recorded for symmetry with `anchor_high` and read by nothing: the tail table
            # has no low anchor. Always False so no reader can mistake it for live.
            "anchor_low_used": False,
            "anchor_high_used": bool(high_used),
            "n_actives": int(act.size),
            "n_inactives": int(inact.size),
        }

    def _build_decision_cutoff(self):
        """The decision boundary, fixed on the reference-library rank scale.

        ``DECISION_RANK`` is inverted against this model's reference knots, so the cutoff is
        "beat 99% of drug-like chemical space" on every model and the generic hit rate is 1%
        by construction. What it replaces is a balanced-accuracy threshold learned from
        out-of-fold scores: optimal against the *assay-matched* inactives it was fitted on,
        and on six real antimicrobial models it called 60% to 99% of drug-like space active.

        Every unit is derived here from the one probability, rather than each being averaged
        up the descriptor hierarchy independently. That is what makes them mutually
        consistent: pushing ``proba`` through the forward maps reproduces ``rank``, ``logit``
        and ``lift`` exactly.

        Cannot be aggregated, which is why it lives here and not per descriptor: the mean of
        two descriptors' p99 is a probability whose rank on the *pooled* scale is not 0.65.

        Returns
        -------
        dict or None
            ``{"rank", "proba", "logit", "lift", "source"}``, or ``None`` when this model has
            no reference library -- in which case ``binary`` keeps ``proba >= 0.5``.

        Notes
        -----
        ``lift`` still varies across models even though ``rank`` is fixed: the cutoff
        probability is a property of the reference library, the prior a property of the
        training set. It uses ``combine``'s guarded formula so that the number reported is
        the one the ``lift`` output is actually compared against.
        """
        knots = getattr(self, "pooled_rank_knots_", None)
        if knots is None or not len(knots):
            return None

        anchors = _anchor_pair(getattr(self, "pooled_rank_anchors_", None))
        proba = float(
            proba_from_reference_rank(
                DECISION_RANK,
                prepared=prepare_knots(knots),
                anchor_high=None if anchors is None else anchors[1],
            )
        )
        clipped = float(np.clip(proba, 1e-7, 1.0 - 1e-7))
        prior = getattr(self, "population_prior_", None) or 0.0
        return {
            "rank": DECISION_RANK,
            "proba": proba,
            "logit": float(np.log(clipped / (1.0 - clipped))),
            "lift": float(proba / max(prior, _LIFT_EPS)) if prior > 0 else None,
            "source": DECISION_CUTOFF_SOURCE,
        }

    def _build_oof_diagnostics(self, stacked, y):
        """Advisory numbers describing how this model treats molecules with known labels.

        None of this enters the rank scale, and that is the design. Anchoring the scale on
        the out-of-fold actives -- so that "0.95 means looks like a known active" -- was
        considered and rejected, because it would put *every* model's median active at 0.95
        by construction and make a model with AUC 0.95 indistinguishable from one with 0.55.
        Reporting the same numbers instead keeps the signal: measured on simulated strong,
        moderate and no-skill models, the median active reads 0.937, 0.826 and 0.230, and
        that spread is the useful part.

        Three things, each ``None`` rather than an exception when it cannot be computed --
        these are advisory and must never be able to fail a fit:

        ``actives`` / ``inactives``
            Where known molecules land on the rank scale the user actually sees. This is
            what turns a rank into a decision: "known actives here score 0.85 to 0.95, and
            your compound scored 0.94". For a weak model the band comes out low and says so.

        ``screening_auc``
            Out-of-fold actives against the reference library. ``oof_auc`` separates actives
            from *measured inactives for this target*, which are usually close analogues
            from the same assay; a screen instead asks whether actives rise above generic
            chemical space. A model can do the first well and the second badly, and then a
            real screen drowns in false positives.

        ``generic_hit_rate``
            The share of drug-like chemical space this model would call active. A model
            calling 20% of a generic library a hit is a more actionable finding than any
            rank rescale.
        """
        knots = getattr(self, "pooled_rank_knots_", None)
        if stacked is None or knots is None or not len(knots):
            return None

        p1 = combine(*stacked[:4], spec=stacked[4], outputs=("proba",)).values["proba"][
            :, 1
        ]
        y = np.asarray(y).ravel()
        if len(y) != len(p1):
            return None

        prepared = prepare_knots(knots)
        # Anchored, so the band is in the units `predict_rank` actually returns.
        ranks = rank_from_reference(
            p1,
            prepared=prepared,
            anchors=_anchor_pair(getattr(self, "pooled_rank_anchors_", None)),
        )
        out = {
            "actives": _rank_band(ranks[y == 1]),
            "inactives": _rank_band(ranks[y == 0]),
            "screening_auc": None,
            "generic_hit_rate": None,
            "decision_cutoff_rank": None,
            "sensitivity_at_cutoff": None,
        }

        # The knots are a uniform subsample of the reference's pooled probabilities, so a
        # quantile, an AUROC or a tail fraction taken from them is unbiased and nothing
        # extra has to be held in memory.
        reference = np.asarray(knots, dtype=np.float64)
        act = p1[y == 1]
        if act.size and reference.size:
            from sklearn.metrics import roc_auc_score

            try:
                labels = np.concatenate(
                    [np.ones(act.size, dtype=int), np.zeros(reference.size, dtype=int)]
                )
                out["screening_auc"] = float(
                    roc_auc_score(labels, np.concatenate([act, reference]))
                )
            except Exception:  # pragma: no cover - degenerate label vectors only
                pass

        cut = getattr(self, "decision_cutoff_", None)
        if cut is not None and reference.size:
            p_cut = float(cut["proba"])
            out["decision_cutoff_rank"] = cut["rank"]
            # Now 0.01 by construction, and kept for exactly that reason: it is a one-line
            # check that inverting the rank landed where it claims. It no longer
            # discriminates between models -- `sensitivity_at_cutoff` does that.
            out["generic_hit_rate"] = float((reference >= p_cut).mean())
            # ...and now it is acted on, not merely recorded. The construction fails when
            # the reference's probabilities are nearly all equal: the four TAIL_ANCHORS
            # then interpolate to the same value, `_collapse_anchors` keeps only the
            # topmost, and inverting DECISION_RANK lands below the whole library -- a 100%
            # generic hit rate instead of 1%. Unreachable on a healthy 50,000-molecule
            # library; reachable with a small LAZYQSAR_REFERENCE_N tier or a saturated
            # model, and silent when it happens, which is the part worth fixing.
            _expected = DECISION_HIT_RATE
            if _expected and abs(out["generic_hit_rate"] - _expected) > 10 * _expected:
                logger.warning(
                    f"The decision cutoff admits {out['generic_hit_rate']:.1%} of the "
                    f"reference library, not the {_expected:.1%} that rank "
                    f"{DECISION_RANK} is defined to mean. The reference's pooled "
                    "probabilities are too concentrated for the anchors to separate, so "
                    "`rank` and the binary cutoff are not comparable with other models. "
                    "Check the reference tier (LAZYQSAR_REFERENCE_N) and whether this "
                    "model saturates."
                )
            # The share of its own known actives the model catches while calling that 1% of
            # drug-like space active. Sensitivity at a fixed generic hit rate: the standard
            # screening statistic, and the signal `generic_hit_rate` gives up.
            #
            # Precision is deliberately absent. The out-of-fold inactives are assay-matched
            # analogues, not generic chemistry, so a precision here would look usable and
            # not be -- the same trap the oof_auc/screening_auc note above describes.
            out["sensitivity_at_cutoff"] = (
                float((act >= p_cut).mean()) if act.size else None
            )
        return out

    def _build_reference_rank_knots(self):
        """Score the reference library and keep the sorted pooled probabilities as knots.

        This is what makes ``rank`` mean "beats this fraction of drug-like chemical space"
        rather than "beats this fraction of my own training set". A trained model's
        out-of-fold scores are bimodal -- inactives crushed near 0, actives near 1 -- so
        screening compounds land in the empty middle where the training ECDF is flat and
        every one of them comes back around 0.9.

        The reference molecules go through exactly the path a query takes: same models,
        same pooling, same weighting. That identity is the whole design. It is what makes
        the result uniform on the library by construction, instead of uniform only while
        two code paths happen to agree.

        Streamed in chunks. Materialising three 50,000-row matrices at once would cost
        about 1.2 GB for a value that is consumed row by row.
        """
        from .reference import ReferenceUnavailable, iter_chunks
        from .reference.identity import REFERENCE_ID, default_n
        from .reference.manifest import manifest_sha256

        active_indices = self._active_indices()
        names = [self.descriptor_types[i] for i in active_indices]
        spec = _spec_from_attributes(
            self.descriptor_types,
            active_indices,
            self.quality_aucs_,
            self.proxy_aucs_,
            self._rank_error_curves_,
            self.ad_hard_cutoffs_,
            self.population_prior_,
        )

        # Positional against `active_indices`, so index by name -- never by whatever order
        # a mapping iterates in.
        streams = [
            iter_chunks(name, expected_dim=self.descriptors[i].n_dim)
            for i, name in zip(active_indices, names)
        ]
        pooled = []
        for chunk_set in zip(*streams):
            Y, R, S, A = self._channels_from_matrices(
                list(chunk_set), active_indices, want_score=False
            )
            out = combine(Y, R, S, A, spec=spec, outputs=("proba",))
            pooled.append(out.values["proba"][:, 1].copy())

        p1 = np.concatenate(pooled) if pooled else np.empty(0)
        p1 = p1[np.isfinite(p1)]
        if p1.size == 0:
            raise ReferenceUnavailable(
                "The reference library produced no finite pooled probabilities; "
                "`predict_rank` would have nothing to report against."
            )

        knots = subsample_knots(np.sort(p1))
        meta = {
            "library": {
                "id": REFERENCE_ID,
                # The tier the matrices were read from, which `LAZYQSAR_REFERENCE_N` picks.
                "n": default_n(),
                "manifest_sha256": manifest_sha256(),
            },
            "descriptors": list(names),
            "saturation": {"p_max": float(p1.max())},
        }
        logger.info(
            f"Reference library scored: {p1.size:,} molecules, "
            f"pooled probability max {p1.max():.3f}"
        )
        return knots, meta

    def _channels_from_matrices(self, matrices, active_indices, want_score=True):
        """Channels for feature matrices the caller already has.

        Split out of :meth:`_channels` so the reference library can be scored through
        exactly the path a query takes -- same models, same pooling, same weighting. That
        identity is what makes the resulting percentiles uniform on the reference set by
        construction, rather than uniform only if two code paths happen to agree.

        *matrices* is positional against *active_indices*, so callers building it from a
        name-keyed source must index by ``self.descriptor_types[i]``, never by the order a
        dict happens to iterate in.

        ``want_score=False`` skips ``predict_score``. Nothing in the ``proba`` branch or in
        the weighting reads ``S``; computing it on a 50,000-row reference would be a second
        full pass through every preprocessor and head for a value that is discarded.
        """
        if len(matrices) != len(active_indices):
            raise ValueError(
                f"{len(matrices)} matrices for {len(active_indices)} active descriptors"
            )
        y_hats, score_preds, rank_preds, ad_scores = [], [], [], []
        for i, X in zip(active_indices, matrices):
            y_hats.append(self.models[i].predict_proba(X=X)[:, 1])
            if want_score:
                score_preds.append(_optional(self.models[i].predict_score, X))
            if self.ad_models:
                ad_scores.append(self.ad_models[i].score(X))
            rank_preds.append(_optional(self.models[i]._oof_percentile, X))
        return _stack_channels(y_hats, rank_preds, score_preds, ad_scores)

    def _channels(self, smiles_list):
        active_indices = self._active_indices()
        matrices = [self._transform_cached(i, smiles_list) for i in active_indices]
        Y, R, S, A = self._channels_from_matrices(matrices, active_indices)

        # Weight by quality (= 2*oof - train), not plain OOF AUC. Both loaders and
        # `EnsembleSpec.from_metadata` have always used quality, so passing `oof_aucs_`
        # here made the fitted model in memory weight its descriptors differently from
        # the checkpoint it writes -- the same model predicted one thing before `save()`
        # and another after `load()`.
        skill = getattr(self, "quality_aucs_", None) or getattr(self, "oof_aucs_", None)
        spec = _spec_from_attributes(
            self.descriptor_types,
            active_indices,
            skill,
            getattr(self, "proxy_aucs_", None),
            getattr(self, "_rank_error_curves_", None),
            getattr(self, "ad_hard_cutoffs_", None),
            getattr(self, "population_prior_", 0.5),
            getattr(self, "pooled_rank_knots_", None),
            getattr(self, "pooled_score_knots_", None),
            _anchor_pair(getattr(self, "pooled_rank_anchors_", None)),
            (getattr(self, "decision_cutoff_", None) or {}).get("proba"),
        )
        return Y, R, S, A, spec

    def save_raw(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        meta = {
            "task": "classification",
            "mode": self.mode,
            "descriptor_types": self.descriptor_types,
            "n_compounds": getattr(self, "n_compounds_", None),
            "n_actives": getattr(self, "n_actives_", None),
            "ratio_actives": float(self.population_prior_)
            if hasattr(self, "population_prior_")
            else None,
            "population_prior": float(self.population_prior_)
            if hasattr(self, "population_prior_")
            else 0.5,
            "portfolio": self.models[0]._model.portfolio if self.models else [],
            "num_batches": {
                name: len(m._model.models)
                for name, m in zip(self.descriptor_types, self.models)
            }
            if self.models
            else {},
            # The pre-reference fallback: a mean of the per-descriptor balanced-accuracy
            # cutoffs, each in its own unit. Overwritten in one block below when this model
            # has a reference library, which is the only case that can place the cutoff on
            # the rank scale. Left here so a model fitted without one still reports
            # something, on the scale it always did.
            "decision_cutoff_raw": float(
                np.mean([m._model.decision_cutoff_raw_ for m in self.models])
            )
            if self.models
            else None,
            "decision_cutoff_proba": float(
                np.mean([m._model.decision_cutoff_proba_ for m in self.models])
            )
            if self.models
            else None,
            "decision_cutoff_rank": float(
                np.mean([m._model.decision_cutoff_rank_ for m in self.models])
            )
            if self.models
            else None,
            "decision_cutoff_source": None,
            # An AUC that could not be computed is omitted, exactly as `proxy_aucs`
            # already does, so the reader sees "unknown" rather than a stand-in number.
            "oof_aucs": {
                name: float(auc)
                for name, auc in zip(self.descriptor_types, self.oof_aucs_)
                if auc is not None
            }
            if hasattr(self, "oof_aucs_")
            else {},
            "proxy_aucs": {
                name: float(auc)
                for name, auc in zip(self.descriptor_types, self.proxy_aucs_)
                if auc is not None
            }
            if hasattr(self, "proxy_aucs_")
            else {},
            # An AUC that could not be computed is omitted, exactly as `proxy_aucs`
            # already does, so the reader sees "unknown" rather than a stand-in number.
            "train_aucs": {
                name: float(auc)
                for name, auc in zip(self.descriptor_types, self.train_aucs_)
                if auc is not None
            }
            if hasattr(self, "train_aucs_")
            else {},
            # An AUC that could not be computed is omitted, exactly as `proxy_aucs`
            # already does, so the reader sees "unknown" rather than a stand-in number.
            "quality_aucs": {
                name: float(auc)
                for name, auc in zip(self.descriptor_types, self.quality_aucs_)
                if auc is not None
            }
            if hasattr(self, "quality_aucs_")
            else {},
            "active_descriptors": {
                name: bool(active)
                for name, active in zip(self.descriptor_types, self.active_descriptors_)
            }
            if hasattr(self, "active_descriptors_")
            else {},
            "ad_hard_cutoffs": {
                name: float(c)
                for name, c in zip(self.descriptor_types, self.ad_hard_cutoffs_)
            }
            if hasattr(self, "ad_hard_cutoffs_")
            else {},
            "rank_error_curves": {
                name: curve
                for name, curve in zip(self.descriptor_types, self._rank_error_curves_)
                if curve is not None
            }
            if hasattr(self, "_rank_error_curves_")
            else {},
        }
        if self.models:
            _avg_p = meta["decision_cutoff_proba"]
            if _avg_p is not None:
                _p_clip = float(np.clip(_avg_p, 1e-7, 1.0 - 1e-7))
                meta["decision_cutoff_logit"] = float(np.log(_p_clip / (1.0 - _p_clip)))
                _prior = meta.get("population_prior") or 0
                meta["decision_cutoff_lift"] = (
                    float(_avg_p / _prior) if _prior > 0 else None
                )
            else:
                meta["decision_cutoff_logit"] = None
                meta["decision_cutoff_lift"] = None
        else:
            meta["decision_cutoff_logit"] = None
            meta["decision_cutoff_lift"] = None

        _score_knots = getattr(self, "pooled_score_knots_", None)
        if _score_knots is not None:
            _sx, _sy = _score_knots
            meta["pooled_scorer"] = {
                "knots_x": np.asarray(_sx, dtype=np.float64).tolist(),
                "knots_y": np.asarray(_sy, dtype=np.float64).tolist(),
                "n_train": int(len(_sx)),
                "source": "oof",
            }

        _diag = getattr(self, "oof_diagnostics_", None)
        if _diag is not None:
            # Top level, not inside `pooled_ranker`: these describe the model, not the
            # scale, and nothing may read them as part of the reference.
            meta["oof_diagnostics"] = _diag

        _knots = getattr(self, "pooled_rank_knots_", None)
        if _knots is not None and len(_knots):
            _ref = getattr(self, "reference_meta_", None) or {}
            meta["pooled_ranker"] = {
                "knots": np.asarray(_knots, dtype=np.float64).tolist(),
                "n_train": int(len(_knots)),
                # Readers gate on this. An out-of-fold reference and a library reference
                # are both monotone and both land in [0, 1], so without it a checkpoint
                # from an older version would be read as a library percentile and be
                # wrong in a way nothing downstream could detect.
                "source": "reference_library",
                "library": _ref.get("library"),
                # The knots describe the pooled probability of exactly this descriptor
                # set. The runner builds its active set from whichever sub-directories
                # exist on disk, so a checkpoint missing one would otherwise rank against
                # a distribution it never had.
                "descriptors": _ref.get("descriptors"),
                "saturation": _ref.get("saturation"),
            }
            # The top of the scale is pinned on out-of-fold molecules, which do not travel
            # in the checkpoint, so the anchor itself has to. `anchor_low` travels too, as
            # a diagnostic; it no longer shapes the scale.
            _anchors = getattr(self, "pooled_rank_anchors_", None)
            if _anchors:
                meta["pooled_ranker"].update(_anchors)
            # The whole cutoff family, in one place, derived from the single probability
            # `_build_decision_cutoff` inverted out of the rank scale. Previously
            # `decision_cutoff_rank` was corrected here while proba/logit/lift were built
            # from a mean-of-means far above -- one threshold maintained in two conditional
            # regions, which is how they drifted apart.
            _cut = getattr(self, "decision_cutoff_", None)
            if _cut is not None:
                meta["decision_cutoff_proba"] = _cut["proba"]
                meta["decision_cutoff_rank"] = _cut["rank"]
                meta["decision_cutoff_logit"] = _cut["logit"]
                meta["decision_cutoff_lift"] = _cut["lift"]
                meta["decision_cutoff_source"] = _cut["source"]
                # `decision_cutoff_raw` on the scale the task's `score` output emits. The
                # mean it replaces averaged per-head raw cutoffs living on different
                # quantities (XGB raw probability, RF vote fraction, SVC sigmoid-of-margin,
                # LR probability), giving a number that corresponded to no emitted scale.
                #
                # Reporting only: `score_from_knots` is a running-max staircase, so a query
                # whose score equals this value can sit either side of the probability
                # cutoff. `score >= decision_cutoff_raw` is NOT equivalent to `binary`.
                if _score_knots is not None:
                    meta["decision_cutoff_raw"] = float(
                        score_from_knots(_cut["proba"], _score_knots)
                    )

        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self._save_descriptor_dirs(model_dir)

    @classmethod
    def load_raw(cls, model_dir: str):
        from .agnostic import LazyClassifier

        descriptor_types = _descriptor_dirs(model_dir)
        # Read mode from metadata if available; fall back to inference for old models.
        meta_path = os.path.join(model_dir, "metadata.json")
        mode = None
        active_map = {}
        if os.path.isfile(meta_path):
            _meta = read_json(meta_path)
            mode = _meta.get("mode")
            active_map = _meta.get("active_descriptors") or {}
        if mode is None:
            for k, v in DESCRIPTORS_MODE.items():
                if set(v) == set(descriptor_types):
                    mode = k
                    break
        if mode is None:
            mode = "slow"  # safe default for models saved before mode tracking

        # Same as `load_onnx`: the mask decides what is opened, not just what is scored.
        descriptors, models, ad_models = _load_descriptor_stack(
            model_dir,
            descriptor_types,
            _positions_to_load(
                [active_map.get(d, True) for d in descriptor_types]
                if active_map
                else None,
                len(descriptor_types),
            ),
            model_loader=LazyClassifier.load,
        )

        obj = cls(mode=mode)
        obj.descriptor_types = descriptor_types
        obj.descriptors = descriptors
        obj.models = models
        obj.ad_models = ad_models if any(a is not None for a in ad_models) else []
        if os.path.isfile(meta_path):
            meta = read_json(meta_path)
            oof_map = meta.get("oof_aucs", {})
            proxy_map = meta.get("proxy_aucs", {})
            train_map = meta.get("train_aucs", {})
            quality_map = meta.get("quality_aucs", {})
            active_map = meta.get("active_descriptors", {})
            cutoff_map = meta.get("ad_hard_cutoffs", {})
            curve_map = meta.get("rank_error_curves", {})
            obj.population_prior_ = float(meta.get("population_prior", 0.5))
            # Weighting uses quality (= 2*oof - train), which penalises descriptors
            # that overfit. load_onnx has always done this; load_raw used plain oof,
            # so the same checkpoint scored differently depending on which loader ran.
            obj.oof_aucs_ = [
                quality_map.get(d, oof_map.get(d, 1.0)) for d in descriptor_types
            ]
            obj.proxy_aucs_ = [proxy_map.get(d) for d in descriptor_types]
            obj.train_aucs_ = [train_map.get(d, 0.0) for d in descriptor_types]
            obj.quality_aucs_ = [
                quality_map.get(d, oof_map.get(d, 1.0)) for d in descriptor_types
            ]
            obj.active_descriptors_ = [
                active_map.get(d, True) for d in descriptor_types
            ]
            obj.ad_hard_cutoffs_ = (
                [cutoff_map.get(d, 0.0) for d in descriptor_types]
                if cutoff_map
                else None
            )
            obj._rank_error_curves_ = [
                (np.array(curve_map[d][0]), np.array(curve_map[d][1]))
                if d in curve_map
                else None
                for d in descriptor_types
            ]
            obj.pooled_rank_knots_ = read_pooled_rank_knots(meta)
            obj.pooled_rank_anchors_ = _anchors_from_metadata(meta)
            obj.pooled_score_knots_ = read_pooled_score_knots(meta)
            obj.decision_cutoff_ = _decision_cutoff_from_metadata(meta)
        else:
            obj.population_prior_ = 0.5
            obj.oof_aucs_ = [1.0] * len(descriptor_types)
            obj.proxy_aucs_ = [None] * len(descriptor_types)
            obj.train_aucs_ = [0.0] * len(descriptor_types)
            obj.quality_aucs_ = [1.0] * len(descriptor_types)
            obj.active_descriptors_ = [True] * len(descriptor_types)
            obj.ad_hard_cutoffs_ = None
            obj._rank_error_curves_ = [None] * len(descriptor_types)
            obj.pooled_rank_knots_ = None
            # Was never set on this branch, so only `getattr` defaults elsewhere kept it
            # from raising. Set explicitly alongside its siblings.
            obj.pooled_rank_anchors_ = None
            obj.pooled_score_knots_ = None
            obj.decision_cutoff_ = None
        return obj

    def save_onnx(self, model_dir: str):
        """No-op, kept so ``save`` reads as raw-then-onnx.

        There is nothing to do: ``save_raw`` already writes the ONNX graphs, because the
        per-descriptor ``LazyClassifier.save`` it delegates to exports them. The name is
        retained because it is the obvious counterpart to ``load_onnx`` and removing it
        would break any caller that mirrors ``save_raw``/``save_onnx``.
        """

    @classmethod
    def load_onnx(cls, model_dir: str):
        """Load an exported checkpoint as an ONNX-backed :class:`ArtifactWrapper`.

        The metadata is read before anything is opened: it carries the active mask, and a
        descriptor the portfolio rejected is never loaded at all.
        """
        from .agnostic import LazyClassifier

        descriptor_types = _descriptor_dirs(model_dir)
        meta_path = os.path.join(model_dir, "metadata.json")
        meta = read_json(meta_path) if os.path.isfile(meta_path) else {}
        spec, active_names = EnsembleSpec.from_metadata(meta, descriptor_types)
        active_indices = [descriptor_types.index(n) for n in active_names]
        descriptors, artifacts, ad_artifacts = _load_descriptor_stack(
            model_dir,
            descriptor_types,
            [i in active_indices for i in range(len(descriptor_types))],
            model_loader=LazyClassifier.load,
        )
        has_ad = any(a is not None for a in ad_artifacts)
        return ArtifactWrapper(
            descriptors,
            artifacts,
            ad_artifacts if has_ad else None,
            spec,
            active_indices,
            dict(meta, descriptor_types=descriptor_types),
        )

    def save(self, model_dir: str):
        """Write the checkpoint; a path ending in ``.zip`` writes a zip archive."""
        return _save_maybe_zipped(model_dir, self.save_raw)

    @classmethod
    def load(cls, model_dir: str):
        with _unpacked(model_dir) as path:
            _check_task(path, "classification")
            if any(_has_onnx(os.path.join(path, d)) for d in _descriptor_dirs(path)):
                return cls.load_onnx(model_dir=path)
            return cls.load_raw(model_dir=path)


# Defined in its own module, re-exported here where callers have always looked for it.
from .qsar_regressor import LazyRegressorQSAR  # noqa: E402


class LazyQSAR:
    """
    Dispatcher that returns the appropriate QSAR class based on task.

    LazyQSAR(task='classification', **kwargs)  →  LazyClassifierQSAR(**kwargs)
    LazyQSAR(task='regression', **kwargs)      →  LazyRegressorQSAR(**kwargs)
    LazyQSAR.load(model_dir)                   →  whichever the checkpoint's task is
    """

    def __new__(cls, task: str = "classification", **kwargs):
        if task == "classification":
            return LazyClassifierQSAR(**kwargs)
        elif task == "regression":
            return LazyRegressorQSAR(**kwargs)
        else:
            raise ValueError(
                f"Unknown task {task!r}. Choose 'classification' or 'regression'."
            )

    @staticmethod
    def load(model_dir: str):
        """Load a saved QSAR model of either task, dispatching on its ``metadata.json``."""
        from .qsar_base import _read_task

        with _unpacked(model_dir) as path:
            task = _read_task(path)
        if task == "regression":
            return LazyRegressorQSAR.load(model_dir)
        return LazyClassifierQSAR.load(model_dir)

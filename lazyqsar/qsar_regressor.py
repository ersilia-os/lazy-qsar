"""SMILES-level regression: ``LazyRegressorQSAR`` and its ONNX inference wrapper.

The regression twin of :class:`lazyqsar.qsar.LazyClassifierQSAR`. One
:class:`~lazyqsar.agnostic.LazyRegressor` per selected descriptor, combined per sample by
:func:`lazyqsar.ensemble.combine` with a regression spec:

- **skill** -- each descriptor's quality ``2 * oof_rho - train_rho`` (clipped to [-1, 1]),
  blended with its proxy ρ from descriptor screening;
- **reliability** -- a rank -> error curve learned from *out-of-fold* predictions: where
  a descriptor's prediction sits in its own OOF distribution, and how wrong it tends to
  be there. Out-of-fold rather than in-sample (the classifier's choice) because tree
  heads drive training residuals to almost zero, which would make every curve flat;
- **applicability domain** -- the same per-descriptor veto as classification.

None of the classification calibration exists here: no prior, no reference library,
no decision cutoff. The single output is ``value``, the predicted target in its own
units.

The recipe shared with classification is in :mod:`lazyqsar.qsar_base`.
"""

import json
import os

import numpy as np

from .ensemble import EnsembleSpec, combine
from .ensemble.intervals import DEFAULT_LEVEL, fit_intervals, interval_half_width
from .poolers.regression.inner_pooler import stacking_weights
from .qsar_base import (
    as_smiles_list,
    _ArtifactWrapperBase,
    _EnsemblePredictMixin,
    _QSARBase,
    _check_task,
    _descriptor_dirs,
    _load_descriptor_stack,
    _save_maybe_zipped,
    _unpacked,
    validate_smiles,
)
from .utils.archives import read_json
from .utils.logging import logger
from .utils.metrics import regression_metrics
from .utils.splits import check_trainable_regression

# Pruning thresholds on out-of-fold Spearman ρ. The classification floor (AUC 0.55) and
# gap (0.10 AUC) map to ρ via AUC - 0.5 ≈ ρ / 2.
_PRUNE_FLOOR = 0.10
_PRUNE_GAP = 0.20
# Residuals are normalised by this percentile so one outlier cannot flatten the curve.
_ERROR_NORM_PERCENTILE = 95
_CURVE_KNOTS = 20
# Blend descriptors by NNLS stacking of their out-of-fold predictions (with the AD veto)
# instead of skill x reliability weighting; see docs/regression.md.
_STACK_DESCRIPTORS = True


def _quality_rho(oof_rho, train_rho):
    """``2*oof - train`` on Spearman ρ, clipped to [-1, 1]; OOF alone if train is unknown.

    The same overfitting penalty the classifier applies to AUC. ``None`` stays ``None``:
    unknown skill is weighted as unknown, never as a guess.
    """
    if oof_rho is None:
        return None
    if train_rho is None:
        return float(np.clip(oof_rho, -1.0, 1.0))
    return float(np.clip(2.0 * oof_rho - train_rho, -1.0, 1.0))


def _rank_error_curve(rank, value, y):
    """Rank -> normalised absolute error, ``(r_knots, e_knots)`` lists, or None.

    Windowed means over the OOF rank, exactly as the classifier builds its curve, with
    the error ``|y - value|`` scaled by its 95th percentile and clipped to [0, 1] so that
    ``1 - error`` is a usable reliability in [0, 1].
    """
    rank = np.asarray(rank, dtype=float)
    err = np.abs(np.asarray(y, dtype=float) - np.asarray(value, dtype=float))
    scale = float(np.percentile(err, _ERROR_NORM_PERCENTILE))
    if not np.isfinite(scale) or scale <= 0:
        return None
    err = np.clip(err / scale, 0.0, 1.0)
    order = np.argsort(rank)
    rs, es = rank[order], err[order]
    nk = min(_CURVE_KNOTS, len(rs))
    ki = np.round(np.linspace(0, len(rs) - 1, nk)).astype(int)
    hw = max(1, len(rs) // (nk * 2))
    e_knots = np.array([es[max(0, k - hw) : k + hw + 1].mean() for k in ki])
    return rs[ki].tolist(), e_knots.tolist()


def _regression_spec(
    names,
    active_indices,
    skill,
    proxy,
    curves,
    cutoffs,
    intervals=None,
    descriptor_weights=None,
):
    def sliced(seq):
        return tuple(seq[i] for i in active_indices) if seq else None

    weights = None
    if descriptor_weights is not None:
        w = np.array([descriptor_weights[i] for i in active_indices], dtype=float)
        weights = tuple((w / w.sum()).tolist()) if w.sum() > 0 else None

    return EnsembleSpec(
        descriptor_names=tuple(names[i] for i in active_indices),
        oof_aucs=sliced(skill),
        proxy_aucs=sliced(proxy),
        rank_error_curves=sliced(curves),
        ad_hard_cutoffs=sliced(cutoffs),
        task="regression",
        intervals=intervals,
        descriptor_weights=weights,
    )


class _RegressionPredictMixin(_EnsemblePredictMixin):
    """``predict`` over :func:`combine`, cached per SMILES list, NaN for unparseable rows."""

    def _wanted_outputs(self, spec) -> tuple:
        # `lower`/`upper` at the default level ride along when the checkpoint can make
        # them; other levels are read off the cached result by `predict_interval`.
        return ("value", "lower", "upper") if spec.intervals else ("value",)

    def predict(self, smiles_list):
        """Predicted target values, shape (n_samples,), NaN for unparseable SMILES."""
        return self._combined(smiles_list).values["value"]

    def predict_interval(self, smiles_list, level=DEFAULT_LEVEL):
        """Prediction interval ``(lower, upper)``, each ``(n_samples,)``, at coverage *level*.

        Split-conformal on the ensemble's out-of-fold residuals, binned by the
        molecule's applicability-domain score (:mod:`lazyqsar.ensemble.intervals`):
        about a fraction *level* of molecules like the training set fall inside it, and
        molecules far from the training set get the wider intervals of the least typical
        training molecules. NaN rows stay NaN.
        """
        result = self._combined(smiles_list)
        spec = self._interval_spec()
        if not spec:
            raise ValueError(
                "This model carries no prediction-interval model (fitted before "
                "intervals existed); refit it to get intervals."
            )
        value = result.values["value"]
        half = interval_half_width(spec, len(value), result.ad_summary, level)
        return value - half, value + half

    def predict_value(self, smiles_list):
        """Alias of :meth:`predict`, named after the ``value`` output."""
        return self.predict(smiles_list)


class RegressorArtifactWrapper(_RegressionPredictMixin, _ArtifactWrapperBase):
    """ONNX inference wrapper for a saved ``LazyRegressorQSAR``.

    Returned by :meth:`LazyRegressorQSAR.load`. numpy + onnxruntime for the models;
    the descriptors need whatever their featurizers need.
    """

    task = "regression"

    def _interval_spec(self):
        return self.spec.intervals


class LazyRegressorQSAR(_RegressionPredictMixin, _QSARBase):
    """
    SMILES-aware regressor with built-in descriptor computation.

    Trains one ``LazyRegressor`` per selected descriptor and combines their predictions
    with a Spearman-ρ-weighted, per-sample-reliability-aware ensemble.

    Parameters
    ----------
    mode : str
        ``"fast"`` (Morgan only) or ``"slow"`` (CDDD, Chemeleon, CLAMP, Morgan, RDKit).

    Attributes (after fit)
    ----------------------
    descriptor_types : list[str]
    models : list[LazyRegressor]
    oof_rhos_, train_rhos_, quality_rhos_, proxy_rhos_ : list[float | None]
    active_descriptors_ : list[bool]
    oof_metrics_ : dict
        Spearman ρ, R², RMSE and MAE of the pooled ensemble's out-of-fold predictions.
    """

    task = "regression"
    _skill_attrs = ("oof_rhos_", "train_rhos_", "quality_rhos_", "proxy_rhos_")
    _metric_label = "ρ"
    _prune_floor = _PRUNE_FLOOR
    _prune_gap = _PRUNE_GAP

    def _model_class(self):
        from .agnostic import LazyRegressor

        return LazyRegressor

    @staticmethod
    def _model_skill(model):
        return model.oof_rho_, model.train_rho_

    @staticmethod
    def _quality(oof, train):
        return _quality_rho(oof, train)

    @staticmethod
    def _rank_error_curve(model, X, y, channels):
        return (
            None if channels is None else _rank_error_curve(channels[1], channels[0], y)
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, smiles_list, y, precomputed=None, validate=True):
        """Fit one regressor per selected descriptor.

        Parameters
        ----------
        smiles_list : list of str
        y : array-like
            Continuous target, finite and not constant.
        precomputed : dict, optional
            ``{descriptor_name: feature_matrix}`` aligned with *smiles_list*.
        validate : bool
            Parse-check the SMILES first.
        """
        smiles_list = as_smiles_list(smiles_list)
        self._feature_cache.clear()
        self._ensemble_cache.clear()

        y = np.asarray(y, dtype=float).ravel()
        check_trainable_regression(y, where="LazyRegressorQSAR.fit")
        if validate:
            validate_smiles(smiles_list)
        n = len(smiles_list)
        self.n_compounds_ = n
        self.y_summary_ = {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
        }

        proxy = self._select_descriptors(smiles_list, y, precomputed)
        logger.rule("LazyRegressorQSAR")
        logger.info(
            f"mode={self.mode}  descriptors={self.descriptor_types}  n={n:,}  "
            f"y={self.y_summary_['mean']:.3f}±{self.y_summary_['std']:.3f}"
        )
        oof_channels, train_ad, desc_rows = self._fit_descriptors(smiles_list, y, proxy)

        self.oof_metrics_ = self._build_oof_metrics(oof_channels, train_ad, y)
        logger.rule()
        logger.descriptor_table(desc_rows, metric="ρ")
        if self.oof_metrics_:
            m = self.oof_metrics_
            logger.info(
                "ensemble OOF: "
                + "  ".join(
                    f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, float)
                )
            )

    def _spec(self, active_indices):
        return _regression_spec(
            self.descriptor_types,
            active_indices,
            self.quality_rhos_,
            self.proxy_rhos_,
            self._rank_error_curves_,
            self.ad_hard_cutoffs_,
            getattr(self, "intervals_", None),
            getattr(self, "descriptor_weights_", None),
        )

    def _interval_spec(self):
        return getattr(self, "intervals_", None)

    def _build_oof_metrics(self, oof_channels, train_ad, y):
        """Pool every active descriptor's OOF predictions exactly as a query is pooled.

        Sets ``intervals_``, the conformal interval model, from the pooled out-of-fold
        residuals binned by the training rows' AD score, and returns the OOF metrics.
        """
        self.intervals_ = None
        self.descriptor_weights_ = None
        active = self._active_indices()
        cols = [oof_channels[i] for i in active]
        if any(c is None for c in cols):
            return None
        Y = np.column_stack([c[0] for c in cols])
        if _STACK_DESCRIPTORS and len(active) > 1:
            # One weight per descriptor, fitted on their out-of-fold predictions.
            weights = np.zeros(len(self.descriptor_types))
            weights[active] = stacking_weights(Y, y)
            self.descriptor_weights_ = weights.tolist()
        R = np.column_stack([c[1] for c in cols])
        A = np.column_stack([train_ad[i] for i in active])
        result = combine(Y, R, None, A, spec=self._spec(active), outputs=("value",))
        pooled = result.values["value"]
        self.intervals_ = fit_intervals(y, pooled, result.ad_summary)
        return regression_metrics(y, pooled)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _channels(self, smiles_list):
        from .qsar_base import _empty_channels, _stack_channels

        active = self._active_indices()
        if not smiles_list:
            return _empty_channels(len(active)) + (self._spec(active),)
        y_hats, rank_preds, ad_scores = [], [], []
        for i in active:
            X = self._transform_cached(i, smiles_list)
            value = self.models[i].predict(X=X)
            y_hats.append(value)
            rank_preds.append(self.models[i]._oof_percentile_from_value(value))
            if self.ad_models:
                ad_scores.append(self.ad_models[i].score(X))
        return _stack_channels(y_hats, rank_preds, [], ad_scores) + (
            self._spec(active),
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _metadata(self):
        def by_name(values):
            return {
                name: float(v)
                for name, v in zip(self.descriptor_types, values)
                if v is not None
            }

        return {
            "task": "regression",
            "mode": self.mode,
            "descriptor_types": self.descriptor_types,
            "n_compounds": getattr(self, "n_compounds_", None),
            "y_summary": getattr(self, "y_summary_", None),
            "portfolio": self.models[0]._model.portfolio if self.models else [],
            "num_batches": {
                name: len(m._model.models)
                for name, m in zip(self.descriptor_types, self.models)
            },
            "oof_rhos": by_name(self.oof_rhos_),
            "train_rhos": by_name(self.train_rhos_),
            "quality_rhos": by_name(self.quality_rhos_),
            "proxy_rhos": by_name(self.proxy_rhos_),
            "active_descriptors": {
                name: bool(a)
                for name, a in zip(self.descriptor_types, self.active_descriptors_)
            },
            "ad_hard_cutoffs": by_name(self.ad_hard_cutoffs_),
            "rank_error_curves": {
                name: curve
                for name, curve in zip(self.descriptor_types, self._rank_error_curves_)
                if curve is not None
            },
            "oof_metrics": getattr(self, "oof_metrics_", None),
            "intervals": getattr(self, "intervals_", None),
            "descriptor_weights": (
                None
                if getattr(self, "descriptor_weights_", None) is None
                else {
                    name: float(w)
                    for name, w in zip(self.descriptor_types, self.descriptor_weights_)
                }
            ),
        }

    def save_raw(self, model_dir: str):
        """Write ``metadata.json`` plus one sub-directory per descriptor (ONNX graphs)."""
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(self._metadata(), f, indent=2)
        self._save_descriptor_dirs(model_dir)

    def save(self, model_dir: str):
        """Write the checkpoint; a path ending in ``.zip`` writes a zip archive."""
        return _save_maybe_zipped(model_dir, self.save_raw)

    @classmethod
    def load(cls, model_dir: str) -> RegressorArtifactWrapper:
        """Load a saved regressor as an ONNX-backed :class:`RegressorArtifactWrapper`."""
        from .agnostic import LazyRegressor

        with _unpacked(model_dir) as path:
            _check_task(path, "regression")
            meta = read_json(os.path.join(path, "metadata.json"))
            descriptor_types = _descriptor_dirs(path)
            spec, active_names = EnsembleSpec.from_metadata(meta, descriptor_types)
            active_indices = [descriptor_types.index(n) for n in active_names]
            descriptors, artifacts, ad_artifacts = _load_descriptor_stack(
                path,
                descriptor_types,
                [i in active_indices for i in range(len(descriptor_types))],
                model_loader=LazyRegressor.load,
            )
            has_ad = any(a is not None for a in ad_artifacts)
            return RegressorArtifactWrapper(
                descriptors,
                artifacts,
                ad_artifacts if has_ad else None,
                spec,
                active_indices,
                dict(meta, descriptor_types=descriptor_types),
            )

    load_onnx = load


__all__ = ["LazyRegressorQSAR", "RegressorArtifactWrapper"]

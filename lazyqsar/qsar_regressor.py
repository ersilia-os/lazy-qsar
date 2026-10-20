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

Kept in its own module, and reaching the shared helpers in :mod:`lazyqsar.qsar` lazily,
so neither module has to import the other at load time.
"""

import os
import shutil

import numpy as np

from .ensemble import EnsembleSpec, combine, mask_rows
from .ensemble.channels import score_smiles_chunkwise
from .ensemble.runner import get_chunk_size
from .registry import DESCRIPTOR_TYPES, DESCRIPTORS_MODE
from .utils.archives import read_json, unpack_to_scratch
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


def _qsar():
    from . import qsar

    return qsar


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


def _regression_spec(names, active_indices, skill, proxy, curves, cutoffs):
    def sliced(seq):
        return tuple(seq[i] for i in active_indices) if seq else None

    return EnsembleSpec(
        descriptor_names=tuple(names[i] for i in active_indices),
        oof_aucs=sliced(skill),
        proxy_aucs=sliced(proxy),
        rank_error_curves=sliced(curves),
        ad_hard_cutoffs=sliced(cutoffs),
        task="regression",
    )


class _RegressionPredictMixin:
    """``predict`` over :func:`combine`, cached per SMILES list, NaN for unparseable rows."""

    def _channels(self, smiles_list):
        """Return ``(Y, R, A, spec)``; ``R``/``A`` may be None."""
        raise NotImplementedError

    def _combined(self, smiles_list):
        q = _qsar()
        key = q._smiles_md5(smiles_list)
        result = self._ensemble_cache.get(key)
        if result is None:
            bad = q.invalid_smiles_indices(smiles_list)
            Y, R, A, spec = self._channels(smiles_list)
            result = combine(Y, R, None, A, spec=spec, outputs=("value",))
            if bad:
                logger.warning(
                    f"{len(bad)} SMILES could not be parsed; their predictions are NaN "
                    f"(positions: {bad[:10]}{' ...' if len(bad) > 10 else ''})"
                )
                mask_rows(result.values, bad)
            if result.diagnostics:
                logger.ad_weights_table(result.diagnostics, n_samples=len(smiles_list))
            self._ensemble_cache[key] = result
        return result

    def predict(self, smiles_list):
        """Predicted target values, shape (n_samples,), NaN for unparseable SMILES."""
        return self._combined(smiles_list).values["value"]

    def predict_value(self, smiles_list):
        """Alias of :meth:`predict`, named after the ``value`` output."""
        return self.predict(smiles_list)


class RegressorArtifactWrapper(_RegressionPredictMixin):
    """ONNX inference wrapper for a saved ``LazyRegressorQSAR``.

    Returned by :meth:`LazyRegressorQSAR.load`. numpy + onnxruntime for the models;
    the descriptors need whatever their featurizers need.
    """

    task = "regression"

    def __init__(
        self, descriptors, artifacts, ad_artifacts, spec, active_indices, meta
    ):
        self.descriptors = descriptors
        self.artifacts = artifacts
        self.ad_artifacts = ad_artifacts
        self.spec = spec
        self.active_indices = active_indices
        self.metadata = meta
        self.descriptor_types = meta.get("descriptor_types")
        self._ensemble_cache = {}

    def _channels(self, smiles_list):
        q = _qsar()
        if not smiles_list:
            Y, R, _, A = q._empty_channels(len(self.active_indices))
            return Y, R, A, self.spec
        chunk_size = get_chunk_size()
        y_hats, rank_preds, ad_scores = [], [], []
        for i in self.active_indices:
            ad = self.ad_artifacts[i] if self.ad_artifacts else None
            ch = score_smiles_chunkwise(
                self.descriptors[i],
                self.artifacts[i],
                ad,
                smiles_list,
                chunk_size,
                want={"y", "r", "a"},
                logger=logger,
            )
            y_hats.append(ch.y)
            rank_preds.append(ch.r)
            if ad is not None:
                ad_scores.append(ch.a)
        Y, R, _, A = q._stack_channels(y_hats, rank_preds, [], ad_scores)
        return Y, R, A, self.spec


class LazyRegressorQSAR(_RegressionPredictMixin):
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

    def __init__(self, mode: str = "slow"):
        assert mode in ("fast", "slow"), (
            f"Mode '{mode}' not recognized. Choose from 'fast' or 'slow'."
        )
        self.mode = mode
        self.descriptor_types = DESCRIPTORS_MODE[mode]
        self.descriptors = []
        self._feature_cache = {}
        self._ensemble_cache = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def _transform_cached(self, i, smiles_list):
        key = (i, _qsar()._smiles_md5(smiles_list))
        if key not in self._feature_cache:
            self._feature_cache[key] = self.descriptors[i].transform(smiles_list)
        return self._feature_cache[key]

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
        import time

        from .agnostic import LazyRegressor
        from .applicability import ApplicabilityDomain
        from .descriptors.portfolio import DescriptorPortfolio

        self._feature_cache.clear()
        self._ensemble_cache.clear()

        y = np.asarray(y, dtype=float).ravel()
        check_trainable_regression(y, where="LazyRegressorQSAR.fit")
        if validate:
            _qsar().validate_smiles(smiles_list)
        n = len(smiles_list)
        self.n_compounds_ = n
        self.y_summary_ = {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
        }

        applicable = DescriptorPortfolio(self.mode).select(
            smiles_list, y=y, precomputed=precomputed, task="regression"
        )
        self.descriptor_types = [name for name, _, _, _ in applicable]
        self.descriptors = [desc for _, desc, _, _ in applicable]
        self.proxy_rhos_ = [score for _, _, _, score in applicable]
        smiles_hash = _qsar()._smiles_md5(smiles_list)
        for i, (_, _, X, _) in enumerate(applicable):
            if X is not None:
                self._feature_cache[(i, smiles_hash)] = X

        logger.rule("LazyRegressorQSAR")
        logger.info(
            f"mode={self.mode}  descriptors={self.descriptor_types}  n={n:,}  "
            f"y={self.y_summary_['mean']:.3f}±{self.y_summary_['std']:.3f}"
        )

        self.models, self.ad_models = [], []
        self.oof_rhos_, self.train_rhos_, self.quality_rhos_ = [], [], []
        self._rank_error_curves_ = []
        self.ad_hard_cutoffs_ = []
        oof_channels, train_ad, desc_rows = [], [], []

        for i, desc_name in enumerate(self.descriptor_types):
            t0 = time.perf_counter()
            X = self._transform_cached(i, smiles_list)
            feat_time = time.perf_counter() - t0
            sparsity = float((X == 0).mean())
            logger.info(
                f"[{desc_name}] p={X.shape[1]:,}  sparsity={sparsity:.3f}  "
                f"feat_time={feat_time:.1f}s"
            )

            model = LazyRegressor()
            model.fit(X=X, y=y)
            self.models.append(model)

            channels = model.oof_channels(X=X)
            oof_channels.append(channels)
            self._rank_error_curves_.append(
                None
                if channels is None
                else _rank_error_curve(channels[1], channels[0], y)
            )

            oof_rho, train_rho = model.oof_rho_, model.train_rho_
            quality = _quality_rho(oof_rho, train_rho)
            self.oof_rhos_.append(oof_rho)
            self.train_rhos_.append(train_rho)
            self.quality_rhos_.append(quality)

            ad = ApplicabilityDomain()
            ad.fit(X)
            self.ad_models.append(ad)
            scores = ad.score(X)
            train_ad.append(scores)
            self.ad_hard_cutoffs_.append(float(np.percentile(scores, 5)))

            def _fmt(value):
                return "unknown" if value is None else f"{value:.4f}"

            logger.info(
                f"[{desc_name}] OOF ρ={_fmt(oof_rho)}  train ρ={_fmt(train_rho)}  "
                f"quality={_fmt(quality)}  AD comps={ad.pca_.n_components_}"
            )
            desc_rows.append(
                {
                    "name": desc_name,
                    "n_features": X.shape[1],
                    "sparsity": sparsity,
                    "feat_time": feat_time,
                    "ad_n_components": ad.pca_.n_components_,
                    "ad_cal_min": float(ad.cal_knots_[0]),
                    "ad_cal_max": float(ad.cal_knots_[-1]),
                    "proxy_auc": self.proxy_rhos_[i],
                    "train_auc": train_rho,
                    "quality_auc": quality,
                }
            )

        # Pruning on OOF ρ; an unknown ρ keeps its descriptor (a failed measurement says
        # nothing about the descriptor), and pruning everything falls back to keeping all.
        known = [r for r in self.oof_rhos_ if r is not None]
        best = max(known) if known else 0.0
        mask = [
            True if r is None else (r >= _PRUNE_FLOOR and r >= best - _PRUNE_GAP)
            for r in self.oof_rhos_
        ]
        self.active_descriptors_ = mask if any(mask) else [True] * len(mask)
        for row, active in zip(desc_rows, self.active_descriptors_):
            row["active"] = active

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

    def _active_indices(self):
        mask = getattr(self, "active_descriptors_", None) or [True] * len(
            self.descriptor_types
        )
        idx = [i for i, a in enumerate(mask) if a]
        return idx or list(range(len(self.descriptor_types)))

    def _spec(self, active_indices):
        return _regression_spec(
            self.descriptor_types,
            active_indices,
            self.quality_rhos_,
            self.proxy_rhos_,
            self._rank_error_curves_,
            self.ad_hard_cutoffs_,
        )

    def _build_oof_metrics(self, oof_channels, train_ad, y):
        """Pool every active descriptor's OOF predictions exactly as a query is pooled."""
        active = self._active_indices()
        cols = [oof_channels[i] for i in active]
        if any(c is None for c in cols):
            return None
        Y = np.column_stack([c[0] for c in cols])
        R = np.column_stack([c[1] for c in cols])
        A = np.column_stack([train_ad[i] for i in active])
        pooled = combine(
            Y, R, None, A, spec=self._spec(active), outputs=("value",)
        ).values["value"]
        return regression_metrics(y, pooled)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _channels(self, smiles_list):
        active = self._active_indices()
        if not smiles_list:
            Y, R, _, A = _qsar()._empty_channels(len(active))
            return Y, R, A, self._spec(active)
        y_hats, rank_preds, ad_scores = [], [], []
        for i in active:
            X = self._transform_cached(i, smiles_list)
            value = self.models[i].predict(X=X)
            y_hats.append(value)
            rank_preds.append(self.models[i]._oof_percentile_from_value(value))
            if self.ad_models:
                ad_scores.append(self.ad_models[i].score(X))
        Y, R, _, A = _qsar()._stack_channels(y_hats, rank_preds, [], ad_scores)
        return Y, R, A, self._spec(active)

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
        }

    def save(self, model_dir: str):
        """Write ``metadata.json`` plus one sub-directory per descriptor (ONNX graphs)."""
        import json

        zip_out = model_dir.endswith(".zip")
        if zip_out:
            model_dir = model_dir[:-4]
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(self._metadata(), f, indent=2)
        for i, name in enumerate(self.descriptor_types):
            sub = os.path.join(model_dir, name)
            os.makedirs(sub, exist_ok=True)
            self.models[i].save(sub)
            self.descriptors[i].save(sub)
            if self.ad_models:
                self.ad_models[i].save(os.path.join(sub, "applicability_domain"))
        if zip_out:
            shutil.make_archive(model_dir, "zip", model_dir)
            shutil.rmtree(model_dir, ignore_errors=True)
            return model_dir + ".zip"
        return model_dir

    @classmethod
    def load(cls, model_dir: str) -> RegressorArtifactWrapper:
        """Load a saved regressor as an ONNX-backed :class:`RegressorArtifactWrapper`."""
        from .agnostic import LazyRegressor

        q = _qsar()
        scratch = None
        try:
            if model_dir.endswith(".zip"):
                scratch, model_dir = unpack_to_scratch(model_dir)
            q._check_task(model_dir, "regression")
            meta = read_json(os.path.join(model_dir, "metadata.json"))
            descriptor_types = sorted(
                fn for fn in os.listdir(model_dir) if fn in DESCRIPTOR_TYPES
            )
            spec, active_names = EnsembleSpec.from_metadata(meta, descriptor_types)
            active_indices = [descriptor_types.index(n) for n in active_names]
            to_load = [i in active_indices for i in range(len(descriptor_types))]
            descriptors, artifacts, ad_artifacts = q._load_descriptor_stack(
                model_dir, descriptor_types, to_load, model_loader=LazyRegressor.load
            )
            has_ad = any(a is not None for a in ad_artifacts)
            meta = dict(meta, descriptor_types=descriptor_types)
            return RegressorArtifactWrapper(
                descriptors,
                artifacts,
                ad_artifacts if has_ad else None,
                spec,
                active_indices,
                meta,
            )
        finally:
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)


__all__ = ["LazyRegressorQSAR", "RegressorArtifactWrapper"]

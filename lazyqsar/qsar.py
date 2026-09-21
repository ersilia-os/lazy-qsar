import hashlib
import json
import os
import shutil
import numpy as np

from .ensemble import (
    OUTPUT_NAMES,
    EnsembleSpec,
    build_pooled_rank_knots,
    combine,
    mask_rows,
)
from .ensemble.channels import score_smiles_chunkwise
from .ensemble.runner import get_chunk_size
from .registry import (  # noqa: F401  (re-exported for backwards compatibility)
    DESCRIPTOR_TYPES,
    DESCRIPTORS_MODE,
    get_descriptor_type,
)
from .ensemble.combine import read_pooled_rank_knots
from .utils.logging import logger
from .utils.ranking import prepare_knots, rank_from_knots


def _smiles_md5(smiles_list):
    return hashlib.md5("\x00".join(smiles_list).encode()).hexdigest()


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


def validate_smiles(smiles_list):
    """Parse-check SMILES, importing RDKit only when actually called.

    Kept out of the module imports so that ``LazyClassifierQSAR`` and ``ArtifactWrapper``
    can be imported on a base install. RDKit is still required to featurize anything, so
    a real prediction needs it either way — but importing the class must not.
    """
    from .descriptors._validate import validate_smiles as _validate

    return _validate(smiles_list)


def invalid_smiles_indices(smiles_list):
    """Positions RDKit cannot parse, importing RDKit only when actually called.

    The predict counterpart to :func:`validate_smiles`: prediction reports the gap rather
    than refusing the batch, so a single malformed row cannot abort a library screen.
    """
    from .descriptors._validate import invalid_smiles_indices as _indices

    return _indices(smiles_list)


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


def _stack_channels(y_hats, rank_preds, score_preds, ad_scores):
    """Stack per-descriptor prediction lists into the (n_samples, n_descriptors) channels.

    ``R`` and ``S`` collapse to None unless *every* descriptor supplied them, matching how
    the weighting treats a partially available channel: it is not meaningful to mix a real
    rank for one descriptor with a placeholder for another.
    """
    Y = np.stack(y_hats, axis=1).astype(np.float64)
    R = (
        np.stack(rank_preds, axis=1).astype(np.float64)
        if rank_preds and all(r is not None for r in rank_preds)
        else None
    )
    S = (
        np.stack(score_preds, axis=1).astype(np.float64)
        if score_preds and all(sc is not None for sc in score_preds)
        else None
    )
    A = (
        np.stack(ad_scores, axis=1).astype(np.float64)
        if len(ad_scores) == Y.shape[1]
        else None
    )
    return Y, R, S, A


class _EnsemblePredictMixin:
    """The six ``predict_*`` methods, implemented once over :func:`combine`.

    Subclasses supply :meth:`_channels`, which is the only thing that genuinely differs
    between the fit-time estimator and the ONNX artifact wrapper: one reads in-memory
    scikit-learn models, the other reads ONNX sessions. Everything downstream — weighting,
    the six output formulas, the diagnostics table — is identical, and used to be
    maintained as two near-verbatim copies that drifted apart.

    Results are cached per SMILES list, so asking for several outputs featurizes once.
    """

    def _channels(self, smiles_list):
        """Return ``(Y, R, S, A, spec)`` for *smiles_list*.

        ``R``, ``S`` and ``A`` may be ``None`` when a descriptor cannot supply them.
        """
        raise NotImplementedError

    def _combined(self, smiles_list):
        cache_key = _smiles_md5(smiles_list)
        result = self._ensemble_cache.get(cache_key)
        if result is None:
            # Unparseable SMILES are scored like any other row -- their descriptors come
            # back all-NaN and the exported preprocessor imputes them -- and then blanked.
            # The positions are taken up front rather than inferred from the NaN pattern
            # afterwards, because an all-NaN descriptor row is not proof the SMILES was
            # bad: a descriptor can fail on a molecule that parses perfectly well, and
            # that row should keep its score from the descriptors that did work.
            bad = invalid_smiles_indices(smiles_list)
            Y, R, S, A, spec = self._channels(smiles_list)
            result = combine(Y, R, S, A, spec=spec, outputs=OUTPUT_NAMES)
            if bad:
                logger.warning(
                    f"{len(bad)} SMILES could not be parsed; their predictions are NaN "
                    f"(positions: {bad[:10]}{' ...' if len(bad) > 10 else ''})"
                )
                mask_rows(result.values, bad)
            if result.diagnostics:
                logger.ad_weights_table(result.diagnostics, n_samples=len(smiles_list))
            self._ensemble_cache[cache_key] = result
        return result

    def predict_proba(self, smiles_list):
        """Calibrated ensemble probabilities, shape (n_samples, 2)."""
        return self._combined(smiles_list).values["proba"]

    def predict_logit(self, smiles_list):
        """Log-odds of the calibrated probabilities, shape (n_samples, 2)."""
        return self._combined(smiles_list).values["logit"]

    def predict_rank(self, smiles_list):
        """Weighted quantile ranks in [0, 1], shape (n_samples, 2)."""
        return self._combined(smiles_list).values["rank"]

    def predict_score(self, smiles_list):
        """Weighted raw (pre-calibration) scores, shape (n_samples, 2)."""
        return self._combined(smiles_list).values["score"]

    def predict_lift(self, smiles_list):
        """Probability over the population prior, shape (n_samples, 2)."""
        return self._combined(smiles_list).values["lift"]

    def predict(self, smiles_list, threshold=0.5, cutoff=None):
        """Binary labels, shape (n_samples,).

        ``cutoff`` is accepted as an alias for ``threshold``: the two classes this mixin
        replaced spelled the same argument differently, and both spellings are in use.
        """
        if cutoff is not None:
            threshold = cutoff
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


def _spec_from_attributes(
    names,
    active_indices,
    oof_aucs,
    proxy_aucs,
    curves,
    cutoffs,
    prior,
    pooled_knots=None,
):
    """Build an :class:`EnsembleSpec` from the per-descriptor attribute lists.

    The lists are indexed by *full* descriptor position; the spec is sliced down to the
    active ones so the weighting code never has to re-index. *pooled_knots* is the one
    argument that is not per-descriptor: it describes the pooled probability of the
    active set as a whole, and is passed through unsliced.
    """

    def sliced(seq):
        return tuple(seq[i] for i in active_indices) if seq else None

    return EnsembleSpec(
        descriptor_names=tuple(names[i] for i in active_indices),
        oof_aucs=sliced(oof_aucs),
        proxy_aucs=sliced(proxy_aucs),
        rank_error_curves=sliced(curves),
        ad_hard_cutoffs=sliced(cutoffs),
        population_prior=prior,
        pooled_rank_knots=pooled_knots,
    )


class ArtifactWrapper(_EnsemblePredictMixin):
    """
    ONNX inference wrapper for a multi-descriptor LazyClassifierQSAR model.

    Returned by ``LazyClassifierQSAR.load()`` / ``load_onnx()``. Holds one
    ONNX artifact per descriptor type and exposes the same predict_* API as
    the training-time ``LazyClassifierQSAR``.

    Parameters
    ----------
    descriptors : list
        Fitted descriptor objects (one per descriptor type).
    artifacts : list
        Loaded ONNX artifact objects (one per descriptor type).
    ad_artifacts : list or None
        Applicability domain artifacts (optional).
    active_descriptors : list[bool] or None
        Mask of which descriptors passed the applicability check.
    ad_hard_cutoffs : list[float] or None
        AD threshold below which a descriptor is vetoed for a sample.
    oof_aucs : list[float | None]
        Out-of-fold AUC per descriptor (used for weighting).
    proxy_aucs : list[float | None]
        Proxy AUC per descriptor (used for weighting).
    rank_error_curves : list[tuple | None]
        (r_knots, e_knots) rank→error curves per descriptor.
    population_prior : float
        Fraction of positives in the original training set.
    descriptor_types : list[str] or None
        Descriptor names in the same order as artifacts.
    """

    def __init__(
        self,
        descriptors,
        artifacts,
        ad_artifacts=None,
        active_descriptors=None,
        ad_hard_cutoffs=None,
        oof_aucs=None,
        proxy_aucs=None,
        rank_error_curves=None,
        population_prior=0.5,
        descriptor_types=None,
        pooled_rank_knots=None,
    ):
        self.descriptors = descriptors
        self.artifacts = artifacts
        self.ad_artifacts = ad_artifacts
        self.active_descriptors = active_descriptors  # list[bool] or None
        self.ad_hard_cutoffs = ad_hard_cutoffs  # list[float] or None
        self.oof_aucs = oof_aucs  # list[float|None]
        self.proxy_aucs = proxy_aucs  # list[float|None]
        self.rank_error_curves = rank_error_curves  # list[(r_knots, e_knots)|None]
        self.population_prior = population_prior
        self.descriptor_types = descriptor_types  # list[str] or None
        self.pooled_rank_knots = pooled_rank_knots  # ndarray or None
        self._ensemble_cache = {}

    def _channels(self, smiles_list):

        active_mask = self.active_descriptors or [True] * len(self.descriptors)
        active_indices = [i for i, a in enumerate(active_mask) if a]
        if not active_indices:
            active_indices = list(range(len(self.descriptors)))

        # Featurize and score in chunks rather than transforming the whole list first.
        # A million compounds against a 2048-dimensional descriptor is ~8 GB of float32,
        # and this is the entry point used to score large libraries from Python.
        chunk_size = get_chunk_size()
        y_hats, score_preds, rank_preds, ad_scores = [], [], [], []
        for i in active_indices:
            ad = (
                self.ad_artifacts[i]
                if self.ad_artifacts is not None and self.ad_artifacts[i] is not None
                else None
            )
            channels = score_smiles_chunkwise(
                self.descriptors[i],
                self.artifacts[i],
                ad,
                smiles_list,
                chunk_size,
                want={"y", "r", "s", "a"},
                logger=logger,
            )
            y_hats.append(channels.y)
            score_preds.append(channels.s)
            rank_preds.append(channels.r)
            if ad is not None:
                ad_scores.append(channels.a)

        names = self.descriptor_types or [str(i) for i in range(len(self.descriptors))]
        spec = _spec_from_attributes(
            names,
            active_indices,
            self.oof_aucs,
            self.proxy_aucs,
            self.rank_error_curves,
            self.ad_hard_cutoffs,
            self.population_prior,
            getattr(self, "pooled_rank_knots", None),
        )
        return _stack_channels(y_hats, rank_preds, score_preds, ad_scores) + (spec,)


class LazyClassifierQSAR(_EnsemblePredictMixin):
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

    def __init__(
        self,
        mode: str = "slow",
    ):
        assert mode in ("fast", "slow"), (
            f"Mode '{mode}' not recognized. Choose from 'fast' or 'slow'."
        )
        self.mode = mode
        self.descriptor_types = DESCRIPTORS_MODE[mode]
        self.descriptors = []  # populated in fit() after applicability check
        self.is_saved = False
        self._feature_cache = {}
        self._ensemble_cache = {}

    def _smiles_hash(self, smiles_list):
        return _smiles_md5(smiles_list)

    def _transform_cached(self, i, smiles_list):
        key = (i, self._smiles_hash(smiles_list))
        if key not in self._feature_cache:
            self._feature_cache[key] = self.descriptors[i].transform(smiles_list)
        else:
            logger.debug(
                f"Using cached features for descriptor: {self.descriptor_types[i]}"
            )
        return self._feature_cache[key]

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
        import time
        from .agnostic import LazyClassifier
        from .applicability import ApplicabilityDomain
        from .descriptors.portfolio import DescriptorPortfolio

        # Clear any cached state from a previous fit.
        self._feature_cache.clear()
        self._ensemble_cache.clear()

        y = np.array(y, dtype=int)
        if validate:
            validate_smiles(smiles_list)
        n = len(smiles_list)
        pos_rate = float(y.mean())
        self.population_prior_ = pos_rate
        self.n_compounds_ = n
        self.n_actives_ = int((y == 1).sum())

        applicable = DescriptorPortfolio(self.mode).select(
            smiles_list, y=y, precomputed=precomputed
        )
        self.descriptor_types = [name for name, _, _, _ in applicable]
        self.descriptors = [desc for _, desc, _, _ in applicable]
        self.proxy_aucs_ = [pauc for _, _, _, pauc in applicable]

        # Pre-populate feature cache with matrices computed during screening.
        smiles_hash = self._smiles_hash(smiles_list)
        for i, (_, _, X, _) in enumerate(applicable):
            if X is not None:
                self._feature_cache[(i, smiles_hash)] = X

        logger.rule("LazyClassifierQSAR")
        logger.info(
            f"mode={self.mode}  descriptors={self.descriptor_types}  "
            f"n={n:,}  pos_rate={pos_rate:.1%}"
        )

        self.models = []
        self.ad_models = []
        self.oof_aucs_ = []
        self.train_aucs_ = []
        self.quality_aucs_ = []
        self._rank_error_curves_ = []
        _ad_hard_cutoffs_raw = []
        _oof_channels = []
        _train_ad = []
        desc_rows = []

        for i, desc_name in enumerate(self.descriptor_types):
            t0 = time.perf_counter()
            X = self._transform_cached(i, smiles_list)
            feat_time = time.perf_counter() - t0

            sparsity = float((X == 0).mean())
            logger.info(
                f"[{desc_name}] p={X.shape[1]:,}  sparsity={sparsity:.3f}  "
                f"feat_time={feat_time:.1f}s"
            )

            model = LazyClassifier()
            model.fit(X=X, y=y)
            self.models.append(model)

            # Build rank→error curve from training predictions (20 knots, windowed).
            # Training-set predictions are optimistic but sufficient as a relative
            # reliability signal: high |rank - 0.5| should map to low error.
            try:
                _p = model.predict_proba(X=X)[:, 1]
                _r = model.predict_rank(X=X)[:, 1]
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
                self._rank_error_curves_.append((_r_knots.tolist(), _e_knots.tolist()))
            except Exception:
                self._rank_error_curves_.append(None)

            oof_auc = model.oof_auc_
            train_auc = model.train_auc_
            gap = train_auc - oof_auc
            quality = oof_auc - gap  # α=1: quality = 2*oof - train

            self.oof_aucs_.append(oof_auc)
            self.train_aucs_.append(train_auc)
            self.quality_aucs_.append(quality)

            _oof_channels.append(self.models[i].oof_channels(X=X))

            ad = ApplicabilityDomain()
            ad.fit(X)
            self.ad_models.append(ad)
            train_ad = ad.score(X)
            _train_ad.append(train_ad)
            _ad_hard_cutoffs_raw.append(float(np.percentile(train_ad, 5)))

            logger.info(
                f"[{desc_name}] OOF={oof_auc:.4f}  train={train_auc:.4f}  "
                f"gap={gap:.4f}  quality={quality:.4f}  "
                f"AD comps={ad.pca_.n_components_}"
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
                    "proxy_auc": self.proxy_aucs_[i],
                    "train_auc": train_auc,
                    "quality_auc": quality,
                }
            )

        # Descriptor-level pruning: drop if OOF AUC < floor OR < best - gap
        best_oof = max(self.oof_aucs_)
        _floor, _gap = 0.55, 0.10
        active_mask = [
            (auc >= _floor) and (auc >= best_oof - _gap) for auc in self.oof_aucs_
        ]
        if not any(active_mask):
            active_mask = [True] * len(self.oof_aucs_)
        self.active_descriptors_ = active_mask
        self.ad_hard_cutoffs_ = _ad_hard_cutoffs_raw

        self.pooled_rank_knots_ = self._build_pooled_rank_reference(
            _oof_channels, _train_ad
        )

        for row, active in zip(desc_rows, active_mask):
            row["active"] = active

        logger.rule()
        logger.descriptor_table(desc_rows)

    def _build_pooled_rank_reference(self, oof_channels, train_ad):
        """Learn the pooled out-of-fold distribution the ensemble's ``rank`` reports against.

        Must run after the active set is settled: the reference describes the pooled
        probability of exactly the descriptors that will be scored, and a reference built
        over three descriptors does not describe the pooled probability of two.

        Returns ``None`` -- and ``rank`` keeps its pre-v3.5.0 behaviour -- unless every
        active descriptor supplied out-of-fold channels. A reference assembled from a
        subset would still be monotone and still lie in [0, 1], so nothing downstream
        could tell it was calibrated against the wrong distribution.
        """
        active_indices = [i for i, a in enumerate(self.active_descriptors_) if a]
        if not active_indices:
            return None
        cols = [oof_channels[i] for i in active_indices]
        if any(c is None for c in cols):
            logger.debug(
                "No pooled rank reference: at least one active descriptor has no "
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
        return build_pooled_rank_knots(Y, R, S, A, spec)

    def _channels(self, smiles_list):

        active_mask = getattr(
            self, "active_descriptors_", [True] * len(self.descriptor_types)
        )
        active_indices = [i for i, a in enumerate(active_mask) if a]
        if not active_indices:
            active_indices = list(range(len(self.descriptor_types)))

        y_hats, score_preds, rank_preds, ad_scores = [], [], [], []
        for i in active_indices:
            X = self._transform_cached(i, smiles_list)
            y_hats.append(self.models[i].predict_proba(X=X)[:, 1])
            score_preds.append(_optional(self.models[i].predict_score, X))
            if self.ad_models:
                ad_scores.append(self.ad_models[i].score(X))
            rank_preds.append(_optional(self.models[i].predict_rank, X))

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
        )
        return _stack_channels(y_hats, rank_preds, score_preds, ad_scores) + (spec,)

    def save_raw(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        meta = {
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
            "oof_aucs": {
                name: float(auc)
                for name, auc in zip(self.descriptor_types, self.oof_aucs_)
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
            "train_aucs": {
                name: float(auc)
                for name, auc in zip(self.descriptor_types, self.train_aucs_)
            }
            if hasattr(self, "train_aucs_")
            else {},
            "quality_aucs": {
                name: float(auc)
                for name, auc in zip(self.descriptor_types, self.quality_aucs_)
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

        _knots = getattr(self, "pooled_rank_knots_", None)
        if _knots is not None and len(_knots):
            meta["pooled_ranker"] = {
                "knots": np.asarray(_knots, dtype=np.float64).tolist(),
                "n_train": int(len(_knots)),
                "source": "oof",
            }
            # decision_cutoff_raw/proba/logit/lift are one learned threshold in four
            # units. Its rank image used to be a mean of the per-descriptor rank cutoffs,
            # which after the pooled reference lands on a scale nothing emits. Nothing in
            # the package thresholds on it -- binary is proba >= 0.5 -- but it is reported
            # to users, so it should mean what its name says.
            _cut_p = meta.get("decision_cutoff_proba")
            if _cut_p is not None:
                meta["decision_cutoff_rank"] = float(
                    rank_from_knots(float(_cut_p), prepared=prepare_knots(_knots))
                )

        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        for i, descriptor_name in enumerate(self.descriptor_types):
            model_subdir = os.path.join(model_dir, descriptor_name)
            os.makedirs(model_subdir, exist_ok=True)
            logger.debug(f"Saving model to {model_subdir}")
            self.models[i].save(model_subdir)
            logger.debug(f"Saving descriptor to {model_subdir}")
            self.descriptors[i].save(model_subdir)
            if self.ad_models:
                ad_subdir = os.path.join(model_subdir, "applicability_domain")
                self.ad_models[i].save(ad_subdir)
        self.is_saved = True

    @classmethod
    def load_raw(cls, model_dir: str):
        from .applicability import ApplicabilityDomainArtifact

        descriptor_types = []
        for fn in os.listdir(model_dir):
            if fn in DESCRIPTOR_TYPES.keys():
                descriptor_types += [fn]
        descriptor_types = sorted(descriptor_types)
        # Read mode from metadata if available; fall back to inference for old models.
        meta_path = os.path.join(model_dir, "metadata.json")
        mode = None
        if os.path.isfile(meta_path):
            with open(meta_path) as _f:
                _meta = json.load(_f)
            mode = _meta.get("mode")
        if mode is None:
            for k, v in DESCRIPTORS_MODE.items():
                if set(v) == set(descriptor_types):
                    mode = k
                    break
        if mode is None:
            mode = "slow"  # safe default for models saved before mode tracking
        from .agnostic import LazyClassifier

        descriptors = []
        models = []
        ad_models = []
        for descriptor_type in descriptor_types:
            model_subdir = os.path.join(model_dir, descriptor_type)
            if not os.path.exists(model_subdir):
                raise FileNotFoundError(
                    f"Descriptor directory {model_subdir} does not exist."
                )
            descriptors += [get_descriptor_type(descriptor_type).load(model_subdir)]
            models += [LazyClassifier.load(model_subdir)]
            ad_subdir = os.path.join(model_subdir, "applicability_domain")
            if os.path.isdir(ad_subdir):
                ad_models.append(ApplicabilityDomainArtifact.load(ad_subdir))
            else:
                ad_models.append(None)

        obj = cls(mode=mode)
        obj.descriptor_types = descriptor_types
        obj.descriptors = descriptors
        obj.models = models
        obj.ad_models = ad_models if any(a is not None for a in ad_models) else []
        obj.is_saved = True
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
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
        return obj

    def save_onnx(self, model_dir: str, clean: bool = True):
        # ONNX is already written by save_raw() via LazyClassifier.save().
        pass

    @classmethod
    def load_onnx(cls, model_dir: str):
        from .applicability import ApplicabilityDomainArtifact

        descriptor_types = []
        for fn in os.listdir(model_dir):
            if fn in DESCRIPTOR_TYPES.keys():
                descriptor_types += [fn]
        descriptor_types = sorted(descriptor_types)
        from .agnostic import LazyClassifier

        descriptors = []
        artifacts = []
        ad_artifacts = []
        for descriptor_type in descriptor_types:
            model_subdir = os.path.join(model_dir, descriptor_type)
            if not os.path.exists(model_subdir):
                raise FileNotFoundError(
                    f"Descriptor directory {model_subdir} does not exist."
                )
            descriptors += [get_descriptor_type(descriptor_type).load(model_subdir)]
            artifacts += [LazyClassifier.load(model_subdir)]
            ad_subdir = os.path.join(model_subdir, "applicability_domain")
            if os.path.isdir(ad_subdir):
                ad_artifacts.append(ApplicabilityDomainArtifact.load(ad_subdir))
            else:
                ad_artifacts.append(None)

        has_ad = any(a is not None for a in ad_artifacts)
        meta_path = os.path.join(model_dir, "metadata.json")
        oof_aucs = None
        proxy_aucs = None
        active_descriptors = None
        ad_hard_cutoffs = None
        rank_error_curves = None
        population_prior = 0.5
        pooled_rank_knots = None
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            oof_map = meta.get("oof_aucs", {})
            proxy_map = meta.get("proxy_aucs", {})
            quality_map = meta.get("quality_aucs", {})
            active_map = meta.get("active_descriptors", {})
            cutoff_map = meta.get("ad_hard_cutoffs", {})
            curve_map = meta.get("rank_error_curves", {})
            population_prior = float(meta.get("population_prior", 0.5))
            oof_aucs = [
                quality_map.get(d, oof_map.get(d, 1.0)) for d in descriptor_types
            ]
            proxy_aucs = [proxy_map.get(d) for d in descriptor_types]
            active_descriptors = (
                [active_map.get(d, True) for d in descriptor_types]
                if active_map
                else None
            )
            ad_hard_cutoffs = (
                [cutoff_map.get(d, 0.0) for d in descriptor_types]
                if cutoff_map
                else None
            )
            rank_error_curves = (
                [
                    (np.array(curve_map[d][0]), np.array(curve_map[d][1]))
                    if d in curve_map
                    else None
                    for d in descriptor_types
                ]
                if curve_map
                else None
            )
            pooled_rank_knots = read_pooled_rank_knots(meta)
        return ArtifactWrapper(
            descriptors=descriptors,
            artifacts=artifacts,
            ad_artifacts=ad_artifacts if has_ad else None,
            active_descriptors=active_descriptors,
            ad_hard_cutoffs=ad_hard_cutoffs,
            oof_aucs=oof_aucs,
            proxy_aucs=proxy_aucs,
            rank_error_curves=rank_error_curves,
            population_prior=population_prior,
            descriptor_types=descriptor_types,
            pooled_rank_knots=pooled_rank_knots,
        )

    def save(self, model_dir: str):
        if model_dir.endswith(".zip"):
            zip = True
            model_dir = model_dir[:-4]
        else:
            zip = False
        self.save_raw(model_dir)
        self.save_onnx(model_dir)
        if zip:
            shutil.make_archive(model_dir, "zip", model_dir)
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            return model_dir + ".zip"
        return model_dir

    @classmethod
    def load(cls, model_dir: str):
        if model_dir.endswith(".zip"):
            zip = True
        else:
            zip = False
        if zip:
            base_dir = model_dir[:-4]
            if os.path.exists(base_dir):
                shutil.rmtree(base_dir)
            shutil.unpack_archive(model_dir, base_dir)
            model_dir = base_dir
        descriptor_types = []
        for fn in os.listdir(model_dir):
            if fn in DESCRIPTOR_TYPES.keys():
                descriptor_types += [fn]
        descriptor_types = sorted(descriptor_types)
        if any(
            _has_onnx(os.path.join(model_dir, descriptor_type))
            for descriptor_type in descriptor_types
        ):
            return cls.load_onnx(model_dir=model_dir)
        obj = cls.load_raw(model_dir=model_dir)
        if zip:
            shutil.rmtree(base_dir)
        return obj


class LazyRegressorQSAR:
    """Placeholder — not yet implemented."""

    def __init__(self, mode: str = "default"):
        raise NotImplementedError("LazyRegressorQSAR is not yet implemented.")


class LazyQSAR:
    """
    Dispatcher that returns the appropriate QSAR class based on task.

    LazyQSAR(task='classification', **kwargs)  →  LazyClassifierQSAR(**kwargs)
    LazyQSAR(task='regression', **kwargs)      →  LazyRegressorQSAR(**kwargs)
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

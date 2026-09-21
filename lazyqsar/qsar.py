import hashlib
import json
import os
import shutil
import numpy as np

from .ensemble import (
    OUTPUT_NAMES,
    EnsembleSpec,
    build_pooled_score_knots,
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
from .ensemble.combine import read_pooled_rank_knots, read_pooled_score_knots
from .utils.logging import logger
from .utils.ranking import prepare_knots, rank_from_reference, subsample_knots


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
            # Narrowed, not blanket. `combine` raises for `rank` on a checkpoint with no
            # reference library, and this call asks for every output at once -- so without
            # this the raise would take `predict_proba` down with it on every checkpoint
            # fitted before v3.6. `predict_rank` raises it deliberately instead.
            wanted = OUTPUT_NAMES
            if getattr(spec, "pooled_rank_knots", None) is None:
                wanted = tuple(n for n in OUTPUT_NAMES if n != "rank")
            result = combine(Y, R, S, A, spec=spec, outputs=wanted)
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
        """Percentile against the reference library, shape (n_samples, 2).

        ``0.99`` means the molecule scores above 99% of a fixed 50,000-molecule sample of
        drug-like chemical space -- not above 99% of this model's training set, which is
        what versions before 3.6 reported.

        Raises ``ValueError`` on a checkpoint that carries no reference library. The other
        five outputs still work; only this one changed meaning.
        """
        values = self._combined(smiles_list).values
        if "rank" not in values:
            from .ensemble.combine import NO_REFERENCE_MESSAGE

            raise ValueError(NO_REFERENCE_MESSAGE)
        return values["rank"]

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
):
    """Build an :class:`EnsembleSpec` from the per-descriptor attribute lists.

    The lists are indexed by *full* descriptor position; the spec is sliced down to the
    active ones so the weighting code never has to re-index. *pooled_knots* and
    *pooled_score_knots* are the arguments that are not per-descriptor: each describes the
    pooled probability of the active set as a whole, so both pass through unsliced.
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
        pooled_score_knots=pooled_score_knots,
    )


def _positions_to_load(active_descriptors, n_descriptors):
    """Which descriptor positions a loader should actually open.

    All of them when the checkpoint carries no active mask, and all of them when the mask
    rejects every descriptor -- both ``_channels`` implementations fall back to scoring
    everything in that case, so loading nothing would leave them indexing ``None``.
    """
    if not active_descriptors or not any(active_descriptors):
        return [True] * n_descriptors
    return [bool(a) for a in active_descriptors]


def _load_descriptor_stack(model_dir, descriptor_types, to_load):
    """Featurizer, model and applicability domain per descriptor; ``None`` where skipped.

    Inactive positions are kept rather than dropped so that every per-descriptor list stays
    index-aligned with *descriptor_types*, which is what ``_spec_from_attributes`` and both
    ``_channels`` implementations index into.

    Skipping is the point. A descriptor the portfolio rejected at fit time is never scored,
    but it was still being loaded -- an onnxruntime session per head per batch, plus the
    featurizer's own weights, which for chemeleon or cddd is a torch model read off disk.
    The shared inference runner has resolved the active set before opening anything since
    the descriptor union moved into ``ensemble.runner``; this is the Python entry point
    catching up.
    """
    from .agnostic import LazyClassifier
    from .applicability import ApplicabilityDomainArtifact

    descriptors, models, ad_models = [], [], []
    for descriptor_type, wanted in zip(descriptor_types, to_load):
        if not wanted:
            descriptors.append(None)
            models.append(None)
            ad_models.append(None)
            continue
        model_subdir = os.path.join(model_dir, descriptor_type)
        if not os.path.exists(model_subdir):
            raise FileNotFoundError(
                f"Descriptor directory {model_subdir} does not exist."
            )
        descriptors.append(get_descriptor_type(descriptor_type).load(model_subdir))
        models.append(LazyClassifier.load(model_subdir))
        ad_subdir = os.path.join(model_subdir, "applicability_domain")
        ad_models.append(
            ApplicabilityDomainArtifact.load(ad_subdir)
            if os.path.isdir(ad_subdir)
            else None
        )
    return descriptors, models, ad_models


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
        pooled_score_knots=None,
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
        self.pooled_score_knots = pooled_score_knots  # (ndarray, ndarray) or None
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
                # `s` only when this checkpoint has no pooled score map. With one, `score`
                # is read off the pooled probability, so asking for the raw channel would
                # run every graph a second time for a value nothing reads.
                want={"y", "r", "a"}
                if getattr(self, "pooled_score_knots", None) is not None
                else {"y", "r", "s", "a"},
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
            getattr(self, "pooled_score_knots", None),
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
        # After the knots, because the band is expressed on the scale they define.
        self.oof_diagnostics_ = self._build_oof_diagnostics(_stacked, y)

        for row, active in zip(desc_rows, active_mask):
            row["active"] = active

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
        if hit is not None:
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
        ranks = rank_from_reference(p1, prepared=prepared)
        out = {
            "actives": _rank_band(ranks[y == 1]),
            "inactives": _rank_band(ranks[y == 0]),
            "screening_auc": None,
            "generic_hit_rate": None,
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

        cutoffs = [
            m._model.decision_cutoff_proba_
            for m in self.models
            if hasattr(getattr(m, "_model", None), "decision_cutoff_proba_")
        ]
        if cutoffs and reference.size:
            out["generic_hit_rate"] = float(
                (reference >= float(np.mean(cutoffs))).mean()
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
        from .reference.identity import DEFAULT_N, REFERENCE_ID

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
            "library": {"id": REFERENCE_ID, "n": DEFAULT_N},
            "descriptors": list(names),
            "saturation": {"p_max": float(p1.max())},
        }
        logger.info(
            f"Reference library scored: {p1.size:,} molecules, "
            f"pooled probability max {p1.max():.3f}"
        )
        return knots, meta

    def _active_indices(self):
        active_mask = getattr(
            self, "active_descriptors_", [True] * len(self.descriptor_types)
        )
        active_indices = [i for i, a in enumerate(active_mask) if a]
        return active_indices or list(range(len(self.descriptor_types)))

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
            rank_preds.append(_optional(self.models[i].predict_rank, X))
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
        )
        return Y, R, S, A, spec

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
            # decision_cutoff_raw/proba/logit/lift are one learned threshold in four
            # units. Its rank image used to be a mean of the per-descriptor rank cutoffs,
            # which after the pooled reference lands on a scale nothing emits. Nothing in
            # the package thresholds on it -- binary is proba >= 0.5 -- but it is reported
            # to users, so it should mean what its name says.
            _cut_p = meta.get("decision_cutoff_proba")
            if _cut_p is not None:
                meta["decision_cutoff_rank"] = float(
                    rank_from_reference(float(_cut_p), prepared=prepare_knots(_knots))
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
        descriptor_types = []
        for fn in os.listdir(model_dir):
            if fn in DESCRIPTOR_TYPES.keys():
                descriptor_types += [fn]
        descriptor_types = sorted(descriptor_types)
        # Read mode from metadata if available; fall back to inference for old models.
        meta_path = os.path.join(model_dir, "metadata.json")
        mode = None
        active_map = {}
        if os.path.isfile(meta_path):
            with open(meta_path) as _f:
                _meta = json.load(_f)
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
        )

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
            obj.pooled_score_knots_ = read_pooled_score_knots(meta)
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
            obj.pooled_score_knots_ = None
        return obj

    def save_onnx(self, model_dir: str, clean: bool = True):
        # ONNX is already written by save_raw() via LazyClassifier.save().
        pass

    @classmethod
    def load_onnx(cls, model_dir: str):
        descriptor_types = []
        for fn in os.listdir(model_dir):
            if fn in DESCRIPTOR_TYPES.keys():
                descriptor_types += [fn]
        descriptor_types = sorted(descriptor_types)

        # The metadata is read before anything is opened, not after: it carries the active
        # mask, and a descriptor the portfolio rejected should never be loaded at all.
        meta_path = os.path.join(model_dir, "metadata.json")
        oof_aucs = None
        proxy_aucs = None
        active_descriptors = None
        ad_hard_cutoffs = None
        rank_error_curves = None
        population_prior = 0.5
        pooled_rank_knots = None
        pooled_score_knots = None
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
            pooled_score_knots = read_pooled_score_knots(meta)

        descriptors, artifacts, ad_artifacts = _load_descriptor_stack(
            model_dir,
            descriptor_types,
            _positions_to_load(active_descriptors, len(descriptor_types)),
        )
        has_ad = any(a is not None for a in ad_artifacts)

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
            pooled_score_knots=pooled_score_knots,
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

"""What the SMILES-level classifier and regressor share.

``LazyClassifierQSAR`` and ``LazyRegressorQSAR`` follow one recipe: select descriptors,
fit one agnostic model per descriptor, fit an applicability domain on each, prune weak
descriptors, and combine the survivors per sample with :func:`lazyqsar.ensemble.combine`.
At inference both score SMILES the same way, whether the models are in memory or loaded
from ONNX. This module holds that recipe once; :mod:`lazyqsar.qsar` and
:mod:`lazyqsar.qsar_regressor` add what is specific to each task (classification's
reference library, decision cutoff and diagnostics; regression's OOF metrics).

Importable on a base install: RDKit, scikit-learn and the descriptors load lazily.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from contextlib import contextmanager

import numpy as np

from .ensemble import combine, mask_rows
from .ensemble.channels import score_smiles_chunkwise
from .ensemble.runner import get_chunk_size
from .registry import DESCRIPTOR_TYPES, DESCRIPTORS_MODE, get_descriptor_type
from .utils.archives import read_json, unpack_to_scratch
from .utils.logging import logger

# Each descriptor's applicability-domain veto sits at this percentile of its own
# training-set AD scores.
AD_HARD_CUTOFF_PERCENTILE = 5


# ---------------------------------------------------------------------------
# SMILES and channels
# ---------------------------------------------------------------------------


def _smiles_md5(smiles_list):
    return hashlib.md5("\x00".join(smiles_list).encode()).hexdigest()


def validate_smiles(smiles_list):
    """Parse-check SMILES, importing RDKit only when actually called.

    Kept out of the module imports so that the QSAR classes and their artifact wrappers
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


def _empty_channels(n_descriptors):
    """Correctly shaped ``(Y, R, S, A)`` for a query of no molecules."""
    zero = np.zeros((0, n_descriptors), dtype=np.float64)
    return zero, zero.copy(), zero.copy(), zero.copy()


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


# ---------------------------------------------------------------------------
# Checkpoints on disk
# ---------------------------------------------------------------------------


def _descriptor_dirs(model_dir):
    """The descriptor sub-directories of a checkpoint, in sorted (column) order."""
    return sorted(fn for fn in os.listdir(model_dir) if fn in DESCRIPTOR_TYPES)


def _positions_to_load(active_descriptors, n_descriptors):
    """Which descriptor positions a loader should actually open.

    All of them when the checkpoint carries no active mask, and all of them when the mask
    rejects every descriptor -- the ``_channels`` implementations fall back to scoring
    everything in that case, so loading nothing would leave them indexing ``None``.
    """
    if not active_descriptors or not any(active_descriptors):
        return [True] * n_descriptors
    return [bool(a) for a in active_descriptors]


def _load_descriptor_stack(model_dir, descriptor_types, to_load, model_loader):
    """Featurizer, model and applicability domain per descriptor; ``None`` where skipped.

    Inactive positions are kept rather than dropped so that every per-descriptor list stays
    index-aligned with *descriptor_types*, which is what the specs and the ``_channels``
    implementations index into. Skipping is the point: a descriptor the portfolio rejected
    at fit time is never scored, so opening its sessions and featurizer weights (a torch
    model, for chemeleon or cddd) would be pure cost.

    *model_loader* loads one descriptor's model: ``LazyClassifier.load`` or
    ``LazyRegressor.load``.
    """
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
        models.append(model_loader(model_subdir))
        ad_subdir = os.path.join(model_subdir, "applicability_domain")
        ad_models.append(
            ApplicabilityDomainArtifact.load(ad_subdir)
            if os.path.isdir(ad_subdir)
            else None
        )
    return descriptors, models, ad_models


def _read_task(model_dir) -> str:
    """The checkpoint's task; one without the key predates regression: classification."""
    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return "classification"
    return read_json(meta_path).get("task") or "classification"


def _check_task(model_dir, expected):
    """Refuse a checkpoint of the other task with a pointer to the right loader."""
    if not os.path.isfile(os.path.join(model_dir, "metadata.json")):
        return
    task = _read_task(model_dir)
    if task != expected:
        right = "LazyRegressorQSAR" if task == "regression" else "LazyClassifierQSAR"
        raise ValueError(
            f"{model_dir!r} is a {task} checkpoint; load it with {right}.load "
            "(or LazyQSAR.load, which dispatches on the checkpoint's task)."
        )


@contextmanager
def _unpacked(model_dir):
    """Yield a directory for *model_dir*, unzipping a ``.zip`` to scratch first.

    The scratch copy is removed once the caller is done: onnxruntime reads a graph into
    memory when the session is constructed, so the sessions outlive the files.
    """
    scratch = None
    try:
        if model_dir.endswith(".zip"):
            scratch, model_dir = unpack_to_scratch(model_dir)
        yield model_dir
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def _save_maybe_zipped(model_dir, write) -> str:
    """Run ``write(directory)``; for a ``.zip`` path, write to its stem and zip it."""
    if not model_dir.endswith(".zip"):
        write(model_dir)
        return model_dir
    model_dir = model_dir[:-4]
    write(model_dir)
    shutil.make_archive(model_dir, "zip", model_dir)
    shutil.rmtree(model_dir, ignore_errors=True)
    return model_dir + ".zip"


# ---------------------------------------------------------------------------
# Prediction over combine
# ---------------------------------------------------------------------------


class _EnsemblePredictMixin:
    """Combine per-descriptor channels with :func:`combine`, cached per SMILES list.

    Subclasses supply :meth:`_channels` -- the only thing that genuinely differs between
    the fit-time estimator (in-memory models) and the ONNX artifact wrapper (sessions) --
    and :meth:`_wanted_outputs`. Weighting, the output formulas and the diagnostics table
    are identical, so asking for several outputs featurizes once.
    """

    def _channels(self, smiles_list):
        """Return ``(Y, R, S, A, spec)``; ``R``, ``S`` and ``A`` may be ``None``."""
        raise NotImplementedError

    def _wanted_outputs(self, spec) -> tuple:
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
            result = combine(Y, R, S, A, spec=spec, outputs=self._wanted_outputs(spec))
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


class _ArtifactWrapperBase(_EnsemblePredictMixin):
    """A checkpoint loaded for inference: featurizers, ONNX artifacts and AD per descriptor.

    *spec* is already sliced to *active_indices*, the descriptor positions this model
    scores; the per-descriptor lists stay index-aligned with ``descriptor_types``.
    """

    task = ""

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

    def _wanted_channels(self) -> set:
        return {"y", "r", "a"}

    def _channels(self, smiles_list):
        if not smiles_list:
            # Scoring nothing is a legitimate request -- an empty shard, a query whose
            # every row was filtered upstream -- and `combine` handles zero rows.
            return _empty_channels(len(self.active_indices)) + (self.spec,)

        # Featurize and score in chunks rather than transforming the whole list first.
        # A million compounds against a 2048-dimensional descriptor is ~8 GB of float32,
        # and this is the entry point used to score large libraries from Python.
        chunk_size = get_chunk_size()
        want = self._wanted_channels()
        y_hats, score_preds, rank_preds, ad_scores = [], [], [], []
        for i in self.active_indices:
            ad = self.ad_artifacts[i] if self.ad_artifacts is not None else None
            channels = score_smiles_chunkwise(
                self.descriptors[i],
                self.artifacts[i],
                ad,
                smiles_list,
                chunk_size,
                want=want,
                logger=logger,
            )
            y_hats.append(channels.y)
            score_preds.append(channels.s)
            rank_preds.append(channels.r)
            if ad is not None:
                ad_scores.append(channels.a)
        return _stack_channels(y_hats, rank_preds, score_preds, ad_scores) + (
            self.spec,
        )


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


class _QSARBase(_EnsemblePredictMixin):
    """Descriptor selection, one model + applicability domain per descriptor, pruning.

    A subclass names its ``task`` and provides:

    - ``_model_class()``: the agnostic model fitted per descriptor;
    - ``_skill_attrs``: the attribute names the per-descriptor ``(oof, train, quality,
      proxy)`` skills are stored under (``oof_aucs_``... or ``oof_rhos_``...);
    - ``_quality(oof, train)``, ``_prune_floor``/``_prune_gap`` and ``_metric_label``;
    - ``_rank_error_curve(model, X, y, channels)``: the reliability curve, or None.
    """

    task = ""
    _skill_attrs = ("", "", "", "")
    _metric_label = "AUC"
    _prune_floor = 0.0
    _prune_gap = 0.0

    def __init__(self, mode: str = "slow"):
        assert mode in ("fast", "slow"), (
            f"Mode '{mode}' not recognized. Choose from 'fast' or 'slow'."
        )
        self.mode = mode
        self.descriptor_types = DESCRIPTORS_MODE[mode]
        self.descriptors = []  # populated in fit() after the descriptor screen
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

    def _active_indices(self):
        mask = getattr(self, "active_descriptors_", None) or [True] * len(
            self.descriptor_types
        )
        active = [i for i, a in enumerate(mask) if a]
        return active or list(range(len(self.descriptor_types)))

    # ------------------------------------------------------------------ hooks

    def _model_class(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def _model_skill(self, model):  # pragma: no cover - abstract
        """``(oof, train)`` skill of one fitted descriptor model."""
        raise NotImplementedError

    def _quality(self, oof, train):  # pragma: no cover - abstract
        raise NotImplementedError

    def _rank_error_curve(self, model, X, y, channels):  # pragma: no cover
        raise NotImplementedError

    def _before_descriptor_fits(self) -> None:
        """Checks that need the selected descriptors but must precede any training."""

    def _log_descriptor_skill(self, name, oof, train, quality, ad) -> None:
        def fmt(value):
            return "unknown" if value is None else f"{value:.4f}"

        logger.info(
            f"[{name}] OOF {self._metric_label}={fmt(oof)}  "
            f"train {self._metric_label}={fmt(train)}  quality={fmt(quality)}  "
            f"AD comps={ad.pca_.n_components_}"
        )

    # ------------------------------------------------------------------ fit

    def _select_descriptors(self, smiles_list, y, precomputed):
        """Run the descriptor screen and prime the feature cache with its matrices."""
        from .descriptors.portfolio import DescriptorPortfolio

        applicable = DescriptorPortfolio(self.mode).select(
            smiles_list, y=y, precomputed=precomputed, task=self.task
        )
        self.descriptor_types = [name for name, _, _, _ in applicable]
        self.descriptors = [desc for _, desc, _, _ in applicable]
        proxy = [score for _, _, _, score in applicable]
        smiles_hash = self._smiles_hash(smiles_list)
        for i, (_, _, X, _) in enumerate(applicable):
            if X is not None:
                self._feature_cache[(i, smiles_hash)] = X
        return proxy

    def _fit_descriptors(self, smiles_list, y, proxy):
        """One model and one applicability domain per descriptor, then pruning.

        Returns ``(oof_channels, train_ad, desc_rows)``: each descriptor's out-of-fold
        channels and training-set AD scores (for the pooled out-of-fold statistics) and
        the rows of the descriptor table.
        """
        from .applicability import ApplicabilityDomain

        self._before_descriptor_fits()
        model_cls = self._model_class()
        self.models, self.ad_models = [], []
        oofs, trains, qualities = [], [], []
        self._rank_error_curves_ = []
        self.ad_hard_cutoffs_ = []
        oof_channels, train_ad, desc_rows = [], [], []

        for i, name in enumerate(self.descriptor_types):
            t0 = time.perf_counter()
            X = self._transform_cached(i, smiles_list)
            feat_time = time.perf_counter() - t0
            sparsity = float((X == 0).mean())
            logger.info(
                f"[{name}] p={X.shape[1]:,}  sparsity={sparsity:.3f}  "
                f"feat_time={feat_time:.1f}s"
            )

            model = model_cls()
            model.fit(X=X, y=y)
            self.models.append(model)
            channels = model.oof_channels(X=X)
            oof_channels.append(channels)
            self._rank_error_curves_.append(
                self._rank_error_curve(model, X, y, channels)
            )

            oof, train = self._model_skill(model)
            quality = self._quality(oof, train)
            oofs.append(oof)
            trains.append(train)
            qualities.append(quality)

            ad = ApplicabilityDomain()
            ad.fit(X)
            self.ad_models.append(ad)
            scores = ad.score(X)
            train_ad.append(scores)
            self.ad_hard_cutoffs_.append(
                float(np.percentile(scores, AD_HARD_CUTOFF_PERCENTILE))
            )
            self._log_descriptor_skill(name, oof, train, quality, ad)
            desc_rows.append(
                {
                    "name": name,
                    "n_features": X.shape[1],
                    "sparsity": sparsity,
                    "feat_time": feat_time,
                    "ad_n_components": ad.pca_.n_components_,
                    "ad_cal_min": float(ad.cal_knots_[0]),
                    "ad_cal_max": float(ad.cal_knots_[-1]),
                    "proxy_auc": proxy[i],
                    "train_auc": train,
                    "quality_auc": quality,
                }
            )

        for attr, values in zip(self._skill_attrs, (oofs, trains, qualities, proxy)):
            setattr(self, attr, values)

        # Pruning on out-of-fold skill: below the floor, or too far below the best. An
        # unknown skill keeps its descriptor (a failed measurement says nothing about the
        # descriptor), and pruning everything falls back to keeping all.
        known = [s for s in oofs if s is not None]
        best = max(known) if known else 0.0
        mask = [
            True
            if s is None
            else (s >= self._prune_floor and s >= best - self._prune_gap)
            for s in oofs
        ]
        self.active_descriptors_ = mask if any(mask) else [True] * len(mask)
        for row, active in zip(desc_rows, self.active_descriptors_):
            row["active"] = active
        return oof_channels, train_ad, desc_rows

    # ------------------------------------------------------------------ save

    def _save_descriptor_dirs(self, model_dir) -> None:
        """One sub-directory per descriptor: model, featurizer, applicability domain."""
        for i, name in enumerate(self.descriptor_types):
            sub = os.path.join(model_dir, name)
            os.makedirs(sub, exist_ok=True)
            logger.debug(f"Saving model and descriptor to {sub}")
            self.models[i].save(sub)
            self.descriptors[i].save(sub)
            if self.ad_models:
                self.ad_models[i].save(os.path.join(sub, "applicability_domain"))

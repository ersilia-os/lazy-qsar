import os

import h5py
import numpy as np

from .artifacts.classifier import LazyClassifierArtifact
from .artifacts.regressor import LazyRegressorArtifact
from .utils.logging import logger
from .utils.ranking import (
    DECISION_CUTOFF_SOURCE,
    NO_REFERENCE_MESSAGE as _NO_REFERENCE_MESSAGE,
)


def _load_h5(h5_file: str, h5_idxs=None) -> np.ndarray:
    with h5py.File(h5_file, "r") as f:
        keys = list(f.keys())
        for candidate in ("X", "data", "values", "Values"):
            if candidate in keys:
                return (
                    f[candidate][:].astype("float32")
                    if h5_idxs is None
                    else f[candidate][h5_idxs].astype("float32")
                )
        raise ValueError(f"No recognised dataset key in {h5_file!r}. Found: {keys}")


def _iter_h5(h5_file: str, chunk_size: int = 4096):
    """Yield float32 row blocks of an Ersilia ``.h5``.

    Beside :func:`_load_h5` rather than replacing it: a reference library is 50,000 rows
    wide enough to cost hundreds of megabytes, and it is consumed row by row, so there is
    no reason to hold it whole. The yielded block is reused between iterations -- copy it
    if you keep it.
    """
    with h5py.File(h5_file, "r") as f:
        keys = list(f.keys())
        for candidate in ("X", "data", "values", "Values"):
            if candidate in keys:
                dset = f[candidate]
                buf = np.empty((chunk_size, dset.shape[1]), dtype="float32")
                for start in range(0, dset.shape[0], chunk_size):
                    end = min(start + chunk_size, dset.shape[0])
                    view = buf[: end - start]
                    view[:] = dset[start:end]
                    yield view
                return
        raise ValueError(f"No recognised dataset key in {h5_file!r}. Found: {keys}")


_FLOAT32_MAX = float(np.finfo(np.float32).max)


def _checked_features(X):
    """Refuse feature values the ONNX graphs cannot represent; returns *X* unchanged.

    Every model runs as a float32 ONNX graph. NaN is fine -- it is imputed exactly as it
    was at fit time -- but ``inf`` is not imputed, and a finite float64 beyond the
    float32 range silently *becomes* ``inf`` in the cast. Either would yield a prediction
    for a value the model never represented, so both are rejected here, with the count,
    rather than turned into a plausible-looking number downstream.
    """
    data = X.data if hasattr(X, "toarray") else np.asarray(X)
    if not np.issubdtype(data.dtype, np.floating):
        return X
    with np.errstate(invalid="ignore"):
        n_inf = int(np.isinf(data).sum())
        n_big = int((np.abs(data[np.isfinite(data)]) > _FLOAT32_MAX).sum())
    if n_inf or n_big:
        parts = []
        if n_inf:
            parts.append(f"{n_inf} infinite value{'s' if n_inf != 1 else ''}")
        if n_big:
            parts.append(
                f"{n_big} value{'s' if n_big != 1 else ''} beyond the float32 range "
                f"(|x| > {_FLOAT32_MAX:.3g})"
            )
        raise ValueError(
            f"Feature matrix has {' and '.join(parts)}. Models run as float32 "
            "ONNX graphs, which cannot represent them; replace them with NaN (imputed "
            "like any missing value) or clip them before fitting or predicting."
        )
    return X


# Re-exported, not redefined: there used to be a second, divergent copy here, and the
# inference artifacts raised whichever of the two their module happened to import.
NO_REFERENCE_MESSAGE = _NO_REFERENCE_MESSAGE


def _resolve_X(X, h5_file, h5_idxs):
    """The feature matrix from *X* or an Ersilia ``.h5``, checked for ONNX-safe values."""
    if X is None:
        if h5_file is None:
            raise ValueError("Provide either X or h5_file.")
        X = _load_h5(h5_file, h5_idxs)
    return _checked_features(X)


class _AgnosticBase:
    """Fit-on-features wrapper around an assembler; saves and loads its ONNX checkpoint.

    A subclass names its ``task`` and the artifact class a checkpoint loads as.
    """

    task = ""
    _artifact_class = None

    def _oof_percentile(self, X=None, h5_file=None, h5_idxs=None) -> np.ndarray:
        """Percentile against this model's own out-of-fold distribution.

        Internal: the per-sample reliability signal the QSAR ensemble weights descriptors
        by. Relative to this model's training data, so not comparable with a position
        against the reference library, and deliberately not public.
        """
        return self._model._oof_percentile(_resolve_X(X, h5_file, h5_idxs))

    def oof_channels(self, X=None, h5_file=None, h5_idxs=None):
        """Out-of-fold channels on the training rows, or None.

        Fit-time only, and only on the object that did the fitting -- a loaded checkpoint
        has no out-of-fold predictions.
        """
        X = _resolve_X(X, h5_file, h5_idxs)
        getter = getattr(self._model, "oof_channels", None)
        return getter(X) if getter is not None else None

    def save(self, model_dir: str) -> str:
        """Write the checkpoint; a path ending in ``.zip`` writes a zip archive."""
        from .qsar_base import _save_maybe_zipped

        def write(directory):
            logger.info(f"Saving model to {directory!r}")
            os.makedirs(directory, exist_ok=True)
            self._model.save(directory)

        out = _save_maybe_zipped(model_dir, write)
        logger.success(f"Model saved → {out}")
        return out

    @classmethod
    def load(cls, model_dir: str):
        """Load a saved checkpoint as its numpy + onnxruntime artifact.

        Unpacked into scratch, not alongside the archive. Sessions are built before the
        scratch copy goes, and onnxruntime holds the graph in memory, so they outlive it.
        """
        from .qsar_base import _unpacked

        with _unpacked(model_dir) as path:
            if not os.path.isfile(os.path.join(path, "metadata.json")):
                raise NotImplementedError(
                    "Loading a raw (non-ONNX) model is not supported. Every checkpoint "
                    "`save` writes carries metadata.json, so this is reachable only for "
                    "a hand-assembled directory."
                )
            logger.info(f"Loading ONNX artifact from {path!r}")
            artifact = cls._artifact_class.load(path)
            logger.success(f"Artifact loaded from {path!r}")
            return artifact


class LazyClassifier(_AgnosticBase):
    """
    Descriptor-agnostic binary classifier.

    Accepts pre-computed feature arrays (X) or Ersilia .h5 files directly.
    Wraps the internal assembler and saves/loads via ONNX.
    """

    task = "classification"
    _artifact_class = LazyClassifierArtifact

    def __init__(
        self,
        calibrated: bool = True,
        max_rounds: int | None = None,
        max_imbalance_ratio: int = 100,
    ):
        self._model = None
        self.calibrated = calibrated
        self.max_rounds = max_rounds
        self.max_imbalance_ratio = max_imbalance_ratio

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X=None,
        y=None,
        h5_file=None,
        h5_idxs=None,
        reference_X=None,
        reference_h5_file=None,
    ):
        """Fit, optionally against a reference library so that `rank` becomes available.

        ``reference_X`` is a ``(n_reference, n_features)`` matrix: the descriptors of the
        molecules in :func:`lazyqsar.reference.reference_smiles`, computed with the same
        featurizer as *X* and in that order. ``reference_h5_file`` is the same thing as an
        Ersilia ``.h5``, read in chunks so a large one is never held whole.

        Without one the model fits normally and every output works except ``rank``, which
        has nothing to be a position against and raises.
        """
        # Lazy import so inference-only environments do not need fit dependencies.
        from .assemblers.classifier import LazyClassifier as _AssemblerClassifier

        logger.rule("LazyClassifier (agnostic) — fit")
        X = _resolve_X(X, h5_file, h5_idxs)
        y = np.asarray(y, dtype=int)

        self._model = _AssemblerClassifier(
            calibrated=self.calibrated,
            max_rounds=self.max_rounds,
            max_imbalance_ratio=self.max_imbalance_ratio,
        )
        self._model.fit(X, y)
        self._build_reference_rank(X, y, reference_X, reference_h5_file)
        logger.success("LazyClassifier (agnostic) — fit complete")

    def _build_reference_rank(self, X, y, reference_X, reference_h5_file):
        """Score the caller's reference library and keep what `rank` is read against.

        The reference goes through the same ``predict_proba`` a query does, so the result is
        a position on the same scale. Chunked, because a reference matrix is the largest
        thing this class is ever handed and it is consumed row by row.
        """
        from .utils.ranking import subsample_knots

        self._model.reference_rank_knots_ = None
        self._model.reference_rank_anchors_ = None
        self._model.decision_cutoff_reference_rank_ = None
        if reference_X is None and reference_h5_file is None:
            logger.info(
                "No reference library given; `rank` will not be available on this model."
            )
            return

        if reference_h5_file is not None:
            chunks = _iter_h5(reference_h5_file)
        else:
            reference_X = np.asarray(reference_X, dtype="float32")
            chunks = (
                reference_X[i : i + 4096] for i in range(0, len(reference_X), 4096)
            )

        pooled = [self._model.predict_proba(c)[:, 1].copy() for c in chunks]
        p1 = np.concatenate(pooled) if pooled else np.empty(0)
        p1 = p1[np.isfinite(p1)]
        if p1.size == 0:
            raise ValueError(
                "The reference library produced no finite probabilities; `rank` would "
                "have nothing to be a position against."
            )
        knots = subsample_knots(np.sort(p1))
        self._model.reference_rank_knots_ = knots
        self._model.reference_rank_anchors_ = self._reference_anchors(X, y, knots)
        # Overrides the batch-mean-of-head-means balanced-accuracy cutoff the assembler set
        # during `fit`. Only now is there a reference to place the cutoff against, and a
        # cutoff on the rank scale is the one thing a mean of per-head cutoffs cannot be.
        self._set_decision_cutoff(knots)
        logger.info(
            f"Reference library scored: {p1.size:,} molecules, "
            f"probability {p1.min():.3f} to {p1.max():.3f}"
        )

    def _set_decision_cutoff(self, knots):
        """Place the decision cutoff at :data:`DECISION_RANK` on the reference rank scale.

        Mirrors :meth:`lazyqsar.qsar.LazyClassifierQSAR._build_decision_cutoff`: the cutoff
        becomes "beat 99% of drug-like chemical space", so the model calls 1% of it active by
        construction instead of whatever the out-of-fold balanced-accuracy search produced.

        Written in probability units only. ``predict`` then thresholds ``predict_proba``
        against it rather than ``predict_score`` against ``decision_cutoff_raw_``, because a
        rank-derived cutoff exists on the probability scale and nowhere else.
        """
        from .utils.ranking import (
            DECISION_RANK,
            prepare_knots,
            proba_from_reference_rank,
        )

        anchors = getattr(self._model, "reference_rank_anchors_", None) or {}
        high = anchors.get("anchor_high") if anchors.get("anchor_high_used") else None
        proba = float(
            proba_from_reference_rank(
                DECISION_RANK, prepared=prepare_knots(knots), anchor_high=high
            )
        )
        self._model.decision_cutoff_proba_ = proba
        # Its own attribute, not `decision_cutoff_rank_`: that one is the out-of-fold
        # percentile the assembler computed, and it is saved under that name.
        self._model.decision_cutoff_reference_rank_ = DECISION_RANK
        self._model.decision_cutoff_source_ = DECISION_CUTOFF_SOURCE
        clipped = float(np.clip(proba, 1e-7, 1.0 - 1e-7))
        self._model.decision_cutoff_logit_ = float(np.log(clipped / (1.0 - clipped)))
        prior = getattr(self._model, "population_prior_", None) or 0.0
        self._model.decision_cutoff_lift_ = (
            float(proba / max(prior, 1e-7)) if prior > 0 else None
        )

    def _reference_anchors(self, X, y, knots):
        """The top of the scale, pinned on this model's own known actives. See
        :func:`lazyqsar.utils.ranking.rank_from_reference`."""
        from .utils.ranking import prepare_knots, reference_anchor_table

        channels = self._model.oof_channels(X)
        if channels is None:
            return None
        p1 = np.asarray(channels[0], dtype=float)
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
                f"({last_reference:.3f})."
            )
        return {
            "anchor_low": low,
            "anchor_high": high,
            # Recorded, read by nothing: the tail table has no low anchor.
            "anchor_low_used": False,
            "anchor_high_used": bool(high_used),
            "n_actives": int(act.size),
            "n_inactives": int(inact.size),
        }

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    @property
    def oof_auc_(self) -> float:
        return self._model.oof_auc_

    @property
    def train_auc_(self) -> float:
        return self._model.train_auc_

    def predict_proba(self, X=None, h5_file=None, h5_idxs=None) -> np.ndarray:
        """Return calibrated class probabilities, shape (n, 2)."""
        X = _resolve_X(X, h5_file, h5_idxs)
        logger.debug(f"predict_proba: X={X.shape}")
        return self._model.predict_proba(X)

    def predict(
        self, X=None, h5_file=None, h5_idxs=None, cutoff: float = None
    ) -> np.ndarray:
        """Return binary labels, shape (n,).

        The cutoff depends on what the model carries. With a reference library it is
        :data:`~lazyqsar.utils.ranking.DECISION_RANK` (0.65) mapped back to a probability
        -- the point that admits 1% of drug-like space -- and the threshold is applied to
        ``predict_proba``. Without one it falls back to the pre-3.6 behaviour: a
        balanced-accuracy cutoff learned out of fold, applied to the raw score.

        Pass ``cutoff`` to override it. It is interpreted on whichever scale the model's
        own cutoff uses, so on a reference-carrying model it is a probability.
        """
        X = _resolve_X(X, h5_file, h5_idxs)
        return self._model.predict(X, cutoff=cutoff)

    def predict_lift(self, X=None, h5_file=None, h5_idxs=None) -> np.ndarray:
        """Return lift over population prior, shape (n, 2)."""
        X = _resolve_X(X, h5_file, h5_idxs)
        return self._model.predict_lift(X)

    def predict_logit(self, X=None, h5_file=None, h5_idxs=None) -> np.ndarray:
        """Return log-odds of calibrated probabilities, shape (n, 2)."""
        X = _resolve_X(X, h5_file, h5_idxs)
        return self._model.predict_logit(X)

    def predict_score(self, X=None, h5_file=None, h5_idxs=None) -> np.ndarray:
        """Return raw (pre-calibration) scores, shape (n, 2)."""
        X = _resolve_X(X, h5_file, h5_idxs)
        return self._model.predict_score(X)

    def _oof_percentile_from_proba(self, p1):
        """Out-of-fold percentile of probabilities already computed, or ``None``.

        The same lookup :meth:`_oof_percentile` performs after re-running every
        preprocessor, head and gate to get the probability back; callers that already
        hold it skip that second pass. ``None`` when the model kept no out-of-fold
        reference (the fallback then averages the batches' own percentiles).
        """
        from .utils.ranking import prepare_knots, rank_from_knots

        model = self._model
        prepared = getattr(model, "_oof_percentile_prepared_", None)
        if prepared is None:
            knots = getattr(model, "oof_percentile_knots_", None)
            if knots is None or not len(knots):
                return None
            prepared = model._oof_percentile_prepared_ = prepare_knots(knots)
        return rank_from_knots(np.asarray(p1, dtype=np.float64), prepared=prepared)

    def predict_rank(self, X=None, h5_file=None, h5_idxs=None) -> np.ndarray:
        """Position against the reference library given at fit, shape (n, 2).

        Anchored on the reference's upper tail: 0.50 is the top 10% of it, 0.65 the top 1%,
        0.75 the top 0.1%. Past p99.9 the scale is pinned on this model's own known actives.
        See :func:`lazyqsar.utils.ranking.rank_from_reference`.

        Raises when no reference was given. What earlier versions returned here was a
        percentile against this model's own training distribution -- a different quantity
        that is not comparable with a position against drug-like chemical space, and that
        would be read as one. It survives as :meth:`_oof_percentile`, where it weights the
        ensemble.
        """
        from .utils.ranking import prepare_knots, rank_from_reference

        knots = getattr(self._model, "reference_rank_knots_", None)
        if knots is None or not len(knots):
            raise ValueError(NO_REFERENCE_MESSAGE)
        X = _resolve_X(X, h5_file, h5_idxs)
        anchors = getattr(self._model, "reference_rank_anchors_", None) or {}
        pair = (
            anchors.get("anchor_low") if anchors.get("anchor_low_used") else None,
            anchors.get("anchor_high") if anchors.get("anchor_high_used") else None,
        )
        rank_1 = rank_from_reference(
            self._model.predict_proba(X)[:, 1],
            prepared=prepare_knots(np.asarray(knots, dtype=float)),
            anchors=pair if any(a is not None for a in pair) else None,
        )
        return np.column_stack([1 - rank_1, rank_1])


class LazyRegressor(_AgnosticBase):
    """
    Descriptor-agnostic regressor.

    Accepts pre-computed feature arrays (X) or Ersilia .h5 files directly, with a
    continuous target. Wraps the regression assembler and saves/loads via ONNX.
    Predictions are point estimates in the target's own units.
    """

    task = "regression"
    _artifact_class = LazyRegressorArtifact

    def __init__(self, max_rounds: int | None = None, max_batch_size: int = 100_000):
        self._model = None
        self.max_rounds = max_rounds
        self.max_batch_size = max_batch_size

    def fit(self, X=None, y=None, h5_file=None, h5_idxs=None):
        """Fit on ``(X, y)``; ``y`` is a finite, non-constant numeric target."""
        from .assemblers.regressor import LazyRegressor as _AssemblerRegressor

        logger.rule("LazyRegressor (agnostic) — fit")
        X = _resolve_X(X, h5_file, h5_idxs)
        y = np.asarray(y, dtype=float)

        self._model = _AssemblerRegressor(
            max_batch_size=self.max_batch_size, max_rounds=self.max_rounds
        )
        self._model.fit(X, y)
        logger.success("LazyRegressor (agnostic) — fit complete")

    @property
    def oof_rho_(self):
        return self._model.oof_rho_

    @property
    def train_rho_(self):
        return self._model.train_rho_

    def predict(self, X=None, h5_file=None, h5_idxs=None) -> np.ndarray:
        """Return predicted values, shape (n,)."""
        X = _resolve_X(X, h5_file, h5_idxs)
        logger.debug(f"predict: X={X.shape}")
        return self._model.predict(X)

    def predict_interval(
        self, X=None, h5_file=None, h5_idxs=None, level: float = 0.9
    ) -> tuple:
        """Prediction interval ``(lower, upper)`` at coverage *level*, each ``(n,)``.

        Split-conformal on the ensemble's pooled out-of-fold residuals (see
        :mod:`lazyqsar.ensemble.intervals`); about a fraction *level* of molecules like
        the training set fall inside it.
        """
        return self._model.predict_interval(_resolve_X(X, h5_file, h5_idxs), level)

    def _oof_percentile_from_value(self, values):
        return self._model._oof_percentile_from_value(values)


def load(model_dir: str):
    """Load any saved LazyQSAR checkpoint, dispatching on its task and kind.

    A SMILES-level checkpoint (``LazyQSAR`` / ``LazyClassifierQSAR`` /
    ``LazyRegressorQSAR``) has one sub-directory per descriptor; a feature-level one
    (``LazyClassifier`` / ``LazyRegressor``) has ``batch_*`` sub-directories. Either loads
    as the inference object for its task.
    """
    from .qsar_base import _descriptor_dirs, _read_task, _unpacked

    with _unpacked(model_dir) as path:
        task = _read_task(path)
        smiles_level = bool(_descriptor_dirs(path))
        if smiles_level:
            from .qsar import LazyQSAR

            return LazyQSAR.load(path)
        cls = LazyRegressor if task == "regression" else LazyClassifier
        return cls.load(path)

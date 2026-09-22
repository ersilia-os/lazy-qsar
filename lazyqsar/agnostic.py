import os
import shutil

import h5py
import numpy as np

from .artifacts.classifier import LazyClassifierArtifact
from .utils.logging import logger


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


NO_REFERENCE_MESSAGE = (
    "predict_rank needs a reference library. `rank` is a position against a fixed set of "
    "drug-like molecules, and this entry point takes a descriptor matrix -- it never sees "
    "the molecules, so it cannot featurize that set itself.\n"
    "Fit with `reference_X=` (or `reference_h5_file=`): featurize the molecules from "
    "`lazyqsar.reference.reference_smiles()`, in that order, and pass the matrix.\n"
    "Or use predict_proba, which needs no reference."
)


class LazyClassifier:
    """
    Descriptor-agnostic binary classifier.

    Accepts pre-computed feature arrays (X) or Ersilia .h5 files directly.
    Wraps the internal assembler and saves/loads via ONNX.
    """

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

        if X is None:
            if h5_file is None:
                raise ValueError("Provide either X or h5_file.")
            logger.info(f"Loading features from {h5_file!r}")
            X = _load_h5(h5_file, h5_idxs)

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
        logger.info(
            f"Reference library scored: {p1.size:,} molecules, "
            f"probability {p1.min():.3f} to {p1.max():.3f}"
        )

    def _reference_anchors(self, X, y, knots):
        """The tails, pinned on this model's own known molecules. See
        :func:`lazyqsar.utils.ranking.rank_from_reference`."""
        from .utils.ranking import prepare_knots

        channels = self._model.oof_channels(X)
        if channels is None:
            return None
        p1 = np.asarray(channels[0], dtype=float)
        y = np.asarray(y).ravel()
        if len(y) != len(p1):
            return None

        vals, midranks = prepare_knots(knots)
        q1 = float(np.interp(0.25, midranks, vals))
        q3 = float(np.interp(0.75, midranks, vals))
        act, inact = p1[y == 1], p1[y == 0]
        high = float(np.percentile(act, 95)) if act.size else None
        low = float(np.percentile(inact, 5)) if inact.size else None
        high_used = high is not None and q3 < high < 1.0
        low_used = low is not None and 0.0 < low < q1
        if high is not None and not high_used:
            logger.warning(
                f"Upper rank anchor unused: p95 of the out-of-fold actives ({high:.3f}) "
                f"does not exceed the reference's third quartile ({q3:.3f})."
            )
        if low is not None and not low_used:
            logger.warning(
                f"Lower rank anchor unused: p05 of the out-of-fold inactives ({low:.3f}) "
                f"is not below the reference's first quartile ({q1:.3f})."
            )
        return {
            "anchor_low": low,
            "anchor_high": high,
            "anchor_low_used": bool(low_used),
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
        if X is None:
            X = _load_h5(h5_file, h5_idxs)
        logger.debug(f"predict_proba: X={X.shape}")
        return self._model.predict_proba(X)

    def predict(
        self, X=None, h5_file=None, h5_idxs=None, cutoff: float = None
    ) -> np.ndarray:
        """Return binary labels using the OOF-learned decision cutoff, shape (n,)."""
        if X is None:
            X = _load_h5(h5_file, h5_idxs)
        return self._model.predict(X, cutoff=cutoff)

    def predict_lift(self, X=None, h5_file=None, h5_idxs=None) -> np.ndarray:
        """Return lift over population prior, shape (n, 2)."""
        if X is None:
            X = _load_h5(h5_file, h5_idxs)
        return self._model.predict_lift(X)

    def predict_logit(self, X=None, h5_file=None, h5_idxs=None) -> np.ndarray:
        """Return log-odds of calibrated probabilities, shape (n, 2)."""
        if X is None:
            X = _load_h5(h5_file, h5_idxs)
        return self._model.predict_logit(X)

    def predict_score(self, X=None, h5_file=None, h5_idxs=None) -> np.ndarray:
        """Return raw (pre-calibration) scores, shape (n, 2)."""
        if X is None:
            X = _load_h5(h5_file, h5_idxs)
        return self._model.predict_score(X)

    def _oof_percentile(self, X=None, h5_file=None, h5_idxs=None) -> np.ndarray:
        """Percentile against this model's own out-of-fold distribution, shape (n, 2).

        Internal. This is the weighting signal -- ``LazyClassifierQSAR`` blends descriptors
        by how reliable each one looks at a given percentile -- and it is deliberately not a
        public rank: it is relative to this model's training data, so it is not comparable
        with a position against the reference library, and would be read as one.
        """
        if X is None:
            X = _load_h5(h5_file, h5_idxs)
        return self._model._oof_percentile(X)

    def predict_rank(self, X=None, h5_file=None, h5_idxs=None) -> np.ndarray:
        """Position against the reference library given at fit, shape (n, 2).

        Between the reference's first and third quartiles this is the exact percentile, so
        0.25, 0.50 and 0.75 are its quartiles; outside, the tails are pinned on this model's
        own known molecules. See :func:`lazyqsar.utils.ranking.rank_from_reference`.

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
        if X is None:
            X = _load_h5(h5_file, h5_idxs)
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

    def oof_channels(self, X=None, h5_file=None, h5_idxs=None):
        """Out-of-fold ``(proba, rank, score)`` on the training rows, or None.

        Fit-time only, and only on the object that did the fitting -- a loaded checkpoint
        has no out-of-fold predictions. Used to build the pooled rank reference; see
        :meth:`lazyqsar.assemblers.classifier.LazyClassifier.oof_channels`.
        """
        if X is None:
            X = _load_h5(h5_file, h5_idxs)
        getter = getattr(self._model, "oof_channels", None)
        return getter(X) if getter is not None else None

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, model_dir: str) -> str:
        if model_dir.endswith(".zip"):
            zip_out = True
            model_dir = model_dir[:-4]
        else:
            zip_out = False
        logger.info(f"Saving model to {model_dir!r}")
        os.makedirs(model_dir, exist_ok=True)
        self._model.save(model_dir)
        if zip_out:
            shutil.make_archive(model_dir, "zip", model_dir)
            shutil.rmtree(model_dir)
            logger.success(f"Model saved → {model_dir}.zip")
            return model_dir + ".zip"
        logger.success(f"Model saved → {model_dir}")
        return model_dir

    @classmethod
    def load(cls, model_dir: str):
        if model_dir.endswith(".zip"):
            base_dir = model_dir[:-4]
            if os.path.exists(base_dir):
                shutil.rmtree(base_dir)
            shutil.unpack_archive(model_dir, base_dir)
            model_dir = base_dir
        # ONNX / artifact path
        if os.path.isfile(os.path.join(model_dir, "metadata.json")):
            logger.info(f"Loading ONNX artifact from {model_dir!r}")
            artifact = LazyClassifierArtifact.load(model_dir)
            logger.success(f"Artifact loaded from {model_dir!r}")
            return artifact
        # Raw assembler path
        from .assemblers.classifier import LazyClassifier as _AssemblerClassifier

        obj = cls.__new__(cls)
        obj._model = _AssemblerClassifier()
        raise NotImplementedError(
            "Loading a raw (non-ONNX) LazyClassifier is not yet supported."
        )


class LazyRegressor:
    """Placeholder — not yet implemented."""

    def __init__(self, **kwargs):
        raise NotImplementedError("LazyRegressor is not yet implemented.")

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

    def fit(self, X=None, y=None, h5_file=None, h5_idxs=None):
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
        logger.success("LazyClassifier (agnostic) — fit complete")

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

    def predict_rank(self, X=None, h5_file=None, h5_idxs=None) -> np.ndarray:
        """Return rank quantiles relative to the training OOF distribution, shape (n, 2)."""
        if X is None:
            X = _load_h5(h5_file, h5_idxs)
        return self._model.predict_rank(X)

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

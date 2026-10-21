"""
Inference-only artifacts for BaseSVCClassifier and BaseSVR.

Loads an svc.onnx written by BaseSVCClassifier.save(), or an svr.onnx written
by BaseSVR.save(). Only requires numpy and onnxruntime — no sklearn or SVC.
"""

import numpy as np

from lazyqsar.artifacts._head import HeadArtifact
from lazyqsar.utils.logging import logger

# The probe BaseSVCClassifier.save() records decision_function on (see
# _resolve_score_column).
_SCORE_CANARY_SEED = 20240917
_SCORE_CANARY_ROWS = 8


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


class SVCArtifact(HeadArtifact):
    """
    Inference-only loader for a model saved by BaseSVCClassifier.save().

    The ONNX model outputs raw decision function scores.  This artifact applies:
      1. sigmoid(score)          → pre-calibration [0,1] probability (predict_score)
      2. calibrator(sigmoid)     → calibrated probability (run / predict_proba)
    """

    _stem = "svc"

    def _after_load(self) -> None:
        self._score_col = self._resolve_score_column()

    def _resolve_score_column(self) -> int:
        """Return which ONNX score column carries sklearn's ``decision_function``.

        ``_to_onnx`` exports the two SVC flavours through two different skl2onnx
        converters -- ``raw_scores=True`` for LinearSVC, ``zipmap=False`` for kernel SVC --
        and they do not order the two mutually-negated score columns the same way. A
        hardcoded column is therefore correct for one flavour and inverts the other,
        silently, for every prediction the head makes.

        Resolution is primarily against the graph's OWN label output: sklearn's binary SVC
        predicts class 1 exactly when ``decision_function > 0``, so the score column whose
        sign reproduces ``output[0]`` is the decision function. That needs nothing from the
        metadata, which is what makes it work for models saved before the canary existed.
        The canary, when present, is used as a cross-check.
        """
        n_features = int(self.metadata["n_features_in"])
        canary = self.metadata.get("score_canary")
        seed = int(canary["seed"]) if canary else _SCORE_CANARY_SEED
        n_rows = int(canary["n_rows"]) if canary else _SCORE_CANARY_ROWS
        probe = np.random.default_rng(seed).standard_normal((n_rows, n_features))

        outputs = self._session.run(
            None, {self._input_name: np.asarray(probe, dtype=np.float32)}
        )
        raw = np.asarray(outputs[1], dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 2:
            return 0

        # Primary: agreement between each column's sign and the exported labels.
        labels = np.asarray(outputs[0]).ravel().astype(int)
        agree = [float((labels == (raw[:, c] > 0).astype(int)).mean()) for c in (0, 1)]
        by_label = int(np.argmax(agree))
        # The columns are exact negations, so a usable probe separates them completely.
        # Anything less means every probe row sat on the decision boundary.
        label_decisive = abs(agree[0] - agree[1]) > 0.5

        if canary is not None:
            expected = np.asarray(canary["decision"], dtype=np.float64)
            # Relative tolerance: decision values reach O(100) on LinearSVC heads, where a
            # float32 round-trip costs far more than a fixed 1e-3 would allow.
            scale = max(1.0, float(np.abs(expected).max()))
            errors = [np.abs(raw[:, c] - expected).max() / scale for c in (0, 1)]
            by_canary = int(np.argmin(errors))
            if label_decisive and by_canary != by_label:
                logger.warning(
                    f"SVC score canary picks column {by_canary} but the exported labels "
                    f"pick column {by_label}; trusting the labels."
                )
                return by_label
            if errors[by_canary] > 1e-3:
                logger.warning(
                    f"SVC score canary did not match either ONNX column "
                    f"(best relative residual {errors[by_canary]:.3g})."
                )
            return by_canary

        if label_decisive:
            return by_label

        # Degenerate probe. Fall back to the estimator type, recorded in every svc.json.
        use_linear = self.metadata.get("use_linear")
        if use_linear is None:
            logger.warning(
                "SVC artifact: could not resolve the score column from the exported "
                "labels and no use_linear flag is present; assuming column 0."
            )
            return 0
        return 1 if use_linear else 0

    def _raw_scores(self, X) -> np.ndarray:
        """Return shape-(n,) raw decision function scores from ONNX."""
        raw = np.asarray(self._outputs(X)[1], dtype=np.float64)
        if raw.ndim == 2 and raw.shape[1] > 1:
            return raw[:, self._score_col]
        return raw.ravel()

    def _score_from_outputs(self, outputs) -> np.ndarray:
        raw = np.asarray(outputs[1], dtype=np.float64)
        df = (
            raw[:, self._score_col]
            if raw.ndim == 2 and raw.shape[1] > 1
            else raw.ravel()
        )
        p_raw = _sigmoid(df)
        return np.column_stack([1 - p_raw, p_raw])


class SVRArtifact(HeadArtifact):
    """
    Inference-only loader for a model saved by BaseSVR.save().

    Applies the inverse target standardisation to the ONNX output.
    """

    _stem = "svr"

    def _after_load(self) -> None:
        self.task = "regression"
        self._y_mean = float(self.metadata["y_mean"])
        self._y_std = float(self.metadata["y_std"])

    def _values_from_outputs(self, outputs) -> np.ndarray:
        """Predictions in original target units, shape (n_samples,)."""
        out = np.asarray(outputs[0], dtype=np.float64).ravel()
        return out * self._y_std + self._y_mean

"""
Inference-only artifact for BaseXGBClassifier / BaseXGBRegressor.

Loads an xgboost.onnx written by BaseXGBClassifier.save() or BaseXGBRegressor.save().
Only requires numpy and onnxruntime — no xgboost or sklearn.
"""

import json
import os

import numpy as np
import onnxruntime as rt


class XGBoostArtifact:
    """Load and run a saved XGBoost ONNX model."""

    def __init__(self):
        self._session = None
        self._input_name: str = ""
        self.task: str = ""
        self.metadata: dict = {}

    @classmethod
    def load(cls, directory: str) -> "XGBoostArtifact":
        json_path = os.path.join(directory, "xgboost.json")
        onnx_path = os.path.join(directory, "xgboost.onnx")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"xgboost.json not found in {directory!r}")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"xgboost.onnx not found in {directory!r}")
        self = cls.__new__(cls)
        with open(json_path) as f:
            self.metadata = json.load(f)
        self.task = self.metadata["task"]
        self._session = rt.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._cal = self.metadata.get("calibrator", None)
        self._ranker = self.metadata.get("ranker", None)
        return self

    def run(self, X) -> np.ndarray:
        """
        Run inference on X.

        Returns
        -------
        Classification : ndarray shape (n_samples, 2) — [P(0), P(1)]
        Regression     : ndarray shape (n_samples,)
        """
        X_f32 = np.asarray(X, dtype=np.float32)
        outputs = self._session.run(None, {self._input_name: X_f32})
        if self.task == "classification":
            # onnxmltools XGBoost classifier: output named "probabilities"
            prob_output = next(
                o
                for o, meta in zip(outputs, self._session.get_outputs())
                if meta.name == "probabilities"
            )
            proba = np.asarray(prob_output, dtype=np.float64)
            if self._cal is not None:
                raw_p1 = proba[:, 1]
                if self._cal["method"] == "isotonic":
                    p1 = np.clip(
                        np.interp(
                            raw_p1, self._cal["X_thresholds"], self._cal["y_thresholds"]
                        ),
                        0,
                        1,
                    )
                else:  # platt
                    A, B = self._cal["coef"], self._cal["intercept"]
                    p1 = 1.0 / (1.0 + np.exp(-(A * raw_p1 + B)))
                proba = np.column_stack([1 - p1, p1])
            return proba
        else:
            return np.asarray(outputs[0], dtype=np.float64).ravel()

    def predict_logit(self, X) -> np.ndarray:
        """Return logit of calibrated probabilities, shape (n_samples, 2)."""
        p = np.clip(self.run(X)[:, 1], 1e-7, 1.0 - 1e-7)
        logit_1 = np.log(p / (1.0 - p))
        return np.column_stack([-logit_1, logit_1])

    def predict_score(self, X) -> np.ndarray:
        """Return raw (pre-calibration) ONNX probabilities, shape (n_samples, 2)."""
        X_f32 = np.asarray(X, dtype=np.float32)
        outputs = self._session.run(None, {self._input_name: X_f32})
        prob_output = next(
            o
            for o, meta in zip(outputs, self._session.get_outputs())
            if meta.name == "probabilities"
        )
        return np.asarray(prob_output, dtype=np.float64)

    def predict_rank(self, X) -> np.ndarray:
        """Map calibrated scores to [0, 1] ranks via OOF ECDF, shape (n_samples, 2)."""
        if self._ranker is None:
            raise RuntimeError("No ranker stored in this artifact.")
        knots = np.asarray(self._ranker["knots"])
        rank_1 = np.interp(
            self.predict_score(X)[:, 1], knots, np.linspace(0.0, 1.0, len(knots))
        )
        return np.column_stack([1 - rank_1, rank_1])

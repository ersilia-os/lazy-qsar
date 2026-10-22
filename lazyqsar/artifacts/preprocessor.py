"""
Inference-only artifact for the base preprocessor.

Loads a preprocessor.onnx written by BasePreprocessor.save().
Only requires numpy and onnxruntime — no sklearn.
"""

import json
import os

import numpy as np
import onnxruntime as rt


class PreprocessorArtifact:
    """Load and run a saved preprocessor ONNX model."""

    def __init__(self):
        self._session = None
        self._input_name: str = ""
        self.metadata: dict = {}

    @classmethod
    def load(cls, directory: str) -> "PreprocessorArtifact":
        json_path = os.path.join(directory, "preprocessor.json")
        onnx_path = os.path.join(directory, "preprocessor.onnx")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"preprocessor.json not found in {directory!r}")
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"preprocessor.onnx not found in {directory!r}")
        self = cls.__new__(cls)
        with open(json_path) as f:
            self.metadata = json.load(f)
        self._session = rt.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        return self

    def run(self, X) -> np.ndarray:
        """Transform X through the preprocessor. Returns float32 array.

        Infinite values, and finite values beyond the float32 range (which the cast turns
        into infinities), are refused: the graph imputes NaN but not infinity, so they
        would produce a prediction for a value the model never represented. The same
        contract the fitting entry points enforce.
        """
        X32 = np.asarray(X, dtype=np.float32)
        n_inf = int(np.isinf(X32).sum())
        if n_inf:
            raise ValueError(
                f"Feature matrix has {n_inf} infinite or beyond-float32 value"
                f"{'s' if n_inf != 1 else ''}. Models run as float32 ONNX graphs, which "
                "cannot represent them; replace them with NaN (imputed like any missing "
                "value) or clip them before predicting."
            )
        return self._session.run(None, {self._input_name: X32})[0]

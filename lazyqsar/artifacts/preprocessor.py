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
        """Transform X through the preprocessor. Returns float32 array."""
        return self._session.run(
            None, {self._input_name: np.asarray(X, dtype=np.float32)}
        )[0]

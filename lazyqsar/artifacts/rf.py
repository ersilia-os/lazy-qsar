"""
Inference-only artifact for BaseRFClassifier / BaseRFRegressor.

Loads a randomforest.onnx written by either head's save(). Only requires numpy and
onnxruntime — no sklearn.
"""

import numpy as np

from lazyqsar.artifacts._head import HeadArtifact, proba_from_skl2onnx


class RandomForestArtifact(HeadArtifact):
    """Load and run a saved random forest ONNX model."""

    _stem = "randomforest"

    def _score_from_outputs(self, outputs) -> np.ndarray:
        return proba_from_skl2onnx(outputs[1])
